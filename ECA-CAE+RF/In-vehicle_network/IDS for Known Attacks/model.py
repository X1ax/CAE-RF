import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy import stats


CONFIG = {
    'img_size': (9, 9, 3),
    'batch_size': 64,
    'cae_epochs': 1,
    'learning_rate': 0.001,
    'k_folds': 5,
    'n_estimators': 50,
    'random_state': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_root': 'Car_Hacking_images',
    'output_dir': './results_CAE_RF',

    
    'known_classes': [0, 1, 2, 3, 4],
    'class_names': {
        0: 'DoS',
        1: 'Gear',
        2: 'Fuzzy',
        3: 'RPM',
        4: 'Normal'
    },

    
    'feature_dim': 9,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
for subdir in ['models', 'plots', 'reports', 'attention_maps']:
    os.makedirs(os.path.join(CONFIG['output_dir'], subdir), exist_ok=True)

np.random.seed(CONFIG['random_state'])
torch.manual_seed(CONFIG['random_state'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['random_state'])

print(f"🖥️  Using device: {CONFIG['device']}")



class CANDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path)
            img = np.array(img).astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            img = torch.from_numpy(img).permute(2, 0, 1)
            return img, label
        except Exception:
            return torch.zeros(3, 9, 9), label



class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs(np.log2(channels) + b) / gamma)
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        att_weights = self.sigmoid(y)
        return x * att_weights.expand_as(x), att_weights



    def __init__(self):
        super(ECACAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.eca1 = ECAModule(16)

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.eca2 = ECAModule(32)

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.eca3 = ECAModule(64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        e1 = self.encoder(x)
        e1_att, att1 = self.eca1(e1)
        e2 = self.encoder2(e1_att)
        e2_att, att2 = self.eca2(e2)
        e3 = self.encoder3(e2_att)
        encoded, att3 = self.eca3(e3)
        decoded = self.decoder(encoded)

        if return_attention:
            return decoded, encoded, [att1, att2, att3]
        return decoded, encoded

    def extract_features(self, x):
        with torch.no_grad():
            _, features = self.forward(x)
        return features


def load_image_dataset():
    image_paths = []
    labels = []

    print("📂 Scanning dataset...")
    for class_id in CONFIG['known_classes']:
        class_dir = os.path.join(CONFIG['image_root'], str(class_id))
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            print(f"  - Class {class_id} ({CONFIG['class_names'][class_id]:10s}): Found {len(files)} images")
            for img_file in files:
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_id)

    return np.array(image_paths), np.array(labels)


def get_dataloader(paths, labels, batch_size, shuffle=True):
    dataset = CANDataset(paths, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=CONFIG['num_workers'], pin_memory=True)


def train_cae(model, train_loader, val_loader, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0
        with tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False, unit="batch") as pbar:
            for imgs, _ in pbar:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss_accum / len(train_loader)

        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, _ = model(imgs)
                val_loss_accum += criterion(recon, imgs).item()

        avg_val_loss = val_loss_accum / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'models', 'best_cae.pth'))

    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('CAE Training History - CAN Dataset')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'cae_training_history.png'), dpi=300)
    plt.close()

    model.load_state_dict(torch.load(os.path.join(CONFIG['output_dir'], 'models', 'best_cae.pth')))
    return model


def extract_latent_features(model, loader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            _, encoded = model(imgs)
            features.append(encoded.view(encoded.size(0), -1).cpu().numpy())
            labels.append(lbls.numpy())
    return np.vstack(features), np.concatenate(labels)


def calculate_fpr_per_class(cm):
    
    n_classes = cm.shape[0]
    fprs = []

    
    total_samples = np.sum(cm)

    for i in range(n_classes):
        
        tp = cm[i, i]
        
        fp = np.sum(cm[:, i]) - tp
        
        fn = np.sum(cm[i, :]) - tp
       
        tn = total_samples - (tp + fp + fn)

        denominator = fp + tn
        fpr = fp / denominator if denominator > 0 else 0.0
        fprs.append(fpr)

    return np.array(fprs)



def multiclass_experiment():
    print("\n" + "="*60)
    print("📊 Multi-class Classification (5 Classes - CAN Dataset)")
    print("="*60)

    paths, labels = load_image_dataset()
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['random_state'])

    
    global_metrics = {
        'acc': [], 'macro_f1': [], 'macro_prec': [], 'macro_rec': [], 'macro_fpr': []
    }

    
    
    class_metrics_history = defaultdict(lambda: defaultdict(list))

    last_model, last_rf, last_data = None, None, {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
        print(f"\n📌 Fold {fold + 1}/{CONFIG['k_folds']}")

        X_train, X_test = paths[train_idx], paths[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, random_state=CONFIG['random_state']
        )

        train_loader = get_dataloader(X_train, y_train, CONFIG['batch_size'])
        val_loader = get_dataloader(X_val, y_val, CONFIG['batch_size'], shuffle=False)
        test_loader = get_dataloader(X_test, y_test, CONFIG['batch_size'], shuffle=False)

        model = ECACAE().to(CONFIG['device'])
        model = train_cae(model, train_loader, val_loader, CONFIG['cae_epochs'], CONFIG['device'])

        X_train_feat, y_train_feat = extract_latent_features(model, train_loader, CONFIG['device'])
        X_test_feat, y_test_feat = extract_latent_features(model, test_loader, CONFIG['device'])

        rf = RandomForestClassifier(n_estimators=CONFIG['n_estimators'], class_weight='balanced', n_jobs=-1)
        rf.fit(X_train_feat, y_train_feat)
        y_pred = rf.predict(X_test_feat)

        
        acc = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='macro')
        prec = precision_score(y_test_feat, y_pred, average='macro')
        rec = recall_score(y_test_feat, y_pred, average='macro')

        
        cm = confusion_matrix(y_test_feat, y_pred, labels=CONFIG['known_classes'])
        fprs = calculate_fpr_per_class(cm)
        macro_fpr = np.mean(fprs)

        
        global_metrics['acc'].append(acc)
        global_metrics['macro_f1'].append(f1)
        global_metrics['macro_prec'].append(prec)
        global_metrics['macro_rec'].append(rec)
        global_metrics['macro_fpr'].append(macro_fpr)

        
        p, r, f, _ = precision_recall_fscore_support(y_test_feat, y_pred, labels=CONFIG['known_classes'])

        for idx, cls_id in enumerate(CONFIG['known_classes']):
            cls_name = CONFIG['class_names'][cls_id]
            class_metrics_history[cls_name]['precision'].append(p[idx])
            class_metrics_history[cls_name]['recall'].append(r[idx])
            class_metrics_history[cls_name]['f1-score'].append(f[idx])
            class_metrics_history[cls_name]['fpr'].append(fprs[idx])

        print(f"  Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, FPR={macro_fpr:.4f}")

        last_model = model
        last_rf = rf
        last_data = {'X': X_test_feat, 'y': y_test_feat, 'pred': y_pred}

    
    print("\n" + "="*80)
    print(f"{'Final 5-Fold Global Metrics (Mean ± Std)':^80}")
    print("="*80)
    print(f"{'Metric':<20} | {'Mean':<15} | {'Std':<15} | {'95% CI':<20}")
    print("-" * 80)

    for metric_name, values in global_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=stats.sem(values)) if len(values) > 1 else (0,0)

        
        disp_name = metric_name.replace('macro_', 'Macro ').replace('acc', 'Accuracy').upper()
        print(f"{disp_name:<20} | {mean_val:.4f}          | {std_val:.4f}          | [{ci[0]:.4f}, {ci[1]:.4f}]")
    print("="*80)

    
    print("\n" + "="*100)
    print(f"{'5-Fold Classification Report (Mean ± Std)':^100}")
    print("="*100)
    header = f"{'Class':<15} | {'Precision':^20} | {'Recall':^20} | {'F1-Score':^20} | {'FPR':^20}"
    print(header)
    print("-" * 100)

    for cls_id in CONFIG['known_classes']:
        cls_name = CONFIG['class_names'][cls_id]
        metrics = class_metrics_history[cls_name]

        p_mean, p_std = np.mean(metrics['precision']), np.std(metrics['precision'])
        r_mean, r_std = np.mean(metrics['recall']), np.std(metrics['recall'])
        f_mean, f_std = np.mean(metrics['f1-score']), np.std(metrics['f1-score'])
        fpr_mean, fpr_std = np.mean(metrics['fpr']), np.std(metrics['fpr'])

        row_str = (
            f"{cls_name:<15} | "
            f"{p_mean:.4f} ± {p_std:.4f}    | "
            f"{r_mean:.4f} ± {r_std:.4f}    | "
            f"{f_mean:.4f} ± {f_std:.4f}    | "
            f"{fpr_mean:.4f} ± {fpr_std:.4f}"
        )
        print(row_str)

    print("-" * 100)

    
    with open(os.path.join(CONFIG['output_dir'], 'reports', 'final_5fold_metrics.txt'), 'w') as f:
        f.write("Final 5-Fold Global Metrics\n")
        f.write("="*50 + "\n")
        for k, v in global_metrics.items():
            f.write(f"{k}: {np.mean(v):.4f} ± {np.std(v):.4f}\n")

        f.write("\n\nPer-Class Metrics (Mean ± Std)\n")
        f.write("="*50 + "\n")
        for cls_name, metrics in class_metrics_history.items():
            f.write(f"\nClass: {cls_name}\n")
            for m_k, m_v in metrics.items():
                f.write(f"  {m_k}: {np.mean(m_v):.4f} ± {np.std(m_v):.4f}\n")

    
    cm = confusion_matrix(last_data['y'], last_data['pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CONFIG['class_names'][i] for i in CONFIG['known_classes']],
                yticklabels=[CONFIG['class_names'][i] for i in CONFIG['known_classes']])
    plt.title('Multi-class Confusion Matrix (Last Fold)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'multiclass_confusion_matrix.png'), dpi=300)
    plt.close()

    
    y_bin = label_binarize(last_data['y'], classes=CONFIG['known_classes'])
    y_score = last_rf.predict_proba(last_data['X'])

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(CONFIG['known_classes'])))

    for i, cls_id in enumerate(CONFIG['known_classes']):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{CONFIG["class_names"][cls_id]} (AUC={auc(fpr, tpr):.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves (Last Fold)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'multiclass_roc.png'), dpi=300)
    plt.close()

    return last_model, last_rf


def analyze_feature_importance_merged(model, class_ids):
    print("\n" + "="*60)
    print("🔬 Feature Importance Analysis (All 9 Features)")
    print("="*60)
    model.eval()
    criterion = nn.MSELoss()

    try:
        with open(os.path.join(CONFIG['image_root'], 'feature_columns.pkl'), 'rb') as f:
            feat_names_list = pickle.load(f)
        print(f"✓ Loaded {len(feat_names_list)} feature names from feature_columns.pkl")
    except:
        feat_names_list = ['ID'] + [f'Data{i}' for i in range(8)]
        print(f"⚠ Using default feature names")

    n_classes = len(class_ids)
    if n_classes <= 3:
        fig, axes = plt.subplots(1, n_classes, figsize=(6*n_classes, 5))
    else:
        n_cols = 3
        n_rows = (n_classes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))

    axes = axes.flatten() if n_classes > 1 else [axes]

    for i, cls in enumerate(class_ids):
        print(f"  Computing gradients for {CONFIG['class_names'][cls]}...")
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))

        if not os.path.exists(cls_dir):
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center')
            axes[i].set_title(f"{CONFIG['class_names'][cls]}")
            continue

        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
        if not files:
            axes[i].text(0.5, 0.5, 'No Images', ha='center', va='center')
            axes[i].set_title(f"{CONFIG['class_names'][cls]}")
            continue

        samples = np.random.choice(files, min(100, len(files)), replace=False)
        grad_acc = torch.zeros(3, 9, 9).to(CONFIG['device'])

        for fp in samples:
            img = Image.open(fp)
            img = np.array(img).astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CONFIG['device'])
            t.requires_grad = True
            loss = criterion(model(t)[0], t)
            model.zero_grad()
            loss.backward()
            grad_acc += t.grad.data.abs().squeeze()

        flat_grad = (grad_acc / len(samples)).cpu().numpy().flatten()

        feature_scores = defaultdict(float)
        for j in range(len(flat_grad)):
            f_idx = j % CONFIG['feature_dim']
            name = feat_names_list[f_idx] if f_idx < len(feat_names_list) else f"F_{f_idx}"
            feature_scores[name] = max(feature_scores[name], flat_grad[j])

        data_rows = [{'Feature': k, 'Score': v} for k, v in feature_scores.items()]
        df_imp = pd.DataFrame(data_rows)
        df_imp = df_imp.sort_values(by='Score', ascending=False)

        df_imp.to_csv(
            os.path.join(CONFIG['output_dir'], 'reports', f'imp_{CONFIG["class_names"][cls]}.csv'),
            index=False
        )

        sns.barplot(x='Score', y='Feature', data=df_imp, ax=axes[i], color='#1f77b4')
        axes[i].set_title(f"{CONFIG['class_names'][cls]}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Importance Score")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    base_path = os.path.join(CONFIG['output_dir'], 'plots', 'feature_importance_merged')
    for fmt in ['png', 'pdf', 'eps']:
        plt.savefig(f"{base_path}.{fmt}", dpi=600, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved merged feature importance plots.")



def visualize_saliency_9x9(model, class_ids):
    print("\n" + "="*60)
    print("A. Generating 9x9 Spatial Saliency Maps")
    print("="*60)
    model.eval()
    criterion = nn.MSELoss()

    for cls in class_ids:
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))
        if not os.path.exists(cls_dir):
            continue

        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
        if not files:
            continue

        samples = np.random.choice(files, min(20, len(files)), replace=False)
        grad_acc = torch.zeros(9, 9).to(CONFIG['device'])

        for fp in samples:
            img = Image.open(fp)
            img = np.array(img).astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CONFIG['device'])
            t.requires_grad = True
            loss = criterion(model(t)[0], t)
            model.zero_grad()
            loss.backward()
            grad_acc += t.grad.data.abs().max(dim=1)[0].squeeze()

        heatmap = (grad_acc / len(samples)).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        plt.figure(figsize=(6, 5))
        im = plt.imshow(heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f'Saliency Map (9x9)\n{CONFIG["class_names"][cls]}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(
            os.path.join(CONFIG['output_dir'], 'attention_maps', f'saliency_9x9_{CONFIG["class_names"][cls]}.png'),
            dpi=300
        )
        plt.close()


def visualize_eca_channel_heatmaps(model, class_ids):
    print("\n" + "="*60)
    print("B. Generating ECA Channel Weights Heatmaps")
    print("="*60)
    model.eval()

    for cls in class_ids:
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))
        if not os.path.exists(cls_dir):
            continue

        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.endswith('.png')]
        if not files:
            continue

        samples = np.random.choice(files, min(20, len(files)), replace=False)
        w1_acc, w2_acc, w3_acc = [], [], []

        with torch.no_grad():
            for fp in samples:
                img = Image.open(fp)
                img = np.array(img).astype(np.float32) / 255.0
                if len(img.shape) == 2:
                    img = np.stack([img]*3, axis=-1)
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CONFIG['device'])
                _, _, atts = model(t, return_attention=True)
                w1_acc.append(atts[0].squeeze().cpu().numpy())
                w2_acc.append(atts[1].squeeze().cpu().numpy())
                w3_acc.append(atts[2].squeeze().cpu().numpy())

        avg_w1 = np.mean(w1_acc, axis=0).reshape(4, 4)
        avg_w2 = np.mean(w2_acc, axis=0).reshape(4, 8)
        avg_w3 = np.mean(w3_acc, axis=0).reshape(8, 8)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        maps = [(avg_w1, "Layer 1: 16 Channels"),
                (avg_w2, "Layer 2: 32 Channels"),
                (avg_w3, "Layer 3: 64 Channels")]

        for ax, (data, title) in zip(axes, maps):
            im = ax.imshow(data, cmap='jet', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f'ECA Channel Weights - {CONFIG["class_names"][cls]}', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(CONFIG['output_dir'], 'attention_maps', f'eca_weights_{CONFIG["class_names"][cls]}.png'),
            dpi=300
        )
        plt.close()





def main():
    print("\n" + "="*60)
    print("ECA-CAE System for CAN Dataset (5-Class Classification)")
    print("="*60)

    
    model, rf = multiclass_experiment()

    
    analyze_feature_importance_merged(model, CONFIG['known_classes'])

    
    visualize_saliency_9x9(model, CONFIG['known_classes'])
    visualize_eca_channel_heatmaps(model, CONFIG['known_classes'])

    print("\n✅ All Done! Check 'results_CAN_CAE_RF' folder.")
    print(f"\n📁 Output structure:")
    print(f"  - reports/final_5fold_metrics.txt: Contains Mean ± Std of Precision, Recall, F1, FPR")


if __name__ == "__main__":

    main()
