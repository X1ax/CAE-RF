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
    roc_curve, auc,
    accuracy_score, f1_score, precision_score, recall_score
)
from scipy import stats

CONFIG = {
    'img_size': (9, 9, 3),
    'batch_size': 64,
    'cae_epochs': 50,
    'learning_rate': 0.001,
    'k_folds': 5,
    'n_estimators': 150,
    'random_state': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,
    'image_root': 'CICIDS2017_images',
    'output_dir': 'results_CAE_RF',
    'known_classes': [0, 1, 2, 3, 4, 5],
    'class_names': {
        0: 'BENIGN', 1: 'DoS', 2: 'PortScan',
        3: 'BruteForce', 4: 'WebAttack', 5: 'Bot'
    },
    'feature_dim': 80,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
for subdir in ['models', 'plots', 'reports', 'attention_maps']:
    os.makedirs(os.path.join(CONFIG['output_dir'], subdir), exist_ok=True)

np.random.seed(CONFIG['random_state'])
torch.manual_seed(CONFIG['random_state'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['random_state'])

print(f"Using device: {CONFIG['device']}")

class CICIDSDataset(Dataset):
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
                img = np.stack([img] * 3, axis=-1)
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

class ECACAE(nn.Module):
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
    image_paths, labels = [], []
    print("Scanning dataset (Known classes only)...")
    for class_id in CONFIG['known_classes']:
        class_dir = os.path.join(CONFIG['image_root'], str(class_id))
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
            print(f" - Class {class_id} ({CONFIG['class_names'][class_id]}): {len(files)} images")
            for img_file in files:
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_id)
    return np.array(image_paths), np.array(labels)

def get_dataloader(paths, labels, batch_size, shuffle=True):
    dataset = CICIDSDataset(paths, labels)
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
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False, unit="batch") as pbar:
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
            torch.save(model.state_dict(),
                       os.path.join(CONFIG['output_dir'], 'models', 'temp_best_cae.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('CAE Training History')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'cae_training_history.png'), dpi=600)
    plt.close()
    print("Saved training history to plots/")

    model.load_state_dict(
        torch.load(os.path.join(CONFIG['output_dir'], 'models', 'temp_best_cae.pth')))
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

def multiclass_experiment():
    print("\n" + "=" * 60)
    print("Multi-class Classification (6 Known Classes, 5-Fold CV)")
    print("=" * 60)

    paths, labels = load_image_dataset()
    skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True,
                          random_state=CONFIG['random_state'])
    scores = defaultdict(list)
    last_model, last_rf, last_data = None, None, {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(paths, labels)):
        print(f"\nFold {fold + 1}/{CONFIG['k_folds']}")
        X_train, X_test = paths[train_idx], paths[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train,
            random_state=CONFIG['random_state']
        )

        train_loader = get_dataloader(X_train, y_train, CONFIG['batch_size'])
        val_loader = get_dataloader(X_val, y_val, CONFIG['batch_size'], shuffle=False)
        test_loader = get_dataloader(X_test, y_test, CONFIG['batch_size'], shuffle=False)

        model = ECACAE().to(CONFIG['device'])
        model = train_cae(model, train_loader, val_loader,
                          CONFIG['cae_epochs'], CONFIG['device'])

        X_train_feat, y_train_feat = extract_latent_features(
            model, train_loader, CONFIG['device'])
        X_test_feat, y_test_feat = extract_latent_features(
            model, test_loader, CONFIG['device'])

        rf = RandomForestClassifier(
            n_estimators=CONFIG['n_estimators'],
            class_weight='balanced', n_jobs=-1)
        rf.fit(X_train_feat, y_train_feat)

        y_pred = rf.predict(X_test_feat)
        acc = accuracy_score(y_test_feat, y_pred)
        f1 = f1_score(y_test_feat, y_pred, average='macro')
        prec = precision_score(y_test_feat, y_pred, average='macro')
        rec = recall_score(y_test_feat, y_pred, average='macro')
        print(f" Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

        scores['acc'].append(acc)
        scores['f1'].append(f1)
        scores['prec'].append(prec)
        scores['rec'].append(rec)

        last_model = model
        last_rf = rf
        last_data = {'X': X_test_feat, 'y': y_test_feat, 'pred': y_pred}

    print("\nCV Summary")
    for k in ['acc', 'f1', 'prec', 'rec']:
        vals = scores[k]
        ci = stats.t.interval(0.95, len(vals) - 1, loc=np.mean(vals),
                              scale=stats.sem(vals)) if len(vals) > 1 else (0, 0)
        print(f"{k.upper():<5}: {np.mean(vals):.4f} ± {np.std(vals):.4f} "
              f"CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    print("\nClassification Report (Last Fold):")
    print(classification_report(
        last_data['y'], last_data['pred'],
        target_names=[CONFIG['class_names'][i] for i in CONFIG['known_classes']],
        digits=4
    ))

    cm = confusion_matrix(last_data['y'], last_data['pred'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CONFIG['class_names'][i] for i in CONFIG['known_classes']],
                yticklabels=[CONFIG['class_names'][i] for i in CONFIG['known_classes']])
    plt.title('Multi-class Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots',
                             'multiclass_confusion_matrix.png'), dpi=600)
    plt.close()
    print("Saved confusion matrix.")

    y_bin = label_binarize(last_data['y'], classes=CONFIG['known_classes'])
    y_score = last_rf.predict_proba(last_data['X'])
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(CONFIG['known_classes'])))
    for i, cls_id in enumerate(CONFIG['known_classes']):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{CONFIG["class_names"][cls_id]} (AUC={auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right")
    plt.title('Multi-class ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots',
                             'multiclass_roc.png'), dpi=600)
    plt.close()
    print("Saved ROC curves.")

    return last_model, last_rf

def analyze_feature_importance_merged(model):
    print("\n" + "=" * 60)
    print("Feature Importance (Merged 2x3 Plot - Top 10)")
    print("=" * 60)

    model.eval()
    criterion = nn.MSELoss()
    try:
        with open(os.path.join(CONFIG['image_root'], 'feature_columns.pkl'), 'rb') as f:
            feat_names_list = pickle.load(f)
    except Exception:
        feat_names_list = [f"Feature_{i}" for i in range(CONFIG['feature_dim'])]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, cls in enumerate(CONFIG['known_classes']):
        print(f" Computing gradients for {CONFIG['class_names'][cls]}...")
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))
        if not os.path.exists(cls_dir):
            axes[i].text(0.5, 0.5, 'No Data', ha='center')
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.endswith('.png')]
        if not files:
            axes[i].text(0.5, 0.5, 'No Images', ha='center')
            continue

        samples = np.random.choice(files, min(100, len(files)), replace=False)
        grad_acc = torch.zeros(3, 9, 9).to(CONFIG['device'])

        for fp in samples:
            img = Image.open(fp)
            img = np.array(img).astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
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
            name = feat_names_list[f_idx] if f_idx < len(feat_names_list) else f"F_{j}"
            feature_scores[name] = max(feature_scores[name], flat_grad[j])

        df_imp = pd.DataFrame(
            [{'Feature': k, 'Score': v} for k, v in feature_scores.items()])
        df_top = df_imp.sort_values(by='Score', ascending=False).head(10)

        df_imp.sort_values(by='Score', ascending=False).to_csv(
            os.path.join(CONFIG['output_dir'], 'reports',
                         f'imp_{CONFIG["class_names"][cls]}.csv'), index=False)

        sns.barplot(x='Score', y='Feature', data=df_top, ax=axes[i], color='#1f77b4')
        axes[i].set_title(f"{CONFIG['class_names'][cls]}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Importance")
        axes[i].set_ylabel("")
        axes[i].ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

    plt.tight_layout()
    base_path = os.path.join(CONFIG['output_dir'], 'plots', 'feature_importance_merged')
    for fmt in ['png', 'pdf', 'eps']:
        plt.savefig(f"{base_path}.{fmt}", dpi=600, bbox_inches='tight')
    plt.close()
    print("Saved merged feature importance plots.")

def visualize_saliency_9x9(model):
    print("\n" + "=" * 60)
    print("Generating 9x9 Spatial Saliency Maps")
    print("=" * 60)

    model.eval()
    criterion = nn.MSELoss()

    for cls in CONFIG['known_classes']:
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))
        if not os.path.exists(cls_dir):
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.endswith('.png')]
        if not files:
            continue

        samples = np.random.choice(files, min(20, len(files)), replace=False)
        grad_acc = torch.zeros(9, 9).to(CONFIG['device'])

        for fp in samples:
            img = Image.open(fp)
            img = np.array(img).astype(np.float32) / 255.0
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CONFIG['device'])
            t.requires_grad = True
            loss = criterion(model(t)[0], t)
            model.zero_grad()
            loss.backward()
            grad_acc += t.grad.data.abs().max(dim=1)[0].squeeze()

        heatmap = (grad_acc / len(samples)).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        plt.figure(figsize=(6, 5))
        im = plt.imshow(heatmap, cmap='jet', interpolation='nearest')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f'Saliency Map (9x9)\n{CONFIG["class_names"][cls]}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'], 'attention_maps',
                                 f'saliency_9x9_{CONFIG["class_names"][cls]}.png'), dpi=600)
        plt.close()

    print("Saved saliency maps.")

def visualize_eca_channel_heatmaps(model):
    print("\n" + "=" * 60)
    print("Generating ECA Channel Weights Heatmaps")
    print("=" * 60)

    model.eval()

    for cls in CONFIG['known_classes']:
        cls_dir = os.path.join(CONFIG['image_root'], str(cls))
        if not os.path.exists(cls_dir):
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.endswith('.png')]
        if not files:
            continue

        samples = np.random.choice(files, min(20, len(files)), replace=False)
        w1_acc, w2_acc, w3_acc = [], [], []

        with torch.no_grad():
            for fp in samples:
                img = Image.open(fp)
                img = np.array(img).astype(np.float32) / 255.0
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(CONFIG['device'])
                _, _, atts = model(t, return_attention=True)
                w1_acc.append(atts[0].squeeze().cpu().numpy())
                w2_acc.append(atts[1].squeeze().cpu().numpy())
                w3_acc.append(atts[2].squeeze().cpu().numpy())

        avg_w1 = np.mean(w1_acc, axis=0).reshape(4, 4)
        avg_w2 = np.mean(w2_acc, axis=0).reshape(4, 8)
        avg_w3 = np.mean(w3_acc, axis=0).reshape(8, 8)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        maps = [(avg_w1, "L1: 16 Ch"), (avg_w2, "L2: 32 Ch"), (avg_w3, "L3: 64 Ch")]
        for ax, (data, title) in zip(axes, maps):
            im = ax.imshow(data, cmap='jet', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f'ECA Channel Weights - {CONFIG["class_names"][cls]}', y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG['output_dir'], 'attention_maps',
                                 f'eca_weights_{CONFIG["class_names"][cls]}.png'), dpi=300)
        plt.close()

    print("Saved ECA channel weight heatmaps.")

def main():
    model, rf = multiclass_experiment()
    analyze_feature_importance_merged(model)
    visualize_saliency_9x9(model)
    visualize_eca_channel_heatmaps(model)
    print("\nAll Done! Check 'results_CAE_RF' folder.")

if __name__ == "__main__":
    main()
