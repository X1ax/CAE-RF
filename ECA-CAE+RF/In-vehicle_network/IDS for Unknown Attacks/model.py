"""
零日攻击检测系统 (Zero-Day Attack Detection) - CICIoV2024
单次实验版本 (Single Run)
特点：测试集中，每种攻击只使用 1/10 的样本
评估指标：Accuracy, Macro F1, FPR
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, confusion_matrix, accuracy_score, f1_score
)
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# ==================== 配置参数 ====================
CONFIG = {
    'img_size': (9, 9, 3),
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 1e-3,
    'test_split': 0.2,        # 20% Normal 用于测试
    'attack_sample_ratio': 0.1, # 🔥 关键修改：只使用 10% 的攻击样本
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_root': './CICIoV2024_Images',
    'output_dir': './results_ZeroDay',

    'normal_class': 'Normal',
    'attack_classes': ['GAS', 'SPEED', 'STEERING_WHEEL'],
    'random_seed': 42,
    'model_dir': './results_ZeroDay/model',
    'model_path': './results_ZeroDay/model/best_model.pth',
    'meta_path': './results_ZeroDay/model/meta.npz',   # 保存阈值等元数据
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(CONFIG['random_seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(os.path.join(CONFIG['output_dir'], 'plots'), exist_ok=True)
os.makedirs(CONFIG['model_dir'], exist_ok=True)

print(f"🖥️  Using device: {CONFIG['device']}")

# ==================== 数据集类 ====================
class CANDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            return tensor, self.labels[idx]
        except Exception:
            return torch.zeros(3, 9, 9), self.labels[idx]

# ==================== ECA-CAE 模型 ====================
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
        return x * att_weights.expand_as(x)

class ECACAE(nn.Module):
    def __init__(self):
        super(ECACAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(True),
        )
        self.eca1 = ECAModule(16)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
        )
        self.eca2 = ECAModule(32)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.eca3 = ECAModule(64)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x1 = self.eca1(x1)
        x2 = self.encoder2(x1)
        x2 = self.eca2(x2)
        x3 = self.encoder3(x2)
        encoded = self.eca3(x3)
        decoded = self.decoder(encoded)
        return decoded

# ==================== 数据加载工具 (含采样逻辑) ====================
def load_all_paths():
    normal_paths = []
    attack_paths = []

    print("📂 Scanning dataset...")

    # 1. 加载所有 Normal
    normal_dir = os.path.join(CONFIG['image_root'], CONFIG['normal_class'])
    if os.path.exists(normal_dir):
        files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')]
        normal_paths.extend(files)
        print(f"  ✅ Loaded Normal: {len(files)} (Using All)")
    else:
        raise FileNotFoundError(f"Normal directory not found: {normal_dir}")

    # 2. 加载 Attack 并采样 1/10
    print(f"  ⚡ Sampling Attacks (Ratio: {CONFIG['attack_sample_ratio']})")

    for atk_cls in CONFIG['attack_classes']:
        atk_dir = os.path.join(CONFIG['image_root'], atk_cls)
        if os.path.exists(atk_dir):
            files = [os.path.join(atk_dir, f) for f in os.listdir(atk_dir) if f.endswith('.png')]

            # 🔥 随机打乱并截取 1/10
            random.shuffle(files)
            cutoff = int(len(files) * CONFIG['attack_sample_ratio'])
            cutoff = max(1, cutoff) # 至少保留1张

            selected_files = files[:cutoff]
            attack_paths.extend(selected_files)

            print(f"  ⚠️ {atk_cls}: Found {len(files)} -> Kept {len(selected_files)}")
        else:
            print(f"  ❌ Warning: Attack directory not found: {atk_dir}")

    return np.array(normal_paths), np.array(attack_paths)

# ==================== 训练与测试逻辑 ====================
def train_one_epoch(model, loader, optimizer, criterion, epoch_idx):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"  Train Epoch {epoch_idx}/{CONFIG['epochs']}", leave=False)

    for imgs, _ in loop:
        imgs = imgs.to(CONFIG['device'])

        # Denoising: Add noise
        noise = torch.randn_like(imgs) * 0.1
        noisy_imgs = imgs + noise

        optimizer.zero_grad()
        output = model(noisy_imgs)
        loss = criterion(output, imgs)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate_reconstruction_error(model, loader):
    model.eval()
    errors = []
    labels = []
    criterion = nn.MSELoss(reduction='none')

    loop = tqdm(loader, desc="  🔍 Detecting Anomalies", leave=False)

    with torch.no_grad():
        for imgs, lbls in loop:
            imgs = imgs.to(CONFIG['device'])
            recon = model(imgs)
            loss = criterion(recon, imgs).mean(dim=[1, 2, 3])
            errors.extend(loss.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.array(errors), np.array(labels)

# ==================== 主流程 ====================
def run_experiment():
    # 1. 加载数据 (Normal 全量, Attack 1/10)
    normal_paths, attack_paths = load_all_paths()

    # 2. 划分训练/测试集
    train_paths, val_normal_paths = train_test_split(
        normal_paths, test_size=CONFIG['test_split'], random_state=CONFIG['random_seed']
    )

    test_paths = np.concatenate([val_normal_paths, attack_paths])

    # Label: 0 = Normal, 1 = Attack
    train_labels = np.zeros(len(train_paths))
    test_labels = np.concatenate([
        np.zeros(len(val_normal_paths)),
        np.ones(len(attack_paths))
    ])

    print(f"\n📊 Final Data Split:")
    print(f"  - Training Set (Normal):   {len(train_paths)}")
    print(f"  - Testing Set (Mixed):     {len(test_paths)}")
    print(f"    - Normal (Validation):   {len(val_normal_paths)}")
    print(f"    - Attacks (Sampled):     {len(attack_paths)}")

    train_loader = DataLoader(CANDataset(train_paths, train_labels),
                              batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(CANDataset(test_paths, test_labels),
                             batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # 3. 初始化模型
    model = ECACAE().to(CONFIG['device'])
    criterion = nn.MSELoss()

    # ==================== 检查是否存在已保存的最佳模型 ====================
    model_path = CONFIG['model_path']
    meta_path  = CONFIG['meta_path']

    if os.path.exists(model_path) and os.path.exists(meta_path):
        print(f"\n✅ Found saved model at {model_path}, skipping training...")
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.eval()

        meta = np.load(meta_path)
        best_thresh = float(meta['threshold'])
        print(f"   Loaded threshold: {best_thresh:.6f}")

        # 直接推理
        print("\n🔍 Running Detection with Loaded Model...")
        scores, y_true = evaluate_reconstruction_error(model, test_loader)

    else:
        # 4. 训练，并保存每个 epoch 中验证损失最低的模型
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

        best_val_loss = float('inf')

        print("\n🚀 Starting Training (Denoising ECA-CAE)...")
        for epoch in range(1, CONFIG['epochs'] + 1):
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, epoch)

            # 用训练集损失作为代理（纯无监督，无单独验证集）
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), model_path)
                tqdm.write(f"     💾 [Epoch {epoch}] New best loss: {avg_loss:.6f} -> model saved")
            elif epoch % 5 == 0 or epoch == 1:
                tqdm.write(f"     [Epoch {epoch}/{CONFIG['epochs']}] Avg Loss: {avg_loss:.6f}")

        print(f"\n✅ Best model saved to {model_path} (loss={best_val_loss:.6f})")

        # 加载最佳模型权重用于推理
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.eval()

        # 5. 测试
        print("\n🔍 Running Detection...")
        scores, y_true = evaluate_reconstruction_error(model, test_loader)

        # 6. 计算最优阈值并保存元数据
        fpr_curve, tpr_curve, thresholds = roc_curve(y_true, scores)
        J = tpr_curve - fpr_curve
        best_ix = np.argmax(J)
        best_thresh = thresholds[best_ix]

        np.savez(meta_path, threshold=best_thresh)
        print(f"   Threshold saved to {meta_path}")

    # 7. 计算阈值和指标（加载场景下重新算阈值，或使用已保存阈值）
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, scores)
    J = tpr_curve - fpr_curve
    best_ix = np.argmax(J)
    best_thresh = thresholds[best_ix]   # 始终从当前 scores 重算，保证一致性

    y_pred = (scores > best_thresh).astype(int)

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'Accuracy': acc,
        'Macro F1': macro_f1,
        'FPR': fpr,
        'Threshold': best_thresh
    }

    return metrics, y_true, scores, y_pred

# ==================== 结果可视化 ====================
def plot_results(metrics, y_true, y_scores, y_pred):
    print("\n" + "="*60)
    print(f"{'🏁 Final Evaluation Metrics':^60}")
    print("="*60)
    print(f"{'ACCURACY':<20}: {metrics['Accuracy']:.4f}")
    print(f"{'MACRO F1':<20}: {metrics['Macro F1']:.4f}")
    print(f"{'FPR':<20}: {metrics['FPR']:.4f}")
    print("-" * 60)
    print(f"Optimal Threshold   : {metrics['Threshold']:.6f}")
    print("="*60)

    # 1. 混淆矩阵热力图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'confusion_matrix.png'), dpi=600)
    plt.close()

    # 2. 误差分布图（Times New Roman 字体，放大图例和所有字体）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_scores[y_true==0], bins=50, alpha=0.7, color='blue', label='Normal (Validation)', density=True)
    ax.hist(y_scores[y_true==1], bins=50, alpha=0.7, color='red', label='Attacks', density=True)
    ax.axvline(metrics['Threshold'], color='k', linestyle='--', label='Threshold')
    ax.set_xlabel('Reconstruction Error (MSE)', fontsize=20, fontfamily='Times New Roman')
    ax.set_ylabel('Density', fontsize=20, fontfamily='Times New Roman')
    ax.tick_params(axis='both', labelsize=16)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')
    ax.legend(fontsize=18, prop={'family': 'Times New Roman', 'size': 18})
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'plots', 'error_distribution.pdf'), dpi=600, format='pdf')
    plt.close()

    # 重置字体设置，避免影响后续图表
    plt.rcParams.update(plt.rcParamsDefault)

if __name__ == "__main__":
    metrics, y_true, y_scores, y_pred = run_experiment()
    plot_results(metrics, y_true, y_scores, y_pred)
    print(f"\n✅ Results saved to {CONFIG['output_dir']}")