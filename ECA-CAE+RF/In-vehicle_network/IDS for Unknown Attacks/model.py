"""
Zero-Day Attack Detection System - CICIoV2024
Single Run Version
Metrics: Accuracy, Macro F1, FPR
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

CONFIG = {
    'img_size': (9, 9, 3),
    'batch_size': 64,
    'epochs': 20,
    'learning_rate': 1e-3,
    'test_split': 0.2,
    'attack_sample_ratio': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_root': './CICIoV2024_Images',
    'output_dir': './results_ZeroDay',

    'normal_class': 'Normal',
    'attack_classes': ['GAS', 'SPEED', 'STEERING_WHEEL'],
    'random_seed': 42,
    'model_dir': './results_ZeroDay/model',
    'model_path': './results_ZeroDay/model/best_model.pth',
    'meta_path': './results_ZeroDay/model/meta.npz',
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

print(f"Using device: {CONFIG['device']}")

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

def load_all_paths():
    normal_paths = []
    attack_paths = []

    print("Scanning dataset...")

    normal_dir = os.path.join(CONFIG['image_root'], CONFIG['normal_class'])
    if os.path.exists(normal_dir):
        files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')]
        normal_paths.extend(files)
        print(f"Loaded Normal: {len(files)} (Using All)")
    else:
        raise FileNotFoundError(f"Normal directory not found: {normal_dir}")

    print(f"Sampling Attacks (Ratio: {CONFIG['attack_sample_ratio']})")

    for atk_cls in CONFIG['attack_classes']:
        atk_dir = os.path.join(CONFIG['image_root'], atk_cls)
        if os.path.exists(atk_dir):
            files = [os.path.join(atk_dir, f) for f in os.listdir(atk_dir) if f.endswith('.png')]
            random.shuffle(files)
            cutoff = int(len(files) * CONFIG['attack_sample_ratio'])
            cutoff = max(1, cutoff)
            selected_files = files[:cutoff]
            attack_paths.extend(selected_files)
            print(f"{atk_cls}: Found {len(files)} -> Kept {len(selected_files)}")
        else:
            print(f"Attack directory not found: {atk_dir}")

    return np.array(normal_paths), np.array(attack_paths)
