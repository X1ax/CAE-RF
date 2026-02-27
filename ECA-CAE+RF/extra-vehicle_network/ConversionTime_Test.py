"""
Benchmark: 测量 CICIDS2017 每条原始数据转换为 9×9×3 RGB 图像的平均时间
使用 5-fold 交叉验证，以 mean ± std 形式输出结果
"""

import os
import time
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

# ────────────────────────────────────────────
# 超参数
# ────────────────────────────────────────────
WINDOW_SIZE = 3
STRIDE      = 3
FEATURE_DIM = 80
IMG_H, IMG_W, CHANNELS = 9, 9, 3
N_SPLITS    = 5          # K-fold 折数
WARMUP_RUNS = 3          # 每折预热次数

LABEL_MAP = {
    "BENIGN": 0, "DoS": 1, "PortScan": 2,
    "BruteForce": 3, "WebAttack": 4, "Bot": 5, "Infiltration": 6
}

MAJORITY_CLASS_LIMITS = {
    "BENIGN": 18423, "DoS": 7234, "PortScan": 5436, "BruteForce": None
}
MINORITY_CLASSES = ["WebAttack", "Bot", "Infiltration"]


# ────────────────────────────────────────────
# 核心函数
# ────────────────────────────────────────────
def window_to_rgb_image(window: np.ndarray) -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)
    flat_features = window.flatten()
    padded_features = np.pad(flat_features, (0, 3), mode='constant', constant_values=0)

    idx = 0
    for t in range(WINDOW_SIZE):
        for f in range(FEATURE_DIM):
            c = idx // (IMG_H * IMG_W)
            temp = idx % (IMG_H * IMG_W)
            i = temp // IMG_W
            j = temp % IMG_W
            img[i, j, c] = padded_features[idx]
            idx += 1

    for _ in range(3):
        c = idx // (IMG_H * IMG_W)
        temp = idx % (IMG_H * IMG_W)
        i = temp // IMG_W
        j = temp % IMG_W
        img[i, j, c] = 0.0
        idx += 1

    return img


def save_image_to_bytes(img: np.ndarray) -> bytes:
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return buf.getvalue()


def majority_label(window_labels):
    most_common = Counter(window_labels).most_common(1)[0][0]
    return LABEL_MAP.get(most_common, -1)


# ────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    print(f"📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.dropna()

    if 'Timestamp' in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    exclude_cols = ['Timestamp', 'Label', ' Timestamp', ' Label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    features_df = df[feature_cols].copy()
    for col in feature_cols:
        if features_df[col].dtype == 'object':
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    features = features_df.values.astype(np.float32)
    label_col = "Label" if "Label" in df.columns else " Label"
    labels = df[label_col].values

    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    if features.shape[1] > FEATURE_DIM:
        print(f"⚠  Features truncated from {features.shape[1]} to {FEATURE_DIM} columns")
        features = features[:, :FEATURE_DIM]
    elif features.shape[1] < FEATURE_DIM:
        pad_cols = FEATURE_DIM - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_cols)), constant_values=0)
        print(f"⚠  Features padded to {FEATURE_DIM} columns")

    print("🔄 Normalizing features...")
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    print("🪟 Building windows (respecting class limits)...")
    class_data = {}
    for name in LABEL_MAP:
        mask = labels == name
        class_data[name] = {"features": features[mask], "labels": labels[mask]}

    windows_list = []

    def _make_windows(feat, max_w=None):
        out = []
        idx = 0
        while idx + WINDOW_SIZE <= len(feat):
            if max_w and len(out) >= max_w:
                break
            out.append(feat[idx:idx + WINDOW_SIZE])
            idx += STRIDE
        return out

    for name in MINORITY_CLASSES:
        if name in class_data and len(class_data[name]["features"]) >= WINDOW_SIZE:
            windows_list.extend(_make_windows(class_data[name]["features"]))

    for name, limit in MAJORITY_CLASS_LIMITS.items():
        if name in class_data and len(class_data[name]["features"]) >= WINDOW_SIZE:
            windows_list.extend(_make_windows(class_data[name]["features"], max_w=limit))

    windows = np.array(windows_list, dtype=np.float32)  # (N, 3, 80)
    print(f"✅ Total windows: {len(windows):,}")
    return windows


# ────────────────────────────────────────────
# 计时
# ────────────────────────────────────────────
def time_conversion_for_fold(windows: np.ndarray) -> float:
    elapsed = []
    for w in windows:
        t0 = time.perf_counter()
        img = window_to_rgb_image(w)
        _   = save_image_to_bytes(img)
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)
    return float(np.mean(elapsed))


# ────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────
def benchmark(csv_path: str):
    windows = load_and_prepare(csv_path)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_means = []

    print(f"\n⏱  Running {N_SPLITS}-fold benchmark "
          f"(warmup={WARMUP_RUNS} rounds per fold)...\n")

    for fold_idx, (_, test_idx) in enumerate(kf.split(windows), start=1):
        fold_windows = windows[test_idx]

        # 预热
        for _ in range(WARMUP_RUNS):
            for w in fold_windows[:min(50, len(fold_windows))]:
                window_to_rgb_image(w)

        # 正式计时
        mean_t = time_conversion_for_fold(fold_windows)
        fold_means.append(mean_t)

        print(f"  Fold {fold_idx}/{N_SPLITS} | "
              f"samples={len(fold_windows):,} | "
              f"mean={mean_t*1e6:.2f} µs/sample")

    fold_means = np.array(fold_means)
    overall_mean = fold_means.mean()
    overall_std  = fold_means.std(ddof=1)

    print("\n" + "=" * 55)
    print("📊  5-Fold Benchmark Results  (CICIDS2017)")
    print("=" * 55)
    print(f"  Per-fold means (µs): "
          f"{[f'{v*1e6:.2f}' for v in fold_means]}")
    print(f"\n  ✅ Mean ± Std (per sample):")
    print(f"     {overall_mean*1e6:.4f} ± {overall_std*1e6:.4f}  µs")
    print(f"     {overall_mean*1e3:.6f} ± {overall_std*1e3:.6f}  ms")
    print(f"     {overall_mean:.8f} ± {overall_std:.8f}  s")
    print("=" * 55)

    if overall_mean > 0:
        print(f"\n  Throughput: ~{1/overall_mean:.1f} samples/sec  (single-threaded)")
    print()


if __name__ == "__main__":
    CSV_PATH = "./dataset/CICIDS2017_with_Timestamp.csv"

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Please update CSV_PATH at the bottom of this script."
        )


    benchmark(CSV_PATH)

