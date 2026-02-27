"""
Benchmark: 测量每条原始数据转换为 9×9×3 RGB 图像的平均时间
使用 5-fold 交叉验证，以 mean ± std 形式输出结果
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

# ────────────────────────────────────────────
# 超参数
# ────────────────────────────────────────────
WINDOW_SIZE = 27
STRIDE      = 27
FEATURE_DIM = 9
IMG_H, IMG_W, CHANNELS = 9, 9, 3
N_SPLITS    = 5          # K-fold 折数
WARMUP_RUNS = 3          # 每折正式计时前的预热次数（消除 JIT / cache 影响）

LABEL_MAP = {"DoS": 0, "Gear": 1, "Fuzzy": 2, "RPM": 3, "Normal": 4}


# ────────────────────────────────────────────
# 核心函数（从原脚本复制）
# ────────────────────────────────────────────
def window_to_rgb_image(window: np.ndarray) -> np.ndarray:
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)
    for t in range(WINDOW_SIZE):
        c = t // 9
        i = t % 9
        for f in range(FEATURE_DIM):
            img[i, f, c] = window[t, f]
    return img


def save_image_to_bytes(img: np.ndarray) -> bytes:
    """将 float32 图像转为 uint8 并编码为 PNG bytes（模拟完整流程）"""
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return buf.getvalue()


def majority_label(window_labels):
    return LABEL_MAP[Counter(window_labels).most_common(1)[0][0]]


# ────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────
def load_and_prepare(csv_path: str):
    print(f"📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    feature_cols = ["ID"] + [f"Data{i}" for i in range(8)]
    features = df[feature_cols].values.astype(np.float32)
    labels   = df["Label"].values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # 构建所有窗口（不重叠，与原脚本一致）
    windows, window_labels = [], []
    for i in range(0, len(features) - WINDOW_SIZE + 1, STRIDE):
        windows.append(features[i:i + WINDOW_SIZE])
        window_labels.append(majority_label(labels[i:i + WINDOW_SIZE]))

    windows = np.array(windows, dtype=np.float32)   # (N, 27, 9)
    window_labels = np.array(window_labels)
    print(f"✅ Total windows: {len(windows):,}")
    return windows, window_labels


# ────────────────────────────────────────────
# 单次转换计时（window → RGB image → PNG bytes）
# ────────────────────────────────────────────
def time_conversion_for_fold(windows: np.ndarray) -> float:
    """
    对给定的一组窗口逐条计时，返回每条数据的平均耗时（秒）。
    计时范围：window_to_rgb_image + 图像编码（PNG bytes），
    与实际落盘流程完全对应。
    """
    elapsed_per_sample = []
    for w in windows:
        t0 = time.perf_counter()
        img = window_to_rgb_image(w)
        _   = save_image_to_bytes(img)
        t1 = time.perf_counter()
        elapsed_per_sample.append(t1 - t0)
    return np.mean(elapsed_per_sample)


# ────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────
def benchmark(csv_path: str):
    windows, window_labels = load_and_prepare(csv_path)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_means = []   # 每折的每样本平均转换时间（秒）

    print(f"\n⏱  Running {N_SPLITS}-fold benchmark "
          f"(warmup={WARMUP_RUNS} runs per fold)...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(
            kf.split(windows), start=1):

        fold_windows = windows[test_idx]   # 用 test split 做计时（更客观）

        # ── 预热（消除首次执行开销）──────────────────
        for _ in range(WARMUP_RUNS):
            for w in fold_windows[:min(50, len(fold_windows))]:
                _ = window_to_rgb_image(w)

        # ── 正式计时 ─────────────────────────────────
        mean_t = time_conversion_for_fold(fold_windows)
        fold_means.append(mean_t)

        print(f"  Fold {fold_idx}/{N_SPLITS} | "
              f"samples={len(fold_windows):,} | "
              f"mean={mean_t*1e6:.2f} µs/sample")

    # ── 汇总 ─────────────────────────────────────────
    fold_means = np.array(fold_means)
    overall_mean = fold_means.mean()
    overall_std  = fold_means.std(ddof=1)   # 样本标准差

    print("\n" + "=" * 55)
    print("📊  5-Fold Benchmark Results")
    print("=" * 55)
    print(f"  Per-fold means (µs): "
          f"{[f'{v*1e6:.2f}' for v in fold_means]}")
    print(f"\n  ✅ Mean ± Std (per sample):")
    print(f"     {overall_mean*1e6:.4f} ± {overall_std*1e6:.4f}  µs")
    print(f"     {overall_mean*1e3:.6f} ± {overall_std*1e3:.6f}  ms")
    print(f"     {overall_mean:.8f} ± {overall_std:.8f}  s")
    print("=" * 55)

    # ── 吞吐量参考 ───────────────────────────────────
    if overall_mean > 0:
        throughput = 1.0 / overall_mean
        print(f"\n  Throughput: ~{throughput:.1f} samples/sec  "
              f"(single-threaded)")
    print()


# ────────────────────────────────────────────
# 入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH = "./dataset/Car_Hacking_with_Timestamp.csv"   # ← 修改为你的路径

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}\n"
            "Please update CSV_PATH at the bottom of this script."
        )


    benchmark(CSV_PATH)
