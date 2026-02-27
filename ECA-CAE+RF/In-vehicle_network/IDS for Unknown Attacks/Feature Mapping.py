import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle

# =================配置区域=================
WINDOW_SIZE = 27
STRIDE = 27  # 不重叠滑动窗口
FEATURE_DIM = 9  # ID (1) + Data (8) = 9

IMG_H, IMG_W, CHANNELS = 9, 9, 3

# 根据你的数据集分布定义的标签映射
LABEL_MAP = {
    "BENIGN": 0,
    "DoS": 1,
    "RPM": 2,
    "SPEED": 3,
    "STEERING_WHEEL": 4,
    "GAS": 5
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 你指定需要转换的目标类别
TARGET_CLASSES = ["SPEED", "GAS", "STEERING_WHEEL"]

OUTPUT_ROOT = "./CICIoV2024_Images"
# =========================================

def load_data_no_timestamp(csv_path):
    """
    加载数据，假设CSV行顺序即为时间顺序
    """
    print(f"📖 Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # ⚠️如果不确定列名，请取消下面这行的注释来查看列名
    # print(df.columns)

    # 假设列名包含 ID, DATA0...DATA7, Label (根据CIC通常的格式)
    # 如果你的列名是小写 (id, data0...) 请在这里修改
    feature_cols = ["ID"] + [f"DATA_{i}" for i in range(8)]

    # 检查列是否存在，防止报错
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        # 尝试查找不区分大小写的匹配
        print(f"⚠️ Warning: Standard columns not found: {missing_cols}")
        print("Trying to auto-detect columns...")
        # 简单的自动修正逻辑（根据你的实际CSV情况调整）
        df.columns = [c.upper() for c in df.columns]

    # 提取特征和标签
    try:
        features = df[feature_cols].values.astype(np.float32)
        labels = df["Label"].values # 确保标签列名为 Label
    except KeyError as e:
        raise KeyError(f"❌ 找不到列名: {e}. 请检查CSV文件的表头是否为 ID, DATA0...DATA7, Label")

    return features, labels

def normalize_features(features):
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, scaler

def window_to_rgb_image(window):
    """
    将27×9的窗口映射到9×9×3的RGB图像
    逻辑：
    - window shape: (27, 9)
    - img shape: (9, 9, 3)
    """
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)

    for t in range(WINDOW_SIZE):
        c = t // 9  # channel (0, 1, 2) --> 对应时间段
        i = t % 9  # row (0-8)         --> 对应每个通道内的行
        for f in range(FEATURE_DIM):   # col (0-8) --> 对应特征
            img[i, f, c] = window[t, f]

    return img

def majority_label(window_labels):
    """获取窗口内出现最多的标签"""
    if len(window_labels) == 0:
        return "BENIGN" # fallback
    return Counter(window_labels).most_common(1)[0][0]

def split_into_attack_segments(labels):
    """
    为了保持攻击的纯度，我们在标签变化的地方切断
    """
    segments = []
    start = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            segments.append((start, i))
            start = i

    segments.append((start, len(labels)))
    return segments

def generate_windows_in_segment(features, labels, start, end):
    windows = []
    idx = start

    # 只有当段落长度大于窗口大小时才处理
    while idx + WINDOW_SIZE <= end:
        window_feat = features[idx:idx + WINDOW_SIZE]
        window_labels = labels[idx:idx + WINDOW_SIZE]

        label_name = majority_label(window_labels)

        windows.append({
            "features": window_feat,
            "label_name": label_name,
            "label_id": LABEL_MAP.get(label_name, -1) # 如果有未知标签，设为-1
        })

        idx += STRIDE  # 不重叠

    return windows

def save_image(img, path):
    # 将 0-1 float 转为 0-255 uint8
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def process_dataset(csv_path):
    # 1. 加载数据
    features, labels = load_data_no_timestamp(csv_path)

    print(f"✓ Data loaded. Shape: {features.shape}")

    # 2. 归一化
    print("🔄 Normalizing features...")
    features, scaler = normalize_features(features)

    # 3. 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 为目标类别创建文件夹
    for target in TARGET_CLASSES:
        if target in LABEL_MAP:
            os.makedirs(os.path.join(OUTPUT_ROOT, target), exist_ok=True)

    # 保存scaler以便后续反向转换
    with open(os.path.join(OUTPUT_ROOT, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # 4. 按标签连续性分段 (Data Segmentation)
    segments = split_into_attack_segments(labels)

    class_counter = Counter()
    total_saved = 0
    img_global_id = 0

    print(f"🚀 Processing segments and generating images for: {TARGET_CLASSES}...")

    for start, end in tqdm(segments):
        # 优化：如果这一段的标签根本不是我们要的，直接跳过 (假设整段标签一致)
        # 取段落中间的一个标签做检查
        segment_label = labels[start]
        if segment_label not in TARGET_CLASSES:
            continue

        # 生成窗口
        windows = generate_windows_in_segment(features, labels, start, end)

        for w in windows:
            label_name = w["label_name"]

            # 二次确认：只处理目标类别
            if label_name in TARGET_CLASSES:
                img = window_to_rgb_image(w["features"])

                # 文件名格式: Label_ID.png
                save_path = os.path.join(
                    OUTPUT_ROOT,
                    label_name,
                    f"{label_name}_{img_global_id}.png"
                )

                save_image(img, save_path)

                class_counter[label_name] += 1
                total_saved += 1
                img_global_id += 1

    return class_counter

if __name__ == "__main__":
    # 请修改这里的路径指向你的 .csv 文件
    csv_file_path = "CICIoV2024.csv"

    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File {csv_file_path} not found.")
    else:
        print("=" * 60)
        print("CICIoV2024 No-Timestamp Image Converter")
        print("=" * 60)

        counter = process_dataset(csv_file_path)

        print("\n📊 Generation Report:")
        for cls_name in TARGET_CLASSES:
            print(f"  - {cls_name}: {counter[cls_name]} images")


        print(f"Output directory: {OUTPUT_ROOT}")