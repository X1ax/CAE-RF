import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle

# ========== 配置参数 ==========
WINDOW_SIZE = 3  # 3条数据
STRIDE = 3  # 不重叠
FEATURE_DIM = 80  # 每条数据80个特征

IMG_H, IMG_W, CHANNELS = 9, 9, 3  # 9×9×3 = 243 (240特征 + 3 padding)

# 定义标签映射
LABEL_MAP = {
    "BENIGN": 0,
    "DoS": 1,
    "PortScan": 2,
    "BruteForce": 3,
    "WebAttack": 4,
    "Bot": 5,
    "Infiltration": 6
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 少数类（全部保留）
MINORITY_CLASSES = ["WebAttack", "Bot", "Infiltration"]

# 多数类窗口数量限制
MAJORITY_CLASS_LIMITS = {
    "BENIGN": 18423,
    "DoS": 7234,
    "PortScan": 5436,
    "BruteForce": None  # 不限制
}

OUTPUT_ROOT = "CICIDS2017_images"


def load_and_preprocess_csv(csv_path):
    """加载CSV并预处理"""
    print("📂 Loading CSV file...")
    df = pd.read_csv(csv_path)

    # 删除空值
    original_len = len(df)
    df = df.dropna()
    print(f"✓ Removed {original_len - len(df)} rows with missing values")

    # 按Timestamp排序（如果存在）
    if 'Timestamp' in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # 提取特征列（除了Timestamp和Label）
    exclude_cols = ['Timestamp', 'Label', ' Timestamp', ' Label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"✓ Found {len(feature_cols)} feature columns")

    # 处理非数值列
    features_df = df[feature_cols]
    for col in feature_cols:
        if features_df[col].dtype == 'object':
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

    features = features_df.values.astype(np.float32)
    labels = df["Label"].values if "Label" in df.columns else df[" Label"].values

    # 替换inf和nan
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    return features, labels, feature_cols


def normalize_features(features):
    """归一化特征到[0,1]"""
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, scaler


def window_to_rgb_image(window):
    """
    将3×80的窗口映射到9×9×3的RGB图像
    参考原始代码的映射逻辑：
    - 时间步t的通道: c = t // 9
    - 时间步t的行: i = t % 9
    - 特征f的列: j = f

    调整后：3条数据，80个特征
    - 将80个特征分3组映射到3个通道
    - 每个通道需要27个位置 (9×3)
    - 3×27 = 81 > 80，所以每个通道放27个，其中第3个通道最后一个padding
    """
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)

    # 展平窗口: 3×80 = 240
    flat_features = window.flatten()

    # 补3个padding到243
    padded_features = np.pad(flat_features, (0, 3), mode='constant', constant_values=0)

    # 按照原始逻辑映射：按时间步遍历，每个时间步内遍历特征
    idx = 0
    for t in range(WINDOW_SIZE):  # 3个时间步
        for f in range(FEATURE_DIM):  # 80个特征
            c = idx // (IMG_H * IMG_W)  # 通道
            temp = idx % (IMG_H * IMG_W)
            i = temp // IMG_W  # 行
            j = temp % IMG_W  # 列
            img[i, j, c] = padded_features[idx]
            idx += 1

    # 填充最后3个padding
    for _ in range(3):
        c = idx // (IMG_H * IMG_W)
        temp = idx % (IMG_H * IMG_W)
        i = temp // IMG_W
        j = temp % IMG_W
        img[i, j, c] = 0.0
        idx += 1

    return img


def image_to_features(img, scaler=None):
    """
    从9×9×3的RGB图像反推原始特征
    返回3×80的窗口（去除padding）
    如果提供scaler，则反归一化
    """
    # 按照相同的映射逻辑提取
    flat_features = []

    for idx in range(240):  # 只取前240个（去除padding）
        c = idx // (IMG_H * IMG_W)
        temp = idx % (IMG_H * IMG_W)
        i = temp // IMG_W
        j = temp % IMG_W
        flat_features.append(img[i, j, c])

    features_flat = np.array(flat_features)

    # 重塑为3×80
    window = features_flat.reshape(WINDOW_SIZE, FEATURE_DIM)

    # 反归一化
    if scaler is not None:
        window = scaler.inverse_transform(window)

    return window


def majority_label(window_labels):
    """返回窗口中出现最多的标签ID"""
    most_common = Counter(window_labels).most_common(1)[0][0]
    return LABEL_MAP.get(most_common, -1)


def save_image(img, path):
    """保存图像"""
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def generate_windows_for_class(features, labels, class_name, max_windows=None):
    """为单个类别生成窗口"""
    windows = []
    idx = 0

    while idx + WINDOW_SIZE <= len(features):
        if max_windows and len(windows) >= max_windows:
            break

        window_feat = features[idx:idx + WINDOW_SIZE]
        window_labels = labels[idx:idx + WINDOW_SIZE]

        # 确保窗口内标签一致性（majority voting）
        label_id = majority_label(window_labels)

        windows.append({
            "features": window_feat,
            "label": label_id
        })

        idx += STRIDE

    return windows


def process_class(features, labels, class_name, img_id_start, max_windows=None):
    """处理单个类别并生成图像"""
    print(f"\n{'=' * 50}")
    print(f"🎯 Processing class: {class_name}")
    print(f"   Total samples: {len(features)}")
    if max_windows:
        print(f"   Window limit: {max_windows}")

    label_id = LABEL_MAP[class_name]
    output_dir = os.path.join(OUTPUT_ROOT, str(label_id))
    os.makedirs(output_dir, exist_ok=True)

    # 生成窗口
    windows = generate_windows_for_class(features, labels, class_name, max_windows)
    print(f"   Generated windows: {len(windows)}")

    # 保存图像
    img_id = img_id_start
    for w in tqdm(windows, desc=f"Saving {class_name} images"):
        img = window_to_rgb_image(w["features"])

        save_path = os.path.join(
            output_dir,
            f"{class_name}_{img_id}.png"
        )

        save_image(img, save_path)
        img_id += 1

    return len(windows), img_id


def generate_full_image_dataset(csv_path):
    """生成完整的图像数据集"""
    # 加载数据
    features, labels, feature_cols = load_and_preprocess_csv(csv_path)

    # 归一化
    print("\n🔄 Normalizing features...")
    features_norm, scaler = normalize_features(features)

    # 保存scaler用于后续反归一化
    scaler_path = os.path.join(OUTPUT_ROOT, "scaler.pkl")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")

    # 保存特征列名（用于后续分析和可解释性）
    feature_cols_path = os.path.join(OUTPUT_ROOT, "feature_columns.pkl")
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Feature columns saved to {feature_cols_path}")

    # 统计各类别数量
    print("\n📊 Original class distribution:")
    label_counts = Counter(labels)
    for label in sorted(LABEL_MAP.keys(), key=lambda x: LABEL_MAP[x]):
        count = label_counts.get(label, 0)
        print(f"   {label:15s}: {count:8d}")

    # 分离少数类和多数类
    print("\n🔍 Separating classes...")
    class_data = {}
    for label_name in LABEL_MAP.keys():
        mask = labels == label_name
        class_data[label_name] = {
            "features": features_norm[mask],
            "labels": labels[mask]
        }
        print(f"   {label_name:15s}: {len(class_data[label_name]['features'])} samples")

    # 创建所有类别文件夹
    for cid in LABEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, str(cid)), exist_ok=True)

    # 处理每个类别
    class_counter = Counter()
    img_id = 0

    print("\n" + "=" * 60)
    print("🚀 Generating RGB images...")
    print("=" * 60)

    # 先处理少数类（全部保留）
    for class_name in MINORITY_CLASSES:
        if class_name in class_data:
            count, img_id = process_class(
                class_data[class_name]["features"],
                class_data[class_name]["labels"],
                class_name,
                img_id,
                max_windows=None  # 不限制
            )
            class_counter[LABEL_MAP[class_name]] = count

    # 再处理多数类（有限制）
    for class_name, limit in MAJORITY_CLASS_LIMITS.items():
        if class_name in class_data:
            count, img_id = process_class(
                class_data[class_name]["features"],
                class_data[class_name]["labels"],
                class_name,
                img_id,
                max_windows=limit
            )
            class_counter[LABEL_MAP[class_name]] = count

    return scaler, class_counter, feature_cols


if __name__ == "__main__":
    # 修改为你的CSV路径
    csv_path = "./dataset/CICIDS2017_with_Timestamp.csv"

    print("=" * 60)
    print("CICIDS2017 Dataset to RGB Image Converter")
    print("=" * 60)

    scaler, class_counter, feature_cols = generate_full_image_dataset(csv_path)

    # 输出最终统计
    print("\n" + "=" * 60)
    print("📊 Final Image Dataset Statistics:")
    print("=" * 60)
    total_images = 0
    for cid in sorted(class_counter.keys()):
        count = class_counter[cid]
        total_images += count
        print(f"Class {cid} ({INV_LABEL_MAP[cid]:15s}): {count:6d} images")

    print(f"\n{'Total':21s}: {total_images:6d} images")
    print(f"\nImages saved to: {OUTPUT_ROOT}/")
    print(f"Scaler saved to: {OUTPUT_ROOT}/scaler.pkl")
    print(f"Feature columns saved to: {OUTPUT_ROOT}/feature_columns.pkl")

    # 测试反向映射
    print("\n" + "=" * 60)
    print("🧪 Testing image-to-features conversion...")
    print("=" * 60)

    # 读取一张图像进行测试
    test_class = 0  # BENIGN
    test_dir = os.path.join(OUTPUT_ROOT, str(test_class))
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        if test_images:
            test_img_path = os.path.join(test_dir, test_images[0])
            test_img = np.array(Image.open(test_img_path)).astype(np.float32) / 255.0

            # 反向映射
            recovered_features = image_to_features(test_img, scaler)

            print(f"✓ Test image: {test_images[0]}")
            print(f"✓ Recovered features shape: {recovered_features.shape}")
            print(f"✓ Feature range: [{recovered_features.min():.4f}, {recovered_features.max():.4f}]")
            print("\n✅ Image-to-features conversion successful!")

            # 显示部分恢复的特征（前3行，前5列）
            print(f"\n📊 Sample recovered features (first 3 rows, first 5 columns):")
            print(recovered_features[:3, :5])