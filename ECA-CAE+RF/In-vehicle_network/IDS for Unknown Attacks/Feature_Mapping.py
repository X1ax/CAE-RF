
import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle


WINDOW_SIZE = 27
STRIDE = 27
FEATURE_DIM = 9  # ID (1) + Data (8) = 9

IMG_H, IMG_W, CHANNELS = 9, 9, 3

LABEL_MAP = {
    "BENIGN": 0,
    "DoS": 1,
    "RPM": 2,
    "SPEED": 3,
    "STEERING_WHEEL": 4,
    "GAS": 5
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

TARGET_CLASSES = ["SPEED", "GAS", "STEERING_WHEEL"]

OUTPUT_ROOT = "./CICIoV2024_Images"


def load_data_no_timestamp(csv_path):
    print(f"📖 Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    feature_cols = ["ID"] + [f"DATA_{i}" for i in range(8)]

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Standard columns not found: {missing_cols}")
        print("Trying to auto-detect columns...")
        df.columns = [c.upper() for c in df.columns]

    try:
        features = df[feature_cols].values.astype(np.float32)
        labels = df["Label"].values
    except KeyError as e:
        raise KeyError("Required columns not found in CSV file.")

    return features, labels


def normalize_features(features):
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, scaler


def window_to_rgb_image(window):
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)

    for t in range(WINDOW_SIZE):
        c = t // 9
        i = t % 9
        for f in range(FEATURE_DIM):
            img[i, f, c] = window[t, f]

    return img


def majority_label(window_labels):
    if len(window_labels) == 0:
        return "BENIGN"
    return Counter(window_labels).most_common(1)[0][0]


def split_into_attack_segments(labels):
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

    while idx + WINDOW_SIZE <= end:
        window_feat = features[idx:idx + WINDOW_SIZE]
        window_labels = labels[idx:idx + WINDOW_SIZE]

        label_name = majority_label(window_labels)

        windows.append({
            "features": window_feat,
            "label_name": label_name,
            "label_id": LABEL_MAP.get(label_name, -1)
        })

        idx += STRIDE

    return windows


def save_image(img, path):
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def process_dataset(csv_path):
    features, labels = load_data_no_timestamp(csv_path)

    print(f"✓ Data loaded. Shape: {features.shape}")

    print("🔄 Normalizing features...")
    features, scaler = normalize_features(features)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for target in TARGET_CLASSES:
        if target in LABEL_MAP:
            os.makedirs(os.path.join(OUTPUT_ROOT, target), exist_ok=True)

    with open(os.path.join(OUTPUT_ROOT, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    segments = split_into_attack_segments(labels)

    class_counter = Counter()
    total_saved = 0
    img_global_id = 0

    print(f"🚀 Processing segments and generating images for: {TARGET_CLASSES}...")

    for start, end in tqdm(segments):
        segment_label = labels[start]
        if segment_label not in TARGET_CLASSES:
            continue

        windows = generate_windows_in_segment(features, labels, start, end)

        for w in windows:
            label_name = w["label_name"]

            if label_name in TARGET_CLASSES:
                img = window_to_rgb_image(w["features"])

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

