import os
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pickle

WINDOW_SIZE = 3
STRIDE = 3
FEATURE_DIM = 80
IMG_H, IMG_W, CHANNELS = 9, 9, 3

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

MINORITY_CLASSES = ["WebAttack", "Bot", "Infiltration"]

MAJORITY_CLASS_LIMITS = {
    "BENIGN": 18423,
    "DoS": 7234,
    "PortScan": 5436,
    "BruteForce": None
}

OUTPUT_ROOT = "CICIDS2017_images"

def load_and_preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    original_len = len(df)
    df = df.dropna()
    if 'Timestamp' in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)
    exclude_cols = ['Timestamp', 'Label', ' Timestamp', ' Label']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    features_df = df[feature_cols]
    for col in feature_cols:
        if features_df[col].dtype == 'object':
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    features = features_df.values.astype(np.float32)
    labels = df["Label"].values if "Label" in df.columns else df[" Label"].values
    features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
    return features, labels, feature_cols

def normalize_features(features):
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, scaler

def window_to_rgb_image(window):
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

def image_to_features(img, scaler=None):
    flat_features = []
    for idx in range(240):
        c = idx // (IMG_H * IMG_W)
        temp = idx % (IMG_H * IMG_W)
        i = temp // IMG_W
        j = temp % IMG_W
        flat_features.append(img[i, j, c])
    features_flat = np.array(flat_features)
    window = features_flat.reshape(WINDOW_SIZE, FEATURE_DIM)
    if scaler is not None:
        window = scaler.inverse_transform(window)
    return window

def majority_label(window_labels):
    most_common = Counter(window_labels).most_common(1)[0][0]
    return LABEL_MAP.get(most_common, -1)

def save_image(img, path):
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)

def generate_windows_for_class(features, labels, class_name, max_windows=None):
    windows = []
    idx = 0
    while idx + WINDOW_SIZE <= len(features):
        if max_windows and len(windows) >= max_windows:
            break
        window_feat = features[idx:idx + WINDOW_SIZE]
        window_labels = labels[idx:idx + WINDOW_SIZE]
        label_id = majority_label(window_labels)
        windows.append({
            "features": window_feat,
            "label": label_id
        })
        idx += STRIDE
    return windows

def process_class(features, labels, class_name, img_id_start, max_windows=None):
    label_id = LABEL_MAP[class_name]
    output_dir = os.path.join(OUTPUT_ROOT, str(label_id))
    os.makedirs(output_dir, exist_ok=True)
    windows = generate_windows_for_class(features, labels, class_name, max_windows)
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
    features, labels, feature_cols = load_and_preprocess_csv(csv_path)
    features_norm, scaler = normalize_features(features)
    scaler_path = os.path.join(OUTPUT_ROOT, "scaler.pkl")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    feature_cols_path = os.path.join(OUTPUT_ROOT, "feature_columns.pkl")
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    label_counts = Counter(labels)
    class_data = {}
    for label_name in LABEL_MAP.keys():
        mask = labels == label_name
        class_data[label_name] = {
            "features": features_norm[mask],
            "labels": labels[mask]
        }
    for cid in LABEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, str(cid)), exist_ok=True)
    class_counter = Counter()
    img_id = 0
    for class_name in MINORITY_CLASSES:
        if class_name in class_data:
            count, img_id = process_class(
                class_data[class_name]["features"],
                class_data[class_name]["labels"],
                class_name,
                img_id,
                max_windows=None
            )
            class_counter[LABEL_MAP[class_name]] = count
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
    csv_path = "./dataset/CICIDS2017_with_Timestamp.csv"
    scaler, class_counter, feature_cols = generate_full_image_dataset(csv_path)
    total_images = 0
    for cid in sorted(class_counter.keys()):
        count = class_counter[cid]
        total_images += count
    test_class = 0
    test_dir = os.path.join(OUTPUT_ROOT, str(test_class))
    if os.path.exists(test_dir):
        test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        if test_images:
            test_img_path = os.path.join(test_dir, test_images[0])
            test_img = np.array(Image.open(test_img_path)).astype(np.float32) / 255.0
            recovered_features = image_to_features(test_img, scaler)
