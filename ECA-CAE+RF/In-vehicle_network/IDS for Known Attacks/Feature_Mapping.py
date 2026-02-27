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
FEATURE_DIM = 9

IMG_H, IMG_W, CHANNELS = 9, 9, 3

LABEL_MAP = {
    "DoS": 0,
    "Gear": 1,
    "Fuzzy": 2,
    "RPM": 3,
    "Normal": 4
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

OUTPUT_ROOT = "./Car_Hacking_images"


def load_and_sort_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    feature_cols = ["ID"] + [f"Data{i}" for i in range(8)]
    features = df[feature_cols].values.astype(np.float32)
    labels = df["Label"].values

    return features, labels


def normalize_features(features):
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, scaler


def window_to_rgb_image(window):
    
    img = np.zeros((IMG_H, IMG_W, CHANNELS), dtype=np.float32)

    for t in range(WINDOW_SIZE):
        c = t // 9  # channel (0, 1, 2)
        i = t % 9  # row (0-8)
        for f in range(FEATURE_DIM):
            img[i, f, c] = window[t, f]

    return img


def image_to_features(img, scaler=None):
    
    window = np.zeros((WINDOW_SIZE, FEATURE_DIM), dtype=np.float32)

    for t in range(WINDOW_SIZE):
        c = t // 9  # channel
        i = t % 9  # row
        for f in range(FEATURE_DIM):
            window[t, f] = img[i, f, c]

    
    if scaler is not None:
        window = scaler.inverse_transform(window)

    return window


def majority_label(window_labels):
    return LABEL_MAP[Counter(window_labels).most_common(1)[0][0]]


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

        windows.append({
            "features": window_feat,
            "label": majority_label(window_labels)
        })

        idx += STRIDE  

    return windows


def save_image(img, path):
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


def generate_full_image_dataset(csv_path):
    
    features, labels = load_and_sort_csv(csv_path)

    
    print("🔄 Normalizing features...")
    features, scaler = normalize_features(features)

    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    
    scaler_path = os.path.join(OUTPUT_ROOT, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")

    
    feature_cols = ["ID"] + [f"Data{i}" for i in range(8)]
    feature_cols_path = os.path.join(OUTPUT_ROOT, "feature_columns.pkl")
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Feature columns saved to {feature_cols_path}")

    
    segments = split_into_attack_segments(labels)

    
    for cid in LABEL_MAP.values():
        os.makedirs(os.path.join(OUTPUT_ROOT, str(cid)), exist_ok=True)

    class_counter = Counter()
    img_id = 0

    print("🚀 Generating non-overlapping RGB images...")
    for start, end in tqdm(segments):
        windows = generate_windows_in_segment(features, labels, start, end)

        for w in windows:
            img = window_to_rgb_image(w["features"])
            label_id = w["label"]

            save_path = os.path.join(
                OUTPUT_ROOT,
                str(label_id),
                f"img_{img_id}.png"
            )

            save_image(img, save_path)
            class_counter[label_id] += 1
            img_id += 1

    return scaler, class_counter


def load_scaler(scaler_path=None):
    
    if scaler_path is None:
        scaler_path = os.path.join(OUTPUT_ROOT, "scaler.pkl")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return scaler


def test_image_to_features_conversion():
    
    print("\n" + "=" * 60)
    print("🧪 Testing image-to-features conversion...")
    print("=" * 60)

    
    try:
        scaler = load_scaler()
    except FileNotFoundError:
        print("❌ Scaler not found. Please run the main process first.")
        return

    
    test_class = 4  
    test_dir = os.path.join(OUTPUT_ROOT, str(test_class))

    if not os.path.exists(test_dir):
        print(f"❌ Directory {test_dir} not found.")
        return

    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    if not test_images:
        print(f"❌ No images found in {test_dir}")
        return

    
    test_img_path = os.path.join(test_dir, test_images[0])
    test_img = np.array(Image.open(test_img_path)).astype(np.float32) / 255.0

    
    recovered_features_norm = image_to_features(test_img, scaler=None)

    
    recovered_features_orig = image_to_features(test_img, scaler=scaler)

    print(f"✓ Test image: {test_images[0]}")
    print(f"✓ Image shape: {test_img.shape}")
    print(f"✓ Recovered features shape: {recovered_features_orig.shape}")
    print(f"✓ Normalized feature range: [{recovered_features_norm.min():.4f}, {recovered_features_norm.max():.4f}]")
    print(f"✓ Original feature range: [{recovered_features_orig.min():.4f}, {recovered_features_orig.max():.4f}]")
    print("\n✅ Image-to-features conversion successful!")

    
    print(f"\n📊 Sample recovered features (first 3 rows, first 5 columns):")
    print(recovered_features_orig[:3, :5])


if __name__ == "__main__":
    csv_path = "./dataset/Car_Hacking_with_Timestamp.csv" 

    print("=" * 60)
    print("CAN Dataset to RGB Image Converter")
    print("=" * 60)

    scaler, class_counter = generate_full_image_dataset(csv_path)

    print("\n📊 Final image dataset statistics:")
    for cid in sorted(class_counter.keys()):
        print(
            f"Class {cid} ({INV_LABEL_MAP[cid]:10s}): {class_counter[cid]:6d} images"
        )

    total_images = sum(class_counter.values())
    print(f"{'Total':16s}: {total_images:6d} images")
    print(f"\nImages saved to: {OUTPUT_ROOT}/")


    test_image_to_features_conversion()
