# ECA-CAE Intrusion Detection System

A multi-class network intrusion detection framework based on **ECA-CAE (Efficient Channel Attention Convolutional Autoencoder)** and **Random Forest**, supporting two independent experiments:

- **CICIDS2017** — Network traffic intrusion detection with 6 known attack classes, 5-fold cross-validation
- **Car Hacking (CAN)** — In-vehicle CAN bus intrusion detection with 5 attack classes, 5-fold cross-validation with per-class FPR reporting

---

## Project Structure

```
ECA-CAE+RF/
│
├── In-vehicle_network/                  # Car Hacking CAN bus experiments
│   ├── IDS for Known Attacks/           # 5-class supervised classification
│   │   ├── ConversionTime_Test.py       # Benchmark: measures average CSV → PNG conversion time (5-fold)
│   │   ├── Feature_Mapping.py           # CSV → 9×9×3 PNG image converter
│   │   └── model.py                     # ECA-CAE + Random Forest training & evaluation
│   │
│   └── IDS for Unknown Attacks/         # Open-set / anomaly detection
│       ├── Feature Mapping.py           # CSV → 9×9×3 PNG image converter
│       └── model.py                     # ECA-CAE + anomaly detection model
│
└── extra-vehicle_network/               # CICIDS2017 network traffic experiments
    ├── ConversionTime_Test.py           # Benchmark: measures average CSV → PNG conversion time (5-fold)
    ├── Feature_Mapping.py               # CSV → 9×9×3 PNG image converter
    └── model.py                         # ECA-CAE + Random Forest training & evaluation
```

---

## Requirements

Python 3.8+ is recommended. Install all dependencies with:

```bash
pip install torch torchvision pillow numpy pandas scikit-learn matplotlib seaborn tqdm scipy
```

For GPU acceleration, install PyTorch matching your CUDA version:

```bash
# Example: CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Datasets

### Dataset 1 — CICIDS2017 (Network Traffic Intrusion Detection)

CICIDS2017 is released by the Canadian Institute for Cybersecurity (CIC) at the University of New Brunswick. It contains realistic labeled network traffic captured over five days in July 2017, covering benign traffic and multiple modern attack types. Features (80 dimensions) are extracted using the CICFlowMeter tool.

**Download Links:**

| Source | URL |
|--------|-----|
| Official CIC page | https://www.unb.ca/cic/datasets/ids-2017.html |
| Kaggle mirror | https://www.kaggle.com/datasets/cicdataset/cicids2017 |

**Citation:**
> Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP 2018*.

**Classes used in this project:**

| Label | Class Name  | Description              |
|:-----:|:-----------:|:------------------------:|
| 0     | BENIGN      | Normal traffic           |
| 1     | DoS         | Denial-of-Service attack |
| 2     | PortScan    | Port scanning            |
| 3     | BruteForce  | Brute-force attack       |
| 4     | WebAttack   | Web-based attack         |
| 5     | Bot         | Botnet traffic           |

Raw CSV data must be pre-processed into `9×9×3` PNG images and placed under `CICIDS2017_images/` in the corresponding class subdirectory. Feature dimensionality is **80**. If `feature_columns.pkl` is provided, real feature names will appear in importance plots; otherwise names default to `Feature_0` through `Feature_79`.

---

### Dataset 2 — Car Hacking Dataset (In-Vehicle CAN Bus Intrusion Detection)

The Car Hacking Dataset is published by the Hacking and Countermeasure Research Lab (HCRL) at Korea University. CAN traffic was collected via the OBD-II port from a real vehicle (Hyundai YF Sonata) while message injection attacks were performed. The dataset covers four attack types — DoS, fuzzy, gear spoofing, and RPM spoofing — alongside normal CAN traffic. Each record contains a CAN ID (1 dimension) and 8 data bytes, giving **9 features** in total. Each dataset contains approximately 300 attack injections over 30–40 minutes of total traffic.

**Download Links:**

| Source | URL |
|--------|-----|
| Official HCRL page | https://ocslab.hksecurity.net/Datasets/car-hacking-dataset |
| Kaggle mirror | https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset |

**Citation:**
> Song, H. M., Woo, J., & Kim, H. K. (2020). In-vehicle network intrusion detection using deep convolutional neural network. *Vehicular Communications*, 21, 100198.

**Classes used in this project:**

| Label | Class Name    | Description                           |
|:-----:|:-------------:|:-------------------------------------:|
| 0     | DoS           | Flooding attack via dominant CAN ID   |
| 1     | Gear          | Gear gauge spoofing attack            |
| 2     | Fuzzy         | Fuzzy / random message injection      |
| 3     | RPM           | RPM gauge spoofing attack             |
| 4     | Normal        | Benign CAN bus traffic                |

Raw CSV data must be pre-processed into `9×9×3` PNG images and placed under `Car_Hacking_images/` in the corresponding class subdirectory. If `feature_columns.pkl` is absent, feature names default to `ID`, `Data0` through `Data7`.

---

## Data Preprocessing

Both datasets must be converted from raw CSV format to `9×9×3` PNG images before running the experiments. The converter scripts are named `Feature_Mapping.py` and are located inside each experiment folder.

### Step 1 — Prepare raw CSV files

Place your raw CSV files in the corresponding dataset directories expected by each script:

```
# CICIDS2017 (extra-vehicle network)
./dataset/CICIDS2017_with_Timestamp.csv

# Car Hacking CAN (in-vehicle network)
./dataset/Car_Hacking_with_Timestamp.csv
```

The CICIDS2017 CSV must contain a `Label` column and a `Timestamp` column. The CAN CSV must contain `Timestamp`, `ID`, `Data0`–`Data7`, and `Label` columns.

### Step 2 — Convert CICIDS2017 to images

```bash
python extra-vehicle_network/Feature_Mapping.py
```

**How it works:**

Each record has 80 features. A sliding window of **3 consecutive records** (stride = 3, no overlap) forms one sample. The resulting `3×80 = 240` values are padded to 243 and mapped into a `9×9×3` RGB image using the following index formula:

```
flat_index → channel  c = flat_index // 81
           → row      i = (flat_index % 81) // 9
           → column   j = (flat_index % 81) % 9
```

To keep the dataset balanced, majority classes are capped at a configurable window limit (`MAJORITY_CLASS_LIMITS`), while minority classes (WebAttack, Bot, Infiltration) are fully retained.

**Output:**

```
CICIDS2017_images/
├── 0/ … 5/       # PNG images grouped by class label
├── scaler.pkl    # Fitted MinMaxScaler
└── feature_columns.pkl
```

### Step 3 — Convert Car Hacking CAN to images

```bash
python "In-vehicle_network/IDS for Known Attacks/Feature_Mapping.py"
```

**How it works:**

Each record has 9 features (CAN ID + Data0–Data7). A sliding window of **27 consecutive records** (stride = 27, no overlap) forms one sample. The `27×9` window maps directly into a `9×9×3` RGB image using:

```
timestep t → channel  c = t // 9
           → row      i = t % 9
feature  f → column   j = f
```

The script first segments the data by attack type (splitting on label boundaries), then generates non-overlapping windows within each segment to prevent label leakage across attack transitions.

**Output:**

```
Car_Hacking_images/
├── 0/ … 4/       # PNG images grouped by class label
├── scaler.pkl    # Fitted MinMaxScaler
└── feature_columns.pkl
```

### Conversion time benchmark (optional)

`ConversionTime_Test.py` measures the average time to convert a single raw record into a `9×9×3` PNG image. It uses **5-fold cross-validation** and reports `mean ± std` per sample across folds, along with single-threaded throughput. Run it after the images have been generated to validate conversion efficiency:

```bash
# CICIDS2017
python extra-vehicle_network/ConversionTime_Test.py

# Car Hacking CAN
python "In-vehicle_network/IDS for Known Attacks/ConversionTime_Test.py"
```

Example output:

```
📊  5-Fold Benchmark Results
=======================================================
  Per-fold means (µs): ['12.34', '11.98', '12.10', '12.45', '12.22']

  ✅ Mean ± Std (per sample):
     12.2180 ± 0.1721  µs
     0.012218 ± 0.000172  ms
  Throughput: ~81,843 samples/sec  (single-threaded)
```

The benchmark includes a **warm-up phase** (3 rounds over the first 50 samples per fold) to eliminate JIT and cache cold-start effects before the timed measurement begins.

---

## Model Architecture

### ECA Module (Efficient Channel Attention)

The ECA module generates channel-wise attention weights via global average pooling and a lightweight 1D convolution, avoiding fully connected layers to keep the parameter count minimal.

```
Input feature map (C × H × W)
  ↓  Global Average Pooling  →  (C × 1 × 1)
  ↓  1D Conv (adaptive kernel size k)
  ↓  Sigmoid
Channel attention weights  →  element-wise multiply with input
```

### ECA-CAE Architecture

```
Input  (3 × 9 × 9)
  ↓  Conv2d(3→16)  + BN + ReLU  →  ECA
  ↓  Conv2d(16→32) + BN + ReLU  →  ECA
  ↓  Conv2d(32→64) + BN + ReLU  →  ECA   ←  Latent features (64 × 9 × 9)
  ↓  ConvTranspose2d(64→32) + BN + ReLU
  ↓  ConvTranspose2d(32→16) + BN + ReLU
  ↓  ConvTranspose2d(16→3)  + Sigmoid
Output (reconstructed image  3 × 9 × 9)
```

The CAE is trained unsupervised with **MSE reconstruction loss**. The 64-channel latent feature maps are then flattened and fed to a **Random Forest** classifier for supervised multi-class classification.

---

## Hyperparameters

### CICIDS2017 (`eca_cae_multiclass_only.py`)

| Parameter        | Value | Description                       |
|:----------------:|:-----:|:---------------------------------:|
| `batch_size`     | 64    | Mini-batch size                   |
| `cae_epochs`     | 50    | CAE training epochs               |
| `learning_rate`  | 0.001 | Adam optimizer learning rate      |
| `k_folds`        | 5     | Number of CV folds                |
| `n_estimators`   | 150   | Random Forest number of trees     |
| `random_state`   | 42    | Global random seed                |
| `feature_dim`    | 80    | Original feature dimensionality   |

### Car Hacking CAN (`eca_cae_can_multiclass.py`)

| Parameter        | Value | Description                       |
|:----------------:|:-----:|:---------------------------------:|
| `batch_size`     | 64    | Mini-batch size                   |
| `cae_epochs`     | 1     | CAE training epochs (adjust as needed) |
| `learning_rate`  | 0.001 | Adam optimizer learning rate      |
| `k_folds`        | 5     | Number of CV folds                |
| `n_estimators`   | 50    | Random Forest number of trees     |
| `random_state`   | 42    | Global random seed                |
| `feature_dim`    | 9     | Original feature dimensionality   |

All parameters are centrally managed in the `CONFIG` dictionary at the top of each script.

---

## Usage

**Full workflow (recommended order):**

```bash
# ── Extra-vehicle network (CICIDS2017) ──────────────────────────────
# Step 1 — Convert raw CSV to images
python extra-vehicle_network/Feature_Mapping.py

# Step 2 — (Optional) Benchmark conversion speed
python extra-vehicle_network/ConversionTime_Test.py

# Step 3 — Run ECA-CAE + RF experiment
python extra-vehicle_network/model.py


# ── In-vehicle network (Car Hacking CAN) — Known Attacks ────────────
# Step 1 — Convert raw CSV to images
python "In-vehicle_network/IDS for Known Attacks/Feature_Mapping.py"

# Step 2 — (Optional) Benchmark conversion speed
python "In-vehicle_network/IDS for Known Attacks/ConversionTime_Test.py"

# Step 3 — Run ECA-CAE + RF experiment
python "In-vehicle_network/IDS for Known Attacks/model.py"


# ── In-vehicle network (Car Hacking CAN) — Unknown Attacks ──────────
# Step 1 — Convert raw CSV to images
python "In-vehicle_network/IDS for Unknown Attacks/Feature Mapping.py"

# Step 2 — Run anomaly detection model
python "In-vehicle_network/IDS for Unknown Attacks/model.py"
```

Both scripts execute the following four-step pipeline:

1. **Multi-class Classification** — 5-fold stratified cross-validation. Each fold independently trains the ECA-CAE and Random Forest, then reports Accuracy, Macro-F1, Precision, and Recall (plus per-class and macro FPR for CAN) with 95% confidence intervals. Confusion matrix and ROC curves are saved from the last fold.

2. **Feature Importance Analysis** — Gradient-based importance scores computed per class via backpropagation through the CAE. Scores are mapped back to original feature dimensions and displayed as merged subplots, saved in PNG, PDF, and EPS formats.

3. **Spatial Saliency Maps** — Per-class averaged gradient heatmaps rendered on the 9×9 pixel grid to highlight which spatial regions (i.e., feature positions) the model is most sensitive to.

4. **ECA Channel Weight Heatmaps** — Visualizes the attention weight distribution across all three ECA encoder layers, showing which channels the model focuses on for each attack class.

---

## Output Files

All results are written to `results_CAE_RF/` inside each experiment folder.

| Path | Content |
|------|---------|
| `results_CAE_RF/models/best_cae.pth` | Best CAE weights (last fold) |
| `results_CAE_RF/plots/cae_training_history.png` | Train / validation MSE loss curves |
| `results_CAE_RF/plots/multiclass_confusion_matrix.png` | Confusion matrix (last fold) |
| `results_CAE_RF/plots/multiclass_roc.png` | Per-class ROC curves with AUC (last fold) |
| `results_CAE_RF/plots/feature_importance_merged.{png,pdf,eps}` | Merged feature importance subplots |
| `results_CAE_RF/reports/imp_{ClassName}.csv` | Full ranked feature importance per class |
| `results_CAE_RF/reports/final_5fold_metrics.txt` | Global + per-class Mean ± Std across 5 folds (CAN only) |
| `results_CAE_RF/attention_maps/saliency_9x9_{ClassName}.png` | 9×9 spatial saliency heatmap per class |
| `results_CAE_RF/attention_maps/eca_weights_{ClassName}.png` | ECA layer-wise channel weight heatmaps |

---

## Pipeline Overview

```
Raw CSV data
     │
     ▼
Feature_Mapping.py                   Feature_Mapping.py
(extra-vehicle_network)              (In-vehicle_network / Known Attacks)
Window=3, Stride=3, 80-dim           Window=27, Stride=27, 9-dim
Balanced class sampling              Segment-aware windowing
     │                                     │
     └──────────────┬──────────────────────┘
                    ▼
           9×9×3 PNG images
           [ConversionTime_Test.py: optional speed benchmark]
                    │
                    ▼
             ┌─────────────┐
             │  ECA-CAE    │  Unsupervised pre-training
             │  model.py   │  (MSE reconstruction loss)
             └──────┬──────┘
                    │  Flattened latent features
                    ▼
             ┌─────────────┐
             │    Random   │  Supervised multi-class
             │    Forest   │  classification
             └──────┬──────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
Classification  Feature        Attention
Metrics & Plots Importance     Visualization
(CM, ROC, FPR)  (Grad-based)   (Saliency, ECA)
```

---

## Notes

- Each fold overwrites `best_cae.pth`, `cae_training_history.png`, and all plot files; only results from the **last fold** are retained on disk.
- Feature importance is derived from **reconstruction error gradients** rather than Random Forest built-in importances. This reflects the CAE's pixel-level sensitivity and is mapped back to original feature names for interpretability.
- Without a GPU, training will be significantly slower. For quick verification, consider reducing `cae_epochs` or `n_estimators`.
- ECA channel weight heatmaps are reshaped purely for visualization (e.g., 64 channels → 8×8 grid) and do not imply any spatial correspondence.
- For the CAN experiment, `final_5fold_metrics.txt` records Mean ± Std of Precision, Recall, F1-Score, and FPR for each class across all five folds, providing a statistically robust summary suitable for academic reporting.
