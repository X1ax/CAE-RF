# ECA-CAE Intrusion Detection System

A multi-class network intrusion detection framework based on **ECA-CAE (Efficient Channel Attention Convolutional Autoencoder)** and **Random Forest**, supporting two independent experiments:

- **CICIDS2017** — Network traffic intrusion detection with 6 known attack classes, 5-fold cross-validation
- **Car Hacking (CAN)** — In-vehicle CAN bus intrusion detection with 5 attack classes, 5-fold cross-validation with per-class FPR reporting

---

## Project Structure

```
.
├── eca_cae_multiclass_only.py       # CICIDS2017 experiment entry point
├── eca_cae_can_multiclass.py        # Car Hacking CAN experiment entry point
│
├── CICIDS2017_images/               # CICIDS2017 image dataset (prepare manually)
│   ├── 0/                           # BENIGN
│   ├── 1/                           # DoS
│   ├── 2/                           # PortScan
│   ├── 3/                           # BruteForce
│   ├── 4/                           # WebAttack
│   ├── 5/                           # Bot
│   └── feature_columns.pkl          # Feature name list (optional, 80-dim)
│
├── Car_Hacking_images/              # CAN image dataset (prepare manually)
│   ├── 0/                           # DoS
│   ├── 1/                           # Gear Spoofing
│   ├── 2/                           # Fuzzy
│   ├── 3/                           # RPM Spoofing
│   ├── 4/                           # Normal
│   └── feature_columns.pkl          # Feature name list (optional, 9-dim)
│
└── results_CAE_RF/                  # Output directory (auto-created)
    ├── models/                      # Saved model weights
    ├── plots/                       # Training curves, confusion matrices, ROC curves,
    │                                #   feature importance plots
    ├── reports/                     # Metric reports and feature importance CSVs
    └── attention_maps/              # Saliency maps and ECA channel weight heatmaps
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

**Run CICIDS2017 experiment:**

```bash
python eca_cae_multiclass_only.py
```

**Run Car Hacking CAN experiment:**

```bash
python eca_cae_can_multiclass.py
```

Both scripts execute the following four-step pipeline:

1. **Multi-class Classification** — 5-fold stratified cross-validation. Each fold independently trains the ECA-CAE and Random Forest, then reports Accuracy, Macro-F1, Precision, and Recall (plus per-class and macro FPR for CAN) with 95% confidence intervals. Confusion matrix and ROC curves are saved from the last fold.

2. **Feature Importance Analysis** — Gradient-based importance scores computed per class via backpropagation through the CAE. Scores are mapped back to original feature dimensions and displayed as merged subplots, saved in PNG, PDF, and EPS formats.

3. **Spatial Saliency Maps** — Per-class averaged gradient heatmaps rendered on the 9×9 pixel grid to highlight which spatial regions (i.e., feature positions) the model is most sensitive to.

4. **ECA Channel Weight Heatmaps** — Visualizes the attention weight distribution across all three ECA encoder layers, showing which channels the model focuses on for each attack class.

---

## Output Files

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
Raw CSV data  →  Pre-processing  →  9×9×3 PNG images
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │  ECA-CAE    │  Unsupervised pre-training
                                   │             │  (MSE reconstruction loss)
                                   └──────┬──────┘
                                          │  Flattened latent features
                                          ▼
                                   ┌─────────────┐
                                   │    Random   │  Supervised multi-class
                                   │    Forest   │  classification
                                   └──────┬──────┘
                                          │
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                   Classification    Feature          Attention
                   Metrics & Plots   Importance       Visualization
                   (CM, ROC, FPR)    (Grad-based)     (Saliency, ECA)
```

---

## Notes

- Each fold overwrites `best_cae.pth`, `cae_training_history.png`, and all plot files; only results from the **last fold** are retained on disk.
- Feature importance is derived from **reconstruction error gradients** rather than Random Forest built-in importances. This reflects the CAE's pixel-level sensitivity and is mapped back to original feature names for interpretability.
- Without a GPU, training will be significantly slower. For quick verification, consider reducing `cae_epochs` or `n_estimators`.
- ECA channel weight heatmaps are reshaped purely for visualization (e.g., 64 channels → 8×8 grid) and do not imply any spatial correspondence.
- For the CAN experiment, `final_5fold_metrics.txt` records Mean ± Std of Precision, Recall, F1-Score, and FPR for each class across all five folds, providing a statistically robust summary suitable for academic reporting.
