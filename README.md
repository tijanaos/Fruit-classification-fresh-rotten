# 🍎 Fruit Classification: Fresh or Rotten

Automated fruit classification system that recognizes fruit type (apple, banana, orange) and assesses its condition (fresh or rotten) using Convolutional Neural Networks.

**Authors:** Tijana Ostojić (RA 83/2022) · David Toth (RA 84/2022)  
**Assistant:** Teodor Vidaković

---

## Results

| Model | Accuracy | Loss | Precision | Recall |
|-------|----------|------|-----------|--------|
| Transfer Learning (MobileNetV2) | **98.90%** | 0.0456 | 98.69% | 98.54% |
| Custom CNN | 92.23% | 0.3046 | 91.07% | 92.33% |

---

## Dataset

- **Source:** [Kaggle — Fruits Fresh and Rotten for Classification](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification/data)
- **Total images:** ~13,000
- **Classes (6):** `freshapples`, `freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`, `rottenoranges`

| Split | Images | Share |
|-------|--------|-------|
| Train | ~6,910 | 53% |
| Validation | ~1,728 | 13% |
| Test | 4,362 | 34% |

The dataset comes pre-split into `Train/` and `Test/` folders on disk. The validation set is created dynamically from the Train folder using an 80/20 split during training.

---

## Project Structure

```
fruit-fresh-rotten/
├── data/
│   └── raw/
│       └── dataset/
│           ├── Train/          # Training data (6 class folders)
│           └── Test/           # Test data (6 class folders)
├── models/
│   ├── custom_cnn/
│   │   └── best.keras
│   └── mobilenet/
│       └── best.keras
├── reports/
│   └── metrics/
│       ├── test_metrics_custom.json
│       ├── test_metrics_mobilenet.json
│       ├── confusion_matrix_custom.csv
│       └── confusion_matrix_mobilenet.csv
├── scripts/
│   ├── download_data.py        # Download dataset via Kaggle CLI
│   ├── train_custom.py         # Train Custom CNN
│   ├── train_mobilenet.py      # Train MobileNetV2 (Transfer Learning)
│   └── evaluate.py             # Evaluate model on test set
├── src/
│   └── fruitcls/
│       ├── config.py           # Central configuration (paths, hyperparameters)
│       ├── data/
│       │   └── loader.py       # Data loading, normalization, augmentation
│       ├── models/
│       │   ├── custom_cnn.py   # Custom CNN architecture
│       │   └── mobilenet_transfer.py  # MobileNetV2 Transfer Learning
│       └── eval/
│           └── evaluate.py     # Metrics: accuracy, precision, recall, confusion matrix
├── tests/
│   ├── test_imports.py
│   ├── test_data_pipeline.py
│   └── test_train_smoke.py
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/tijanaos/Fruit-classification-fresh-rotten
cd fruit-fresh-rotten
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Configure your Kaggle API credentials first (`~/.kaggle/kaggle.json`), then run:

```bash
python scripts/download_data.py
```

This downloads and extracts the dataset into `data/raw/dataset/`.

---

## Training

### Custom CNN

```bash
python scripts/train_custom.py --epochs 10
```

Optional arguments:

```
--epochs    Number of training epochs (default: 10)
--img-size  Input image size (default: 224)
--batch     Batch size (default: 32)
```

### Transfer Learning (MobileNetV2)

```bash
python scripts/train_mobilenet.py --epochs 10
```

Both scripts save the best model (by validation loss) to `models/` automatically.

---

## Evaluation

Evaluate a trained model on the test set:

```bash
# Evaluate Custom CNN
python scripts/evaluate.py --model custom

# Evaluate Transfer Learning model
python scripts/evaluate.py --model mobilenet

# Save metrics and confusion matrix to reports/metrics/
python scripts/evaluate.py --model custom --save
python scripts/evaluate.py --model mobilenet --save
```

---

## Methodology

### Preprocessing (applied to all splits)
- **Resize** to 224×224 pixels
- **Normalization** — pixel values scaled from [0, 255] to [0.0, 1.0]
- **Automatic labels** — inferred from folder names via Keras

### Augmentation (training set only)
| Transform | Parameter |
|-----------|-----------|
| Random Flip | horizontal |
| Random Rotation | max ±8° (factor 0.08) |
| Random Zoom | max 10% (factor 0.10) |
| Random Contrast | max 15% (factor 0.15) |

Each epoch, every image is transformed differently at load time. Original files on disk remain unchanged.

### Custom CNN Architecture

```
Input (224×224×3)
  → Block 1: Conv2D(32)  + BatchNorm + ReLU + MaxPool  → 112×112×32
  → Block 2: Conv2D(64)  + BatchNorm + ReLU + MaxPool  →  56×56×64
  → Block 3: Conv2D(128) + BatchNorm + ReLU + MaxPool  →  28×28×128
  → Block 4: Conv2D(256) + ReLU + GlobalAvgPool        →  256
  → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.3)
  → Dense(6, Softmax)
```

19 layers total. Uses Global Average Pooling instead of Flatten to reduce parameter count and overfitting.

### Transfer Learning (MobileNetV2)

MobileNetV2 pretrained on ImageNet (1.2M images, 1,000 classes) is used as a frozen feature extractor. A new classification head is added and trained for the 6-class task. This approach yields significantly better results (~98.9%) due to rich visual representations learned from large-scale pretraining.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 3e-4) |
| Loss | Categorical Cross-Entropy + Label Smoothing (0.1) |
| Batch size | 32 |
| Callbacks | ModelCheckpoint, EarlyStopping (patience=6), ReduceLROnPlateau (factor=0.5, patience=2) |

---

## Key Findings

- **Transfer Learning outperforms Custom CNN** by ~6.7 percentage points due to ImageNet pretraining.
- **Biggest error:** 140 `rottenapples` misclassified as `rottenoranges` — rotten fruit of different types develops visually similar mold patterns.
- **Best class:** `rottenbanana` achieves 99.21% precision / 98.22% recall — bananas have a distinctive shape and browning pattern that is easy to distinguish.
- **Weakest class:** `rottenoranges` precision at 66.49% — the model over-predicts this class, receiving misclassified samples from multiple other classes.

---

## Running Tests

```bash
pytest tests/
```

---

## Requirements

```
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
pytest
tqdm
kaggle
```
