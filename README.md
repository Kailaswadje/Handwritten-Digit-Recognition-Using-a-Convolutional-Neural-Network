# Handwritten Digit Recognition — CNN on MNIST (Deep Learning Fundamentals)

The third entry in a deep learning progression — Perceptron → ANN → **Convolutional Neural Network** — moving from tabular loan data to real images. A CNN trained on the full **MNIST dataset (70,000 handwritten digits)** reaches **98.56% test accuracy** in just 5 epochs, the moment in the series where deep learning starts looking like the technology everyone talks about.

## Objective

Dense layers treat an image as a flat list of numbers, throwing away the fact that nearby pixels are related. This project introduces the architecture built specifically to fix that: convolution to detect local patterns (edges, curves), pooling to compress them, and dense layers to combine them into a final digit classification — the same blueprint behind image recognition at any scale.

## Dataset

| Property | Detail |
|---|---|
| Source | MNIST (Keras built-in) |
| Training images | 60,000 × 28×28 grayscale |
| Test images | 10,000 × 28×28 grayscale |
| Classes | 10 (digits 0–9) |
| Preprocessing | Reshaped to `(28,28,1)` for CNN input · pixel values normalized 0–255 → 0–1 · labels one-hot encoded |

## Architecture

```
Input (28×28×1)
   → Conv2D(32 filters, 3×3 kernel, ReLU)   # detects edges, curves, corners
   → MaxPooling2D(2×2)                      # compresses feature maps, reduces computation
   → Flatten()                              # 2D feature maps → 1D vector
   → Dense(128 units, ReLU)                 # combines detected features
   → Dense(10 units, Softmax)               # probability distribution over digits 0–9
```

- **Optimizer**: Adam · **Loss**: categorical cross-entropy · **Epochs**: 5 · **Batch size**: 32
- **Validation split**: 20% of training data held out during training (48,000 train / 12,000 val / 10,000 test — three-way split, not just train/test)

## Training Progress (5 epochs)

| Epoch | Train Accuracy | Validation Accuracy |
|---|---|---|
| 1 | 95.05% | 97.71% |
| 2 | 98.18% | 98.23% |
| 3 | 98.83% | 98.33% |
| 4 | 99.21% | 98.52% |
| 5 | **99.45%** | **98.45%** |

## Final Test Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **98.56%** |
| Test Loss | 0.0468 |

The model correctly classifies roughly **986 of every 1,000** unseen handwritten digits — a single Conv2D layer, five epochs, and no data augmentation.

## Reading the Training Curve Honestly

Training accuracy climbs steadily to 99.45% while validation accuracy plateaus around 98.3–98.5% by epoch 3 — a small but visible **generalization gap opening up**. The model is very slightly starting to memorize training examples faster than it learns new general patterns by epoch 5. It's mild here (only ~1 point), but it's the same overfitting signature that motivates dropout, data augmentation, and early stopping in production CNNs — the natural next additions to this notebook.

## Why CNNs Beat Dense Networks on Images

- **Parameter efficiency**: a 3×3 convolutional filter scans the whole image with the same 9 weights, instead of a dense layer needing a separate weight for every one of 784 pixels
- **Translation invariance**: a digit's "7-ness" is detected the same way whether it's centered or shifted slightly
- **Hierarchical features**: stacking more Conv2D layers (a natural extension) lets the network build from edges → shapes → whole digits

## Tech Stack

`Python` · `TensorFlow / Keras` (Sequential, Conv2D, MaxPooling2D, Flatten, Dense) · `NumPy` · `matplotlib`

## Repository Structure

```
├── MNIST_CNN_Digit_Recognition.ipynb   # Full CNN pipeline notebook
└── README.md
```

*(MNIST loads directly from `keras.datasets.mnist` — no external CSV required.)*

## How to Run

```bash
pip install tensorflow numpy matplotlib
jupyter notebook MNIST_CNN_Digit_Recognition.ipynb
```

## Related Repos — The Deep Learning Progression

1. [`loan-approval-perceptron`](../loan-approval-perceptron) — single neuron, tabular data
2. [`loan-approval-ann-keras`](../loan-approval-ann-keras) — hidden layers, tabular data
3. **This repo** — convolutional layers, image data

---

*Author: Kailas Wadje — MSc Data Science & AI, University of Liverpool*
