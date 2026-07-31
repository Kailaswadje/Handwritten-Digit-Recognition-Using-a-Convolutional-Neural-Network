# ✍️ Handwritten Digit Recognition Using a Convolutional Neural Network

Teaching a machine to read handwriting: a **CNN trained on MNIST** that classifies handwritten digits 0–9 — the canonical computer vision project, built layer by layer with the reasoning behind every convolution, pooling, and dense decision explained.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![CNN](https://img.shields.io/badge/Model-Convolutional%20NN-blueviolet)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-lightgrey)

---

## 📌 Why CNNs for Images

A dense network treats a 28×28 image as 784 unrelated numbers — throwing away the fact that **nearby pixels are related**. CNNs respect image structure:

- **Convolutions** slide small filters across the image, detecting local patterns (edges, curves, loops)
- **Pooling** compresses feature maps, buying translation tolerance
- **Depth** composes simple patterns into complex ones: edges → strokes → digit shapes

This project demonstrates that pipeline on the dataset that made it famous.

---

## 📊 Dataset — MNIST

| Property | Value |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 grayscale |
| Classes | Digits 0–9 |

Loaded directly through `keras.datasets.mnist` — no download wrangling.

---

## 🏗️ Architecture

```
Input (28×28×1)
   │
Conv2D (32 filters, 3×3) — ReLU      ← learns edges & strokes
   │
MaxPooling2D (2×2)                   ← downsamples, adds robustness
   │
Conv2D (64 filters, 3×3) — ReLU      ← learns digit parts
   │
MaxPooling2D (2×2)
   │
Flatten
   │
Dense (128) — ReLU                   ← combines features
   │
Dense (10) — Softmax                 ← P(digit) for 0–9
```

- **Loss:** categorical cross-entropy · **Optimiser:** Adam

---

## 🔬 Workflow

1. **Load & inspect** — visualising sample digits with labels
2. **Preprocess** — normalising pixel values to [0,1], reshaping to (28,28,1), one-hot encoding labels
3. **Build** — Keras Sequential CNN as above
4. **Train** — epochs with validation split; accuracy/loss curves plotted
5. **Evaluate** — test accuracy + confusion matrix across all 10 digits
6. **Error analysis** — visualising the digits the model got wrong (spoiler: many are hard for humans too)

---

## 📈 Results

| Metric | Score |
|---|---|
| Test accuracy | XX.X% |
| Most confused pair | X ↔ X |

The confusion matrix shows errors concentrate on genuinely ambiguous shapes — 4s that look like 9s, 3s that look like 5s — with sloppy handwriting to blame as often as the model.

---

## 💡 Key Takeaways

- **Convolution is feature engineering, learned** — nobody told the network what an edge is; the filters discovered edges because they help classify
- **Parameter sharing makes CNNs efficient** — one 3×3 filter scans the whole image, versus a dense layer's per-pixel weights
- **Pooling trades precision for robustness** — a digit shifted a pixel over still classifies correctly
- **Error analysis beats headline accuracy** — looking at misclassified images reveals whether the model fails reasonably
- Normalising inputs and one-hot encoding targets are small steps that make or break training stability

---

## 🧰 Tech Stack

- **Python 3**
- **TensorFlow / Keras** — CNN construction and training
- **NumPy** — array handling
- **Matplotlib / Seaborn** — digit visualisation, training curves, confusion matrix
- **Jupyter Notebook** *(GPU-heavy training runs well on Google Colab)*

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Kailaswadje/Handwritten-Digit-Recognition-Using-a-Convolutional-Neural-Network.git
cd Handwritten-Digit-Recognition-Using-a-Convolutional-Neural-Network

# Install dependencies
pip install tensorflow numpy matplotlib seaborn jupyter

# Launch
jupyter notebook
```

Or open the notebook directly in **Google Colab** for free GPU acceleration.

---

## 🔮 Possible Extensions

- [ ] Add Dropout / BatchNorm and measure the effect
- [ ] Data augmentation (rotation, shift) for robustness
- [ ] Visualise learned filters and intermediate feature maps
- [ ] Graduate to Fashion-MNIST or CIFAR-10
- [ ] Deploy as a draw-a-digit web demo

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this made CNNs click, give it a star!
