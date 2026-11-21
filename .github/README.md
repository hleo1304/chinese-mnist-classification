# Handwritten Chinese Character Classification (Chinese MNIST)

A comparative study of four Machine Learning and Deep Learning models for classifying grayscale images of handwritten Chinese numerals (15 classes: digits `0` through `9` and powers of ten `10`, `100`, `1,000`, `10,000`, `10⁸`).

This project highlights the limitations of 1D image flattening on classical classifiers (**Random Forest**, **MLP**, **SVM**) and demonstrates the performance gain achieved by a **2D Convolutional Neural Network (CNN)**, reaching **97.60% accuracy** on the test set.

---

## Dataset Overview

![Overview of the 15 classes in the Chinese MNIST dataset](../results/apercu_dataset.png)

---

## Experimental Results (on 3,000 test images)

| Model | Input | Selected Hyperparameters | Training Time | Test Accuracy | Macro F1-Score |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Random Forest** | 1D (`1024`) | `n_estimators=300`, `max_depth=None`, `max_features=0.1` | 33.0 s | 81.20% | 81.1% |
| **MLP** | 1D (`1024`) | `hidden_layer_sizes=(256, 128)`, `alpha=1e-4`, `lr=0.003` | 14.9 s | 87.67% | 87.8% |
| **SVM (RBF)** | 1D (`1024`) | `kernel='rbf'`, `C=10`, `gamma='scale'` | 35.1 s | 87.97% | 87.9% |
| **CNN (2D)** | 2D (`32×32×1`) | 2×(`Conv2D` + `MaxPool`), `Dense(128)`, `dropout=0.2`, SGD (`lr=0.02`, `batch=32`) | 175.0 s | **97.60%** | **97.6%** |

> Switching from 1D flattened vectors to a 2D convolutional architecture reduces the error count by a factor of 5 on the test set (**72 errors** vs. **361** for the SVM).

### Confusion Matrix of the Best Model (2D CNN)

![2D CNN Confusion Matrix](../results/confusion_cnn.png)

---

## Tech Stack

- **Language:** Python 3.11
- **Machine Learning:** `scikit-learn` (Random Forest, MLP, SVM, `GridSearchCV`)
- **Deep Learning:** `Keras` / `TensorFlow` (2D CNN, SGD with momentum)
- **Image & Data Processing:** `Pillow` (32×32 Lanczos resizing), `NumPy`, `Pandas`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Scientific Report:** PDF ([Machine_learning - Rapport - Halepidis_Lion.pdf](../Machine_learning%20-%20Rapport%20-%20Halepidis_Lion.pdf), written in LaTeX)

---

## Project Structure

```text
chinese-mnist-classification/
├── projet_ml_halepidis_lion.py              # Full pipeline: preprocessing, GridSearch, training, and evaluation
├── Chinese_MNIST.py                         # Script to generate the 15-class dataset preview
├── requirements.txt                         # Python dependencies
├── Machine_learning - Rapport - Halepidis_Lion.pdf  # Final analysis report
├── results/                                 # Dataset preview and generated confusion matrices
│   ├── apercu_dataset.png
│   ├── confusion_random_forest.png
│   ├── confusion_mlp.png
│   ├── confusion_svm.png
│   └── confusion_cnn.png
└── .github/
    └── README.md
```

---

## Setup and Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/hleo1304/chinese-mnist-classification.git](https://github.com/hleo1304/chinese-mnist-classification.git)
   cd chinese-mnist-classification
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   Download the [Chinese MNIST dataset on Kaggle](https://www.kaggle.com/datasets/gpreda/chinese-mnist), then place `chinese_mnist.csv` and the `data/` folder (containing the 15,000 `.jpg` images) at the root of the project.

4. **Run the full pipeline:**
   ```bash
   python projet_ml_halepidis_lion.py
   ```
   All 4 confusion matrices will be automatically generated in the `results/` directory.

---

## Authors

- **Léo Halepidis** & **Maxence Lion** — Machine Learning Project (Bachelor of Engineering, UMONS)
