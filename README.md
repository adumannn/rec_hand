# Digit Recognizer

A digit recognizer built using NumPy (no deep learning frameworks). Includes a simple GUI using Tkinter to draw and predict digits.

---

## ⚙️ Install

```bash
pip install numpy scikit-learn pillow matplotlib
```

---

## ▶️ Run the App

```bash
python nw.py
```

1. Draw a digit (0–9) in the white canvas.
2. Click **Predict** to see the model's guess.
3. Click **Clear** to reset the canvas.

---

## 🧠 Model Details

- Input: 8×8 grayscale image → flattened (64 features)
- Architecture:  
  `Input (64) → ReLU (32) → Softmax (10)`
- Optimizer: Basic Gradient Descent  
- Epochs: 1000  
- Accuracy: ~97% on test data

---

## 🖼️ Visualization

When predicting, the model input (8×8 image) is displayed using Matplotlib.

> To disable visualization, comment out the `plt.imshow()` and `plt.show()` lines.

---

## 📁 Dataset

- [scikit-learn `load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- Preprocessed to scale pixel values to [0, 1]

---

---

## 📄 License

MIT — do anything you want with it.  
