import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

# ========== Activation & Loss ==========
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(preds, targets):
    return -np.mean(np.sum(targets * np.log(preds + 1e-8), axis=1))

# ========== Load + Prep ==========
digits = load_digits()
X = digits.data / 16.0               # normalize 0–16 → 0–1
y = digits.target.reshape(-1, 1)

enc = OneHotEncoder(sparse_output=False) # sparse=false or sparse_output=false
y_onehot = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# ========== MLP Config & Init ==========
np.random.seed(42)
in_size, hid_size, out_size = 64, 32, 10

W1 = np.random.randn(in_size, hid_size) * np.sqrt(2. / in_size)
b1 = np.zeros((1, hid_size))
W2 = np.random.randn(hid_size, out_size) * np.sqrt(2. / hid_size)
b2 = np.zeros((1, out_size))

lr, epochs = 0.05, 1000

# ========== Training Loop ==========
for epoch in range(epochs):
    # forward
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # metrics
    loss = cross_entropy(a2, y_train)
    preds = np.argmax(a2, axis=1)
    labels = np.argmax(y_train, axis=1)
    acc = np.mean(preds == labels)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}  Loss: {loss:.4f}  Acc: {acc:.4f}")

    # backprop
    d2 = a2 - y_train
    d1 = (d2 @ W2.T) * relu_deriv(a1)

    # update
    W2 -= lr * (a1.T @ d2) / X_train.shape[0]
    b2 -= lr * np.sum(d2, axis=0, keepdims=True) / X_train.shape[0]
    W1 -= lr * (X_train.T @ d1) / X_train.shape[0]
    b1 -= lr * np.sum(d1, axis=0, keepdims=True) / X_train.shape[0]

# final test acc
z1 = X_test @ W1 + b1;  a1 = relu(z1)
z2 = a1 @ W2 + b2;      a2 = softmax(z2)
test_acc = np.mean(np.argmax(a2,1) == np.argmax(y_test,1))
print(f"\nTest Accuracy: {test_acc:.4f}\n")


# ========== Prediction Helper ==========
def predict_digit(x):
    """x is shape (1,64), values 0–1"""
    h = relu(x @ W1 + b1)
    o = softmax(h @ W2 + b2)
    return int(np.argmax(o))

def crop_and_center(img):
    """Crop to content bbox & pad/center to 8×8."""
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        img = ImageOps.pad(img, (8,8), method=Image.BILINEAR, centering=(0.5,0.5))
    else:
        img = Image.new("L",(8,8),255)
    return img

# ========== GUI ==========
canvas_size, brush = 200, 8

class DigitApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        btnf = tk.Frame(self.root); btnf.pack(pady=5)
        tk.Button(btnf, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        tk.Button(btnf, text="Clear",   command=self.clear).pack(side=tk.LEFT, padx=5)

        self.label = tk.Label(self.root, text="Draw a digit", font=("Arial",16))
        self.label.pack(pady=5)

        # PIL image to mirror the canvas
        self.img = Image.new("L",(canvas_size,canvas_size),"white")
        self.draw = ImageDraw.Draw(self.img)

    def paint(self, e):
        x,y = e.x,e.y
        self.canvas.create_oval(x-brush, y-brush, x+brush, y+brush, fill="black")
        self.draw.ellipse([x-brush,y-brush,x+brush,y+brush], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,canvas_size,canvas_size], fill="white")
        self.label.config(text="Draw a digit")

    def predict(self):
        # preprocess
        im = self.img.copy().convert("L")
        im = ImageOps.invert(im)
        im = crop_and_center(im)
        im = im.resize((8,8), Image.Resampling.LANCZOS)
        arr = np.array(im, dtype=np.float32) / 16.0  # back to 0–1
        x = arr.flatten().reshape(1, -1)

        # debug show
        plt.imshow(arr, cmap="gray"); plt.title("Model sees"); plt.show()

        if x.sum() < 0.1:
            self.label.config(text="Draw something first!")
            return

        p = predict_digit(x)
        self.label.config(text=f"Prediction: {p}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    DigitApp().run()
