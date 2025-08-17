import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ========== Activation ==========
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
X = digits.data / 16.0
y = digits.target.reshape(-1, 1)

enc = OneHotEncoder(sparse_output=False)
y_onehot = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# ========== MLP Config ==========
np.random.seed(42)
in_size, hid_size, out_size = 64, 32, 10

W1 = np.random.randn(in_size, hid_size) * np.sqrt(2. / in_size)
b1 = np.zeros((1, hid_size))
W2 = np.random.randn(hid_size, out_size) * np.sqrt(2. / hid_size)
b2 = np.zeros((1, out_size))

lr, epochs = 0.05, 300   # fewer epochs for faster start

# ========== Training ==========
for epoch in range(epochs):
    # forward
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)

    # loss + acc
    loss = cross_entropy(a2, y_train)
    preds = np.argmax(a2, axis=1)
    labels = np.argmax(y_train, axis=1)
    acc = np.mean(preds == labels)

    # backprop
    d2 = a2 - y_train
    d1 = (d2 @ W2.T) * relu_deriv(a1)

    W2 -= lr * (a1.T @ d2) / X_train.shape[0]
    b2 -= lr * np.sum(d2, axis=0, keepdims=True) / X_train.shape[0]
    W1 -= lr * (X_train.T @ d1) / X_train.shape[0]
    b1 -= lr * np.sum(d1, axis=0, keepdims=True) / X_train.shape[0]

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Loss {loss:.4f} | Acc {acc:.4f}")

# final test acc
z1 = X_test @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
test_acc = np.mean(np.argmax(a2,1) == np.argmax(y_test,1))
print(f"\nTest Accuracy: {test_acc:.4f}\n")

# ========== Prediction Helper ==========
def predict_digit(x):
    h = relu(x @ W1 + b1)
    o = softmax(h @ W2 + b2)
    return int(np.argmax(o))

def crop_and_center(img):
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
        img = ImageOps.pad(img, (8,8), method=Image.BILINEAR, centering=(0.5,0.5))
    else:
        img = Image.new("L",(8,8),255)
    return img

# ========== GUI Calculator ==========
canvas_size, brush = 200, 8

class CalculatorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Calculator")

        self.expression = ""   # expression string

        # Display
        self.label = tk.Label(self.root, text="", font=("Arial",20), anchor="e", width=20, relief="sunken")
        self.label.pack(pady=5)

        # Drawing canvas
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        btnf = tk.Frame(self.root); btnf.pack(pady=5)
        tk.Button(btnf, text="Predict Digit", command=self.predict).grid(row=0, column=0, padx=5)
        tk.Button(btnf, text="Clear", command=self.clear_canvas).grid(row=0, column=1, padx=5)
        tk.Button(btnf, text="+", command=lambda: self.add_op("+")).grid(row=0, column=2, padx=5)
        tk.Button(btnf, text="-", command=lambda: self.add_op("-")).grid(row=0, column=3, padx=5)
        tk.Button(btnf, text="*", command=lambda: self.add_op("*")).grid(row=0, column=4, padx=5)
        tk.Button(btnf, text="/", command=lambda: self.add_op("/")).grid(row=0, column=5, padx=5)
        tk.Button(btnf, text="=", command=self.evaluate).grid(row=0, column=6, padx=5)
        tk.Button(btnf, text="C", command=self.clear_all).grid(row=0, column=7, padx=5)

        # PIL image for canvas
        self.img = Image.new("L",(canvas_size,canvas_size),"white")
        self.draw = ImageDraw.Draw(self.img)

    def paint(self, e):
        x,y = e.x,e.y
        self.canvas.create_oval(x-brush, y-brush, x+brush, y+brush, fill="black")
        self.draw.ellipse([x-brush,y-brush,x+brush,y+brush], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,canvas_size,canvas_size], fill="white")

    def clear_all(self):
        self.expression = ""
        self.label.config(text="")

    def predict(self):
        im = self.img.copy().convert("L")
        im = ImageOps.invert(im)
        im = crop_and_center(im)
        im = im.resize((8,8), Image.Resampling.LANCZOS)
        arr = np.array(im, dtype=np.float32) / 16.0
        x = arr.flatten().reshape(1, -1)

        if x.sum() < 0.1:
            self.label.config(text="Draw a digit!")
            return

        digit = predict_digit(x)
        self.expression += str(digit)
        self.label.config(text=self.expression)
        self.clear_canvas()

    def add_op(self, op):
        self.expression += op
        self.label.config(text=self.expression)

    def evaluate(self):
        try:
            result = str(eval(self.expression))
            self.label.config(text=result)
            self.expression = result
        except Exception:
            self.label.config(text="Error")
            self.expression = ""

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    CalculatorApp().run()
