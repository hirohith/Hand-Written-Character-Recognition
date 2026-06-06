from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create images folder
os.makedirs("images", exist_ok=True)

# Load model
model = load_model("models/handwritten_digit_model.keras")

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Predict
predictions = model.predict(x_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.savefig("images/confusion_matrix.png")

print("Confusion Matrix Saved Successfully")

plt.show()
