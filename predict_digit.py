from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model("models/handwritten_digit_model.keras")

# Load MNIST test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Select a test image
image = x_test[0]

# Preprocess
image = image / 255.0
image = image.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(image)

predicted_digit = np.argmax(prediction)

print("Predicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])
