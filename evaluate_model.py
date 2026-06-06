from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

# Load saved model
model = load_model("models/handwritten_digit_model.keras")

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)

print("\n===== MODEL EVALUATION =====")
print("Test Accuracy:", accuracy * 100)
print("Test Loss:", loss)
