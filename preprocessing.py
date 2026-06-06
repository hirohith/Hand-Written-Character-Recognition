from tensorflow.keras.datasets import mnist
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Before Normalization")
print("Min Pixel Value:", x_train.min())
print("Max Pixel Value:", x_train.max())

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nAfter Normalization")
print("Min Pixel Value:", x_train.min())
print("Max Pixel Value:", x_train.max())
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("\nNew Training Shape:", x_train.shape)
print("New Testing Shape:", x_test.shape)
