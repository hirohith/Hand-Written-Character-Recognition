# Handwritten Character Recognition using CNN

## Project Overview

This project implements a Handwritten Character Recognition system using Convolutional Neural Networks (CNN). The model is trained on the MNIST dataset to recognize handwritten digits from 0 to 9.

The system accepts image data, processes handwritten digits, and predicts the corresponding digit with high accuracy.

---

## Objective

The objective of this project is to develop a deep learning model capable of recognizing handwritten characters and digits from images.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Seaborn

---

## Dataset

### MNIST Dataset

* Total Images: 70,000
* Training Images: 60,000
* Testing Images: 10,000
* Image Size: 28 × 28 pixels
* Classes: Digits 0–9

---

## Project Workflow

1. Load MNIST Dataset
2. Preprocess Images
3. Normalize Pixel Values
4. Reshape Images for CNN
5. Build CNN Architecture
6. Train Model
7. Evaluate Performance
8. Predict Handwritten Digits

---

## CNN Architecture

### Layers Used

1. Convolution Layer (32 Filters)
2. Max Pooling Layer
3. Flatten Layer
4. Dense Layer (128 Neurons)
5. Output Layer (10 Classes)

---

## Model Training

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Epochs: 5
* Activation Function: ReLU
* Output Activation: Softmax

---

## Results

### Test Accuracy

98.69%

### Test Loss

0.0447

The model achieved high accuracy on unseen handwritten digit images from the MNIST test dataset.

---

## Generated Outputs

* Accuracy Graph
* Confusion Matrix
* Trained Model File
* Digit Prediction System

---

## Files Included

* load_dataset.py
* preprocessing.py
* cnn_model.py
* train_model.py
* predict_digit.py
* evaluate_model.py
* confusion_matrix.py
* handwritten_digit_model.keras

---

## Future Scope

* Handwritten Alphabet Recognition
* EMNIST Dataset Integration
* OCR System Development
* Word Recognition
* Sentence Recognition
* Streamlit Web Application

---

## Conclusion

The Handwritten Character Recognition system successfully classifies handwritten digits using a Convolutional Neural Network. The project demonstrates the application of deep learning and computer vision techniques for image classification tasks and achieves an accuracy of approximately 98.69%.
