Image Classification with CNN (CIFAR-10)
This project implements an Image Classification model using Convolutional Neural Networks (CNNs) in TensorFlow/Keras.
The model is trained on the CIFAR-10 dataset, which contains 60,000 images across 10 different classes.

The notebook demonstrates:
Data preprocessing
Building a CNN model
Training & validation
Model evaluation on test data
Visualization of training performance

Dataset
Source: CIFAR-10 (available in TensorFlow/Keras datasets)
Size: 60,000 color images (32×32 pixels)
Training: 50,000 images
Testing: 10,000 images
Classes (10): airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Steps Performed
Import TensorFlow and load the CIFAR-10 dataset.
Normalize image pixel values (0–255 → 0–1).
Build a CNN model with:
Conv2D + ReLU layers
MaxPooling layers
Flatten + Dense layers
Softmax output layer
Compile with Adam optimizer & categorical crossentropy loss.
Train the model on training data.
Evaluate on test dataset.
Plot training vs validation accuracy and loss curves.

Results
The CNN achieved good classification performance on CIFAR-10 test data.

Accuracy and loss trends are visualized to monitor training.

The model successfully learns to classify 10 object categories.
