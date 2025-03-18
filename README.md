# Image Classification Using Neural Networks

This project focuses on designing neural network architectures to automate both feature extraction and classification tasks for image datasets. The implementation includes:

- A Multi-Layer Perceptron (MLP) applied to pre-extracted features.
- An MLP applied directly to raw image data.
- A Convolutional Neural Network (CNN) for end-to-end image classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Architectures](#architectures)
  - [MLP on Extracted Features](#mlp-on-extracted-features)
  - [MLP on Raw Images](#mlp-on-raw-images)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [References](#references)

## Project Overview

The goal of this project is to classify images into their respective categories using neural network models. We explore different architectures to understand their performance on two distinct datasets: Fashion-MNIST and Fruits-360.

## Datasets

### Fashion-MNIST

- **Description:** Contains 70,000 grayscale images (32x32 pixels) of 10 clothing categories (e.g., shirts, shoes, bags).
- **Structure:** Pre-divided into training and testing sets.

### Fruits-360

- **Description:** Comprises approximately 55,000 RGB images of 80 different fruit classes.
- **Structure:** Pre-divided into training and testing sets.

## Architectures

### MLP on Extracted Features

An MLP is trained on feature vectors extracted using traditional methods:

- **Feature Selection:** Applied techniques to reduce dimensionality, using RandomForest Feature Selection:
  - *Fashion-MNIST:* Reduced to a maximum of 64 features.
  - *Fruits-360:* Reduced to a maximum of 128 features.
- **Architecture:**
  - Input layer corresponding to the number of selected features.
  - One or more hidden layers with a specified number of neurons, only fully connected layers
  - Output layer with neurons equal to the number of classes.
- **Activation Functions:** Typically ReLU for hidden layers and softmax for the output layer.

### MLP on Raw Images

An MLP is trained directly on raw, flattened image data:

- **Preprocessing:** Flatten images into 1D arrays and normalize pixel values.
- **Architecture:**
  - Input layer matching the total number of pixels.
  - Multiple hidden layers with specified neurons, only fully connected layers.
  - Output layer with neurons equal to the number of classes.
- **Activation Functions:** ReLU for hidden layers and softmax for the output layer.

### Convolutional Neural Network (CNN)

A CNN is designed to automatically extract spatial features from images:

- **Architecture:**
  - Convolutional layers with varying filter sizes to capture spatial hierarchies.
  - Pooling layers to reduce dimensionality.
  - Fully connected layers leading to the output layer.
- **Activation Functions:** ReLU for convolutional and fully connected layers; softmax for the output layer.

## Training and Evaluation

- **Training Parameters:**
  - Number of epochs: Defined based on convergence criteria.
  - Batch size: Set according to dataset size and hardware capabilities.
  - Optimizer: Adam or SGD with appropriate learning rates.
  - Loss Function: Cross-entropy loss for classification tasks.
- **Regularization Techniques:** Dropout, L2 regularization, or batch normalization to prevent overfitting.
- **Evaluation Metrics:** Accuracy, precision, recall, and F1-score on both training and testing sets.

## Results
Long story short the worst results were obtained on the feature based MLP.
The MLP model applied directly on the input images was the fastest with a pretty solid accuracy about >85%, but prone 
to overfitting.
The CNN approach following a typological residual network architecture was the slowest, about twice as slow as the MLP approach
the accuracy was similar but more robust overall for unseen data and less likely to lead to overfitting, or wild spiking in the
validation accuracy during training.

The performance of each model is documented, highlighting:

- Training and testing accuracy.
- Loss curves over epochs.
- Confusion matrices to visualize classification performance.
- Comparative analysis with traditional machine learning methods applied in earlier stages.
