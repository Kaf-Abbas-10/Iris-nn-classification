# Iris Flower Classification using Neural Networks 🌸

This project implements a Neural Network-based classifier to predict the species of iris flowers using the classic **Iris dataset**. The dataset consists of three classes of flowers: *Setosa*, *Versicolor*, and *Virginica*, based on four features.

## 📁 File Overview

- `Iris-Classification.ipynb`: Main Jupyter notebook containing the complete pipeline — data loading, preprocessing, model training, evaluation, and prediction.

## 📊 Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains:
- **150 samples**
- **3 classes** (`Setosa`, `Versicolor`, `Virginica`)
- **4 features**:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

## 🧠 Model Summary

- A simple **feedforward neural network** is used.
- Model architecture typically includes:
  - Input layer (4 nodes for the features)
  - One or more hidden layers with ReLU activation
  - Output layer with 3 nodes (Softmax for multi-class classification)

## 🚀 Project Workflow

1. **Data Loading & Exploration**
   - Load Iris dataset from `sklearn.datasets`.
   - Visualize feature distributions and class balance.

2. **Preprocessing**
   - Normalize features.
   - One-hot encode class labels.
   - Split into training and test sets.

3. **Model Building**
   - Define the neural network using Keras or TensorFlow.
   - Compile with `categorical_crossentropy` and `Adam` optimizer.

4. **Training & Evaluation**
   - Train the model on the training set.
   - Evaluate using accuracy and confusion matrix.

5. **Prediction**
   - Test predictions on new data or test set.
   - Visualize results with plots.

## ✅ Results

- Achieved **high classification accuracy** (>95%) on the test set.
- Model shows strong performance due to the simplicity and separability of the Iris dataset.

## 📦 Requirements

Install dependencies via pip:

```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow
