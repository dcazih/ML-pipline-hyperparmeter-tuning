# Machine Learning Pipeline and Hyperparameter Tuning

This project implements a full machine learning pipeline for classifying images from the MNIST and Fashion-MNIST datasets. The pipeline is built using Python and scikit-learn and includes preprocessing, dimensionality reduction, model training, and performance evaluation.

## Overview

The primary goals of this project are:

- To load and preprocess the MNIST and Fashion-MNIST datasets.
- To construct a modular machine learning pipeline using scikit-learn.
- To apply two dimensionality reduction techniques: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA).
- To train Support Vector Machine classifiers with various kernels.
- To perform hyperparameter tuning using grid search.
- To evaluate the models using confusion matrices and conduct a comparative analysis across multiple configurations.

## Datasets

### MNIST

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. The dataset is split into 60,000 training images and 10,000 test images.

### Fashion-MNIST

Fashion-MNIST is a more challenging alternative to MNIST. It contains 70,000 grayscale images of Zalando fashion products, each also of size 28×28 pixels. Similar to MNIST, it is split into 60,000 training images and 10,000 test images.

Both datasets are used in their original IDX file format, imported using the `idx2numpy` package.

## Implementation

### 1. Data Import

- Import MNIST and Fashion-MNIST data from IDX files.
- Convert data to NumPy arrays using `idx2numpy`.

### 2. Preprocessing

- Flatten 28×28 images to 1D arrays of length 784 using NumPy's `reshape`.
- Standardize the features using `StandardScaler`. Fit is done on the training set only, but the transformation is applied to both train and test sets.

### 3. Dimensionality Reduction

Each dataset undergoes dimensionality reduction using:

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

For each technique, three different reduced dimensionalities are considered:

- 50 dimensions
- 100 dimensions
- 200 dimensions

Reduction is fit on the training set and applied to both training and test sets.

### 4. Classifier: Support Vector Machine

A Support Vector Classifier (SVC) is trained using the scikit-learn `SVC` class. Three kernel types are explored:

- Linear (`kernel='linear'`)
- Radial Basis Function (`kernel='rbf'`)
- Polynomial (`kernel='poly'`)

### 5. Hyperparameter Tuning

Grid search is used to find the optimal parameters for each kernel:

#### Linear Kernel
- C

#### RBF Kernel
- C
- gamma

#### Polynomial Kernel
- C
- gamma
- degree

An example parameter grid:

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'degree': [2, 3, 4]  # only for poly kernel
}
