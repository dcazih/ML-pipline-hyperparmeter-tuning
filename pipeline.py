import numpy as np
import pandas as pd
import idx2numpy as i2n
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import os
print(os.getcwd())


# (1) Load dataset
# If test_fashion false -> test MNIST else test fashion-MNIST
fashion_MNIST = True
if fashion_MNIST:
    # Fashion-MNISTdata/fashsion-mnist/train-images-idx3-ubyte
    training_images = i2n.convert_from_file('data/fashion-mnist/train-images-idx3-ubyte')
    training_labels = i2n.convert_from_file('data/fashion-mnist/train-labels-idx1-ubyte')
    test_images = i2n.convert_from_file('data/fashion-mnist/t10k-images-idx3-ubyte')
    test_labels = i2n.convert_from_file('data/fashion-mnist/t10k-labels-idx1-ubyte')
else:
    # MNIST
    training_images = i2n.convert_from_file('data/mnist/train-images.idx3-ubyte')
    training_labels = i2n.convert_from_file('data/mnist/train-labels.idx1-ubyte')
    test_images = i2n.convert_from_file('data/mnist/t10k-images.idx3-ubyte')
    test_labels = i2n.convert_from_file('data/mnist/t10k-labels.idx1-ubyte')

# (2) Flatten images 
training_images = training_images.reshape(training_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Shuffle the dataset before subsampling
training_images, training_labels = shuffle(training_images, training_labels, random_state=42)
test_images, test_labels = shuffle(test_images, test_labels, random_state=42)

# Then take a smaller, random subset
subset_size = 6000  
X_train = training_images[:subset_size]
y_train = training_labels[:subset_size]
X_test = test_images[:subset_size]
y_test = test_labels[:subset_size]


# (3) Create ML Pipeline
# Define search space (num of paramters had to be reduced to avoid long computation time)
param_range_C = [0.1, 1, 10, 100]
param_range_G = [1e-4, 1e-3, 1e-2, 0.1]
param_range_D = [2, 3, 4]  
param_grid = {
    "linear": { 'svc__C': param_range_C },
    'rbf': { 'svc__C': param_range_C, 'svc__gamma': param_range_G },
    'poly': { 'svc__C': param_range_C, 'svc__gamma': param_range_G, 'svc__degree': param_range_D }
}

# Test loop params
results = {}
PCA_dimensions = [50, 100, 200]
kernels = ["linear", "rbf", 'poly']

# Perform Gridsearch for each PCA dimension and kernel
for d in PCA_dimensions:
    for k in kernels:
                
        # Create pipeline for given pca dimension and kernel
        pipe = make_pipeline(StandardScaler(), PCA(n_components=d), SVC(kernel=k))

        # Fit the pipeline to the training data using gridsearch to find best hyperparameters
        gs = GridSearchCV(pipe, param_grid[k], refit=True, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)

        # Print the best hyperparameters and accuracy given the PCA dimension and kernel
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        print(f"Best params: {gs.best_params_}")
        print(f"Test accuracy: {accuracy:.4f}")

        results[(d, k)] = {
            'accuracy': accuracy,
            'best_params': gs.best_params_,
            'confusion_matrix': cm

        }

# (4) Save the results to a text file 
table_data = []
for (n_components, kernel), info in results.items():
    table_data.append({
        'PCA Components': n_components,
        'Kernel': kernel,
        'Accuracy': info['accuracy'],
        'Best C': info['best_params'].get('svc__C', None),
        'Best Gamma': info['best_params'].get('svc__gamma', None),
        'Best Degree': info['best_params'].get('svc__degree', None),
        'Confusion Matrix': info['confusion_matrix']

    })

# Create DataFrame
df_results = pd.DataFrame(table_data)

# Save DataFrame to CSV
if fashion_MNIST: df_results.to_csv('svc_pca_results_table_fasion_mnist.csv', index=False)
else: df_results.to_csv('svc_pca_results_table.csv', index=False)