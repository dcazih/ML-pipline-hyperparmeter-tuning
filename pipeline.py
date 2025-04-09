import numpy as np
import pandas as pd
import idx2numpy as i2n
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# (1) Load the MNIST dataset
training_images = i2n.convert_from_file('data/train-images-idx3-ubyte/train-images.idx3-ubyte')
training_labels = i2n.convert_from_file('data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images = i2n.convert_from_file('data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte')
test_labels = i2n.convert_from_file('data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

print("Training Images Shape:", training_images.shape)
print("Training Labels Shape:", training_labels.shape)
print("Test Images Shape:", test_images.shape)
print("Test Labels Shape:", test_labels.shape)

# (2) Flatten images 
training_images = training_images.reshape(training_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Print the new shapes of images after flattening
print("Flattened Training Images Shape:", training_images.shape)
print("Flattened Test Images Shape:", test_images.shape)

# (3) Create ML Pipeline
# Define search space 
param_range_C = [0.01, 0.1, 1, 10, 100, 500, 1000, 5000]
param_range_G = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
param_range_D = [2, 3, 4, 5, 6, 7, 8, 9]
param_grid = {
    'linear': { 'svc__C': param_range_C },
    'rbf': { 'svc__C': param_range_C, 'svc__gamma': param_range_G },
    'poly': { 'svc__C': param_range_C, 'svc__gamma': param_range_G, 'svc__degree': param_range_D }
}

# reduce the size of the dataset for faster computation
X_train = training_images[:210]
y_train = training_labels[:210]
X_test = test_images[:210]  
y_test = test_labels[:210]



# Test loop params
results = {}
PCA_dimensions = [50, 100, 200]
kernels = ['poly']

# Perform Gridsearch for each PCA dimension and kernel
for d in PCA_dimensions:
    for k in kernels:
                
        # Create pipeline for given pca dimension and kernel
        pipe = make_pipeline(StandardScaler(), PCA(n_components=d), SVC(kernel=k))

        # Fit the pipeline to the training data using gridsearch to find best hyperparameters
        gs = GridSearchCV(pipe, param_grid[k], refit=True, cv=5, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)

        # Print the best hyperparameters and accuracy given the PCA dimension and kernel
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Best params: {gs.best_params_}")
        print(f"Test accuracy: {accuracy:.4f}")

        results[(d, k)] = {
            'accuracy': accuracy,
            'best_params': gs.best_params_
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
    })

# Create DataFrame
df_results = pd.DataFrame(table_data)

# Save DataFrame to CSV
df_results.to_csv('svc_pca_results_table.csv', index=False)