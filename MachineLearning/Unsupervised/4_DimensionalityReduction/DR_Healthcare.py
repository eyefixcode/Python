# Dimensionality healthcare example using python, utilizing mock healthcare data

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to generate random mock healthcare data
def generate_healthcare_data(num_samples, num_features):
    np.random.seed(42)
    
    # Generating mock data for healthcare features
    data = {
        'Age': np.random.randint(18, 80, num_samples),
        'Blood Pressure': np.random.randint(80, 180, num_samples),
        'Cholesterol Level': np.random.randint(100, 300, num_samples),
        'BMI': np.random.uniform(18.5, 35, num_samples),
        'Heart Rate': np.random.randint(60, 100, num_samples)
    }

    return pd.DataFrame(data)

# Function to perform dimensionality reduction using PCA
def perform_pca(data, num_components):
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(data)
    return pd.DataFrame(reduced_data, columns=[f"PC_{i+1}" for i in range(num_components)])

# Function to plot the results
def plot_results(original_data, reduced_data):
    plt.figure(figsize=(12, 6))

    # Plot original data
    plt.subplot(1, 2, 1)
    for i, feature in enumerate(original_data.columns):
        plt.scatter(np.full_like(original_data[feature], i), original_data[feature], label=feature, alpha=0.5)
    plt.title('Original Data')
    plt.xlabel('Features')
    plt.ylabel('Feature Values')
    plt.xticks(range(len(original_data.columns)), original_data.columns)
    plt.legend()

    # Plot reduced data
    plt.subplot(1, 2, 2)
    for i, pc in enumerate(reduced_data.columns):
        plt.scatter(np.full_like(reduced_data[pc], i), reduced_data[pc], label=pc, alpha=0.5)
    plt.title('Reduced Data (PCA)')
    plt.xlabel('Principal Components')
    plt.ylabel('PC Values')
    plt.xticks(range(len(reduced_data.columns)), reduced_data.columns)
    plt.legend()

    # Save the plot
    plt.savefig('dimensionality_reduction_plot.png')
    plt.show()

# Main script
if __name__ == "__main__":
    # Generate random mock healthcare data (100 samples, 5 features)
    original_data = generate_healthcare_data(100, 5)

    # Print original data
    print("Original Data:")
    print(original_data)

    # Perform dimensionality reduction using PCA (2 components)
    num_components = 2
    reduced_data = perform_pca(original_data, num_components)

    # Print reduced data
    print("\nReduced Data (PCA):")
    print(reduced_data)

    # Plot and save the results
    plot_results(original_data, reduced_data)
