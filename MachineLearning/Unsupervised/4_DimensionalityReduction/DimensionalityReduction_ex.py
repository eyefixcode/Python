import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to generate random mock healthcare data
def generate_healthcare_data(num_samples, num_features):
    np.random.seed(42)
    data = np.random.randn(num_samples, num_features)
    return pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(num_features)])

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
    plt.scatter(original_data.iloc[:, 0], original_data.iloc[:, 1], alpha=0.5)
    plt.title('Original Data')
    plt.xlabel(original_data.columns[0])
    plt.ylabel(original_data.columns[1])

    # Plot reduced data
    plt.subplot(1, 2, 2)
    plt.scatter(reduced_data.iloc[:, 0], reduced_data.iloc[:, 1], alpha=0.5)
    plt.title('Reduced Data (PCA)')
    plt.xlabel(reduced_data.columns[0])
    plt.ylabel(reduced_data.columns[1])

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
