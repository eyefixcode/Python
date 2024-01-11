import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Generate mock healthcare data
np.random.seed(42)
num_samples = 100
num_features = 5

# Creating random data with a correlation structure
data = np.random.randn(num_samples, num_features) + np.linspace(0, 5, num_features)
data[:, 2] = data[:, 0] + data[:, 1]  # Introducing correlation

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Visualize the explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8, align='center')
plt.title('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.grid(True)
plt.savefig('explained_variance_ratio.png')
plt.show()

# Scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('scree_plot.png')
plt.show()

# Biplot (2D)
plt.figure(figsize=(8, 6))
for i in range(len(explained_variance_ratio)):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.8)
    plt.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, f'Feature {i+1}', color='g')

plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('PCA Biplot (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.savefig('biplot_2d.png')
plt.show()

# Save other necessary plots as needed for your example

# Print original data for manual comparison
print("Original Data:")
print(pd.DataFrame(data, columns=[f"Feature {i+1}" for i in range(num_features)]))

# Print PCA-transformed data for manual comparison
print("\nPCA-transformed Data:")
print(pd.DataFrame(pca_result, columns=[f"Principal Component {i+1}" for i in range(num_features)]))
