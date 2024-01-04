# k means clustering unsupervised ML example 

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate mock data within the script
data = {
    "x": [1, 4, 8, 4, 6, 3, 4, 9, 4, 5],
    "y": [2, 5, 8, 6, 4, 5, 9, 2, 7, 3],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the K-means model with 2 clusters
kmeans = KMeans(n_clusters=2)

# Fit the model to the mock data
kmeans.fit(df)

# Get cluster labels
labels = kmeans.labels_

# Visualize the clusters
plt.scatter(df["x"], df["y"], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", s=200, c="red")
plt.title("K-means Clustering Results (Mock Data)")
plt.show()
