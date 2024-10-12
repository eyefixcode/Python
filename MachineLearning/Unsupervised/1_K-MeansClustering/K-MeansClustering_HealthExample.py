import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate mock health data within the script
data = {
    "age": [25, 40, 65, 38, 52, 21, 30, 62, 48, 55],
    "bmi": [22, 28, 31, 25, 30, 19, 24, 32, 27, 26],
    "blood_pressure": [120, 135, 150, 118, 142, 110, 125, 155, 130, 138],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the K-means model with 3 clusters (representing potential risk groups)
kmeans = KMeans(n_clusters=3)

# Fit the model to the mock health data
kmeans.fit(df)

# Get cluster labels
labels = kmeans.labels_

# Visualize the clusters with legend
plt.scatter(df["age"], df["bmi"], c=labels)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", s=200, c="red")

# Create the legend
plt.legend(
    title="Legend",
    handles=[
        plt.scatter([], [], c="blue", label="Cluster 1"),
        plt.scatter([], [], c="green", label="Cluster 2"),
        plt.scatter([], [], c="orange", label="Cluster 3"),
        plt.scatter([], [], marker="x", c="red", label="Centroids"),
    ],
    loc="best",
)

plt.title("K-means Clustering of Health Indicators (Mock Data)")
plt.xlabel("Age")
plt.ylabel("BMI")
plt.show()

# Further analysis and interpretation (example)
# - Analyze characteristics of each cluster (mean age, BMI, blood pressure)
print("Cluster means:")
print(df.groupby("labels").mean())
