# K-Nearest Neighbors application to examining disease via python

# Load in libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import time

# Loading function to simulate data loading
def load_data():
    # Simulate data loading with a loading bar
    print("Loading data...")
    for _ in tqdm(range(10), desc="Loading", position=0, leave=True):
        time.sleep(0.2)  # Simulate loading delay
    print("\nData loaded successfully!")

    # Generate mock healthcare data
    data = {
        'Age': [25, 35, 45, 55, 65, 75, 85, 95],
        'BloodPressure': [120, 130, 140, 150, 160, 170, 180, 190],
        'Cholesterol': [200, 220, 240, 260, 280, 300, 320, 340],
        'Disease': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# Display the mock data
df = load_data()
print("Mock Healthcare Data:")
print(df)
print("\n")

# Visualization: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Cholesterol', hue='Disease', data=df, palette='viridis', s=100)
plt.title('Scatter Plot of Age vs Cholesterol')
plt.savefig(os.path.join(os.getcwd(), 'scatter_plot.png'))
plt.show()

# Visualization: Pair Plot
pair_plot = sns.pairplot(df, hue='Disease', palette='viridis')
pair_plot.fig.suptitle('Pair Plot of Healthcare Data', y=1.02)
pair_plot.savefig(os.path.join(os.getcwd(), 'pair_plot.png'))
plt.show()

# Separate features (X) and target variable (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
print("Training the model...")
knn.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions...")
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("KNN Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report_str)
