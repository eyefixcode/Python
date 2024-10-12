# Support Vector Machines application for examining disease via python

# Load in libraries
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import random
import time

# Set Matplotlib backend to 'agg' (Agg is a non-interactive backend)
import matplotlib
matplotlib.use('agg')

# Function to generate random data for each list
def generate_random_data(size, min_value, max_value):
    return [random.randint(min_value, max_value) for _ in range(size)]

# Function to track the time of execution
def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

# Loading function to simulate data loading
@track_time
def load_data():
    # Simulate data loading with a loading bar
    print("Loading data...")
    # Simulate loading delay
    data_loading_delay = 2
    for _ in range(10):
        print(".", end='', flush=True)
        time.sleep(data_loading_delay / 10)
    print("\nData loaded successfully!")
    # Define the ranges for each list
    age_data = generate_random_data(8, 25, 95)
    blood_pressure_data = generate_random_data(8, 120, 190)
    cholesterol_data = generate_random_data(8, 200, 340)
    disease_data = random.choices(['No', 'Yes'], k=8)
    # Generate mock healthcare data
    data = {
        'Age': age_data,
        'BloodPressure': blood_pressure_data,
        'Cholesterol': cholesterol_data,
        'Disease': disease_data
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# Display the mock data
df = load_data()
print("Mock Healthcare Data:")
print(df)
print("\n")

# Visualization: Pair Plot
@track_time
def visualize_pair_plot():
    pair_plot = sns.pairplot(df, hue='Disease', palette='viridis')
    pair_plot.fig.suptitle('Pair Plot of Healthcare Data', y=1.02)
    pair_plot.savefig(os.path.join(os.getcwd(), 'pair_plot.png'))
    plt.close()  # Close the pair plot figure
    print("Pair Plot saved successfully!")

visualize_pair_plot()

# Encoding target variable 'Disease'
df['Disease'] = df['Disease'].map({'Yes': 1, 'No': 0})

# Separate features (X) and target variable (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier on the full feature set
svm_full = SVC(kernel='linear', C=1.0)

# Train the model
@track_time
def train_svm():
    print("Training the model...")
    global y_pred  # Ensure y_pred is defined in the global scope
    svm_full.fit(X_train, y_train)
    y_pred = svm_full.predict(X_test)

# Make predictions on the test set
@track_time
def make_predictions():
    print("Making predictions...")
    global y_pred  # Ensure y_pred is defined in the global scope
    y_pred = svm_full.predict(X_test)

# Visualize Decision Boundaries
@track_time
def visualize_decision_boundaries():
    for i, feature1 in enumerate(X.columns):
        for j, feature2 in enumerate(X.columns):
            if i < j:
                # Initialize the SVM classifier
                svm = SVC(kernel='linear', C=1.0)

                # Train the model without feature names
                svm.fit(df[[feature1, feature2]], df['Disease'])

                # Create a new figure
                plt.figure()

                # Create a mesh grid for decision boundary visualization
                h = .02
                x_min, x_max = df[feature1].min() - 1, df[feature1].max() + 1
                y_min, y_max = df[feature2].min() - 1, df[feature2].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

                # Predict for each point in the mesh grid
                Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

                # Convert prediction values to float
                Z = Z.astype(float)

                # Reshape the predictions to match the mesh grid shape
                Z = Z.reshape(xx.shape)

                # Plot decision boundary
                plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

                # Plot data points
                sns.scatterplot(x=feature1, y=feature2, hue='Disease', data=df, palette='viridis', s=100)
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                plt.title(f'Decision Boundary for {feature1} vs {feature2}')
                plt.savefig(os.path.join(os.getcwd(), f'decision_boundary_{feature1}_{feature2}.png'))
                plt.close()  # Close the decision boundary plot figure
                print(f"Decision Boundary Plot for {feature1} vs {feature2} saved successfully!")

# Execute training, prediction, and evaluation
train_svm()
make_predictions()

# Visualize Decision Boundaries
visualize_decision_boundaries()

# Evaluate the model
@track_time
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    classification_report_str = classification_report(y_true, y_pred)

    print("SVM Model Evaluation:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report_str)

evaluate_model(y_test, y_pred)
