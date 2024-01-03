import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Create mock dataset
data = {
    'Age': [50, 35, 42, 67, 28, 55, 48],
    'BloodPressure': [120, 135, 110, 140, 95, 150, 125],
    'Cholesterol': [200, 250, 180, 270, 150, 300, 220],
    'Disease': [1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Split into features and target variable
X = df[['Age', 'BloodPressure', 'Cholesterol']]
y = df['Disease']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create logistic regression model
model = LogisticRegression()

# Train without feature names
model.fit(X_train.values, y_train)

# Prepare new data for prediction
new_data = [[53, 128, 215]]  # Restructured for correct dimensions

print("Patient Data Input to Predict Disease Occurence...")
print("Age, Blood Pressure, Cholesterol:", new_data[0][0], ",", new_data[0][1], ",", new_data[0][2])
# Make prediction
prediction = model.predict(new_data)
print("Processing...")
if prediction == 1:
    prediction_str = "Disease expected to occur"
else:
    prediction_str = "No Disease expected to occur"
# Print prediction
print("Prediction Based on Example:", prediction_str)

