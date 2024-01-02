# logistic regression python EXAMPLE
# uses logistic regression on a mock dataset to showcase how machine learning can be used in healthcare to classify disease probability.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


data = {
    'Age': [50, 35, 42, 67, 28, 55, 48],
    'BloodPressure': [120, 135, 110, 140, 95, 150, 125],
    'Cholesterol': [200, 250, 180, 270, 150, 300, 220],
    'Disease': [1, 0, 1, 1, 0, 1, 0]  # 1 for disease present, 0 for absent
}
df = pd.DataFrame(data)

df.to_csv("healthcare_mock_data.csv", index=False)  # Save as CSV

X = df[['Age', 'BloodPressure', 'Cholesterol']]  # Features
y = df['Disease']  # Target variable

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


new_data = [[53, 128, 215]]  # Example prediction
prediction = model.predict(new_data)
print("Prediction:", prediction)  # Output: [1] (prediction of disease presence)


