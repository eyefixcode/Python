import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
import matplotlib.pyplot as plt

# Generate random mock data for healthcare application
np.random.seed(42)
num_samples = 1000

# Generate normal samples representing health metrics
normal_data = pd.DataFrame({
    'HeartRate': np.random.normal(70, 10, num_samples),  # Normal heart rate
    'BloodPressure': np.random.normal(120, 10, num_samples),  # Normal blood pressure
    'Temperature': np.random.normal(98.6, 0.5, num_samples)  # Normal body temperature
})

# Generate anomalous samples representing potential health issues
anomalous_data = pd.DataFrame({
    'HeartRate': np.concatenate([np.random.normal(40, 5, int(0.02 * num_samples)),
                                 np.random.normal(120, 20, int(0.03 * num_samples))]),
    'BloodPressure': np.concatenate([np.random.normal(80, 10, int(0.02 * num_samples)),
                                     np.random.normal(160, 20, int(0.03 * num_samples))]),
    'Temperature': np.concatenate([np.random.normal(101, 0.5, int(0.02 * num_samples)),
                                   np.random.normal(105, 0.5, int(0.03 * num_samples))])
})

# Concatenate normal and anomalous samples
health_data = pd.concat([normal_data, anomalous_data], ignore_index=True)

# Initialize the Isolation Forest model
isolation_forest_model = IsolationForest(contamination=0.05)
isolation_forest_model.fit(health_data[['HeartRate', 'BloodPressure', 'Temperature']])
health_data['IsolationForestAnomaly'] = isolation_forest_model.predict(health_data[['HeartRate', 'BloodPressure', 'Temperature']])

# Initialize the Local Outlier Factor model
lof_model = LocalOutlierFactor(contamination=0.05)
health_data['LocalOutlierFactorAnomaly'] = lof_model.fit_predict(health_data[['HeartRate', 'BloodPressure', 'Temperature']])

# Initialize the Robust Covariance model
robust_covariance_model = EllipticEnvelope(contamination=0.05)
health_data['RobustCovarianceAnomaly'] = robust_covariance_model.fit_predict(health_data[['HeartRate', 'BloodPressure', 'Temperature']])

# Initialize the One-Class SVM model
one_class_svm_model = OneClassSVM(nu=0.05)
health_data['OneClassSVMAnomaly'] = one_class_svm_model.fit_predict(health_data[['HeartRate', 'BloodPressure', 'Temperature']])

# Initialize the One-Class SVM with SGD model
sgd_one_class_svm_model = SGDOneClassSVM(nu=0.05)
health_data['SGDOneClassSVMAnomaly'] = sgd_one_class_svm_model.fit_predict(health_data[['HeartRate', 'BloodPressure', 'Temperature']])

# Save visuals as image files
plt.figure(figsize=(15, 10))

# Isolation Forest
plt.subplot(2, 3, 1)
plt.scatter(health_data.index, health_data['HeartRate'], c=health_data['IsolationForestAnomaly'], cmap='viridis', marker='.')
plt.title('Isolation Forest - Heart Rate')
plt.xlabel('Index')
plt.ylabel('Heart Rate')

plt.subplot(2, 3, 2)
plt.scatter(health_data.index, health_data['BloodPressure'], c=health_data['IsolationForestAnomaly'], cmap='plasma', marker='.')
plt.title('Isolation Forest - Blood Pressure')
plt.xlabel('Index')
plt.ylabel('Blood Pressure')

plt.subplot(2, 3, 3)
plt.scatter(health_data.index, health_data['Temperature'], c=health_data['IsolationForestAnomaly'], cmap='cividis', marker='.')
plt.title('Isolation Forest - Temperature')
plt.xlabel('Index')
plt.ylabel('Temperature')

# Local Outlier Factor
plt.subplot(2, 3, 4)
plt.scatter(health_data.index, health_data['HeartRate'], c=health_data['LocalOutlierFactorAnomaly'], cmap='viridis', marker='.')
plt.title('Local Outlier Factor - Heart Rate')
plt.xlabel('Index')
plt.ylabel('Heart Rate')

plt.subplot(2, 3, 5)
plt.scatter(health_data.index, health_data['BloodPressure'], c=health_data['LocalOutlierFactorAnomaly'], cmap='plasma', marker='.')
plt.title('Local Outlier Factor - Blood Pressure')
plt.xlabel('Index')
plt.ylabel('Blood Pressure')

plt.subplot(2, 3, 6)
plt.scatter(health_data.index, health_data['Temperature'], c=health_data['LocalOutlierFactorAnomaly'], cmap='cividis', marker='.')
plt.title('Local Outlier Factor - Temperature')
plt.xlabel('Index')
plt.ylabel('Temperature')

plt.tight_layout()
plt.savefig("IsolationForest_LocalOutlierFactor.png")
plt.show()

# Save visuals as image files
plt.figure(figsize=(15, 10))

# Robust Covariance
plt.subplot(2, 3, 1)
plt.scatter(health_data.index, health_data['HeartRate'], c=health_data['RobustCovarianceAnomaly'], cmap='viridis', marker='.')
plt.title('Robust Covariance - Heart Rate')
plt.xlabel('Index')
plt.ylabel('Heart Rate')

plt.subplot(2, 3, 2)
plt.scatter(health_data.index, health_data['BloodPressure'], c=health_data['RobustCovarianceAnomaly'], cmap='plasma', marker='.')
plt.title('Robust Covariance - Blood Pressure')
plt.xlabel('Index')
plt.ylabel('Blood Pressure')

plt.subplot(2, 3, 3)
plt.scatter(health_data.index, health_data['Temperature'], c=health_data['RobustCovarianceAnomaly'], cmap='cividis', marker='.')
plt.title('Robust Covariance - Temperature')
plt.xlabel('Index')
plt.ylabel('Temperature')

# One-Class SVM
plt.subplot(2, 3, 4)
plt.scatter(health_data.index, health_data['HeartRate'], c=health_data['OneClassSVMAnomaly'], cmap='viridis', marker='.')
plt.title('One-Class SVM - Heart Rate')
plt.xlabel('Index')
plt.ylabel('Heart Rate')

plt.subplot(2, 3, 5)
plt.scatter(health_data.index, health_data['BloodPressure'], c=health_data['OneClassSVMAnomaly'], cmap='plasma', marker='.')
plt.title('One-Class SVM - Blood Pressure')
plt.xlabel('Index')
plt.ylabel('Blood Pressure')

# One-Class SVM with SGD
plt.subplot(2, 3, 6)
plt.scatter(health_data.index, health_data['Temperature'], c=health_data['SGDOneClassSVMAnomaly'], cmap='cividis', marker='.')
plt.title('SGD One-Class SVM - Temperature')
plt.xlabel('Index')
plt.ylabel('Temperature')

plt.tight_layout()
plt.savefig("RobustCovariance_OneClassSVG.png")
plt.show()


# Print the health data with anomaly predictions
print(health_data)
