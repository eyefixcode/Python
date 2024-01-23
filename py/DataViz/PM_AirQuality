import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Generate mock air quality data
cities = ['City' + str(i) for i in range(1, 11)]
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

data = {'City': np.random.choice(cities, 1000),
        'Month': np.random.choice(months, 1000),
        'AirQuality': np.random.randint(0, 101, 1000)}

df = pd.DataFrame(data)

# Reshape data for heatmap
heatmap_data = df.pivot_table(index='City', columns='Month', values='AirQuality', aggfunc='mean')

# Create a heatmap using seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.1f', linewidths=.5)
plt.title('Air Quality Heatmap by City and Month')
plt.show()
