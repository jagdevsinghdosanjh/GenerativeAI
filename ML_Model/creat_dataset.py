import numpy as np
import pandas as pd

# Set the number of samples and features
num_samples = 100
num_features = 5

# Generate random feature data
X = np.random.rand(num_samples, num_features)

# Generate random target data (binary classification)
y = np.random.randint(0, 2, num_samples)

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(num_features)])
data['target'] = y

# Save the DataFrame to a CSV file
data.to_csv('synthetic_dataset.csv', index=False)