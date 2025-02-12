import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
# Load dataset
data = pd.read_csv('synthetic_dataset.csv')
    
# Split data into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
   
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)    

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
    
# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
    
# Make predictions
y_pred = model.predict(X_test)
   
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
