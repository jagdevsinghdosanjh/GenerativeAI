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

# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# # Load the variables
# X_train = np.load('X_train')
# y_train = np.load('y_train')

# # Train a logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Example predictions and evaluation (for illustrative purposes)
# y_pred = model.predict(X_train)
# accuracy = accuracy_score(y_train, y_pred)
# print(f'Accuracy: {accuracy:.2f}')
