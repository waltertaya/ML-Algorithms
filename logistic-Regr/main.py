import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(w, X):
    y_pred = np.dot(w, X.T)
    return sigmoid(y_pred)

# Batch gradient ascent
def BGA(w, learning_rate, epochs, X, y):
    for i in range(epochs):
        y_pred = predict(w, X)
        error = y - y_pred
        w += learning_rate * np.dot(error, X)
        if i % 10 == 0:
            print(f"Epoch: {i}, Error: {np.mean(error)}")
    return w

path = 'datasets/Cancer_Data.csv'
data = pd.read_csv(path)
data = data.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

y = data['diagnosis']
X = data.drop('diagnosis', axis=1)
X = (X - X.mean()) / X.std()  # Normalize features
X['bias'] = 1
X = X.values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize weights
w = np.zeros(X_train.shape[1])

# Train model
w = BGA(w, 0.01, 100, X_train, y_train)

# Predict
y_pred = predict(w, X_test)
y_pred = np.round(y_pred)

# Evaluate using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
