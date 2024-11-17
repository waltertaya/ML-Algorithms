import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from BGD import bgd
from SGD import sgd
from normal_eq import normalEqution

df = pd.read_csv('datasets/weatherHistory.csv')

y = df['Temperature (C)'].values

X = df[
    ['Humidity', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
    ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45,)

y_train = y_train.reshape([77162, 1])

X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
y_train = (y_train - y_train.mean()) / y_train.std()

w = np.zeros((5, 1))
learning_rate = 0.00001
epochs = 6

print('------ Batch Gradient Descent ----------')

w = bgd(X_train, w, y_train, epochs, learning_rate)
print(f'Weights: {w}')

print('------ Stochastic Gradient Descent ----------')
learning_rate = 0.0001
epochs = 5
w = sgd(X_train, w, y_train, epochs, learning_rate)
print(f'Weights: {w}')

print('------ Normal Equation ----------')
w = normalEqution(X_train, y_train)
print(f'Weights: {w}')
# predicted = X_test @ w
print('------ Predictions ----------')
predicted = X_test @ w
loss = np.mean((predicted - y_test) ** 2)
print(f'Loss: {loss}')
