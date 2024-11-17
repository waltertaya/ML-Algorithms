import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
temperature, humidity, sunny
30, 40, 1
25, 50, 0
20, 60, 0
15, 70, 1
10, 80, 1
'''


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logistic_regression(X, y, alpha, num_iter):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.zeros(n)
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - alpha * gradient
    return theta


def predict(X, theta):
    return sigmoid(np.dot(X, theta))


def main():
    data = {
        'temperature': [30, 25, 20, 15, 10],
        'humidity': [40, 50, 60, 70, 80],
        'sunny': [1, 0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    X = df[['temperature', 'humidity', 'sunny']].values
    y = df['sunny'].values
    X = np.insert(X, 0, 1, axis=1)
    theta = logistic_regression(X, y, 0.01, 10000)
    print(theta)
    print(predict(X, theta))

    # plot decision boundary
    plt.scatter(df['temperature'], df['humidity'], c=df['sunny'])
    plt.xlabel('Temperature')
    plt.ylabel('Humidity')
    x_values = [10, 30]
    y_values = [(-theta[0] - theta[1] * x) / theta[2] for x in x_values]
    plt.plot(x_values, y_values)
    plt.show()


if __name__ == '__main__':
    main()
