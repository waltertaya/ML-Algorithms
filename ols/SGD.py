import numpy as np

def predict(X, w):
    return np.dot(X, w)

def cost(y, y_pred):
    return np.mean((y_pred - y)**2)

def sgd(X, w, y, epochs, learning_rate):
    m, n = X.shape
    for epoch in range(epochs):
        for i in range(m):
            x_i = X[i]
            y_pred = np.dot(x_i, w)
            gradient = x_i.T * (y[i] - y_pred) 
            w += learning_rate * gradient.reshape(w.shape)
        loss = np.mean((np.dot(X, w) - y) ** 2)
        print(f"===== Epoch {epoch + 1} loss : {loss:.4f} =====")
    return w
