import numpy as np

def predict(X, w):
    return np.dot(X, w)

def cost(y, y_pred):
    # return 0.5 * np.sum((y_pred - y)**2)
    return np.mean((y_pred - y)**2)

def bgd(X, w, y, epochs, learning_rate):

    for epoch in range(epochs):
        y_pred = predict(X, w)
        loss = cost(y, y_pred)

        if epoch % 1 == 0:
            print(f'===== Epoch {epoch + 1} loss : {loss:.4f} =====')

        gradient = np.dot(X.T, (y - y_pred))
        w += learning_rate * gradient
    
    return w