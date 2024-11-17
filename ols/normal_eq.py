import numpy as np
# w = ((X.T * X)**(-1)) * X.T * y**3

def normalEqution(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
