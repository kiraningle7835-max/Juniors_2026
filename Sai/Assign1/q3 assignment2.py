#3.1
import numpy as np

X = np.array([[1200, 3, 10, 5], [1500, 3, 8, 4], [1800, 4, 5, 6]])
y = np.array([300000, 360000, 420000])

N, d = X.shape
w = np.zeros((d, 1))
w0 = 0.0
#3.2
def predict(X, w, w0):
    return (X @ w) + w0
#3.3 
def solve_normal_equation(X, y):
    N = X.shape[0]
    X_augmented = np.column_stack([np.ones(N), X])
    
    weights_full = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    
    w0 = weights_full[0]
    w = weights_full[1:]
    
    return w, w0