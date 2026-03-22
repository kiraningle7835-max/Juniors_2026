import numpy as np

def load_data():
    train = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("test.csv", delimiter=",", skiprows=1)

    X_train = train[:, :4]
    y_train = train[:, 4]
    
    X_test = test[:, :4]
    y_test = test[:, 4]

    return X_train, y_train, X_test, y_test

def ols_with_intercept(X, y):
    N = X.shape[0]
    ones = np.ones((N, 1))
    X_augmented = np.concatenate([ones, X], axis=1)

    w_full = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

    w0 = w_full[0]
    w = w_full[1:]

    return w, w0

def predict_with_intercept(X, w, w0):
    return (X @ w) + w0

def compute_metrics(y, y_hat):
    mse = np.mean((y - y_hat)**2)
    
    correlation = np.corrcoef(y, y_hat)[0, 1]
    
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    return mse, correlation, r2

X_train, y_train, X_test, y_test = load_data()
w, w0 = ols_with_intercept(X_train, y_train)
y_hat = predict_with_intercept(X_test, w, w0)

mse, corr, r2 = compute_metrics(y_test, y_hat)

print(f"MSE: {mse}")
print(f"R2 Score: {r2}")