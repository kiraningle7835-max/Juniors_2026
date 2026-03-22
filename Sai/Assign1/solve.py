# solve.py
# Implementation of Linear Regression from Scratch using Ordinary Least Squares (OLS)
import numpy as np

# ============================================================
# OLS IMPLEMENTATIONS
# ============================================================

def ols_with_intercept(X, y):
    """
    Ordinary Least Squares WITH intercept (Task 4.2)[cite: 37].
    """
    # Step 1: Prepend a column of ones to X to account for the intercept w0[cite: 38].
    N = X.shape[0]
    ones = np.ones((N, 1))
    X_augmented = np.concatenate([ones, X], axis=1)

    # Step 2: Solve for weights using the Normal Equation: w = (X^T X)^-1 X^T y[cite: 30, 39].
    # Using np.linalg.inv for inversion and @ for matrix multiplication.
    w_full = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

    # Step 3: Separate intercept (w0) from weight vector (w)[cite: 18, 19].
    w0 = w_full[0]   # The first element corresponds to the column of ones
    w = w_full[1:]   # The remaining elements are the feature weights

    return w, w0


def ols_no_intercept(X, y):
    """
    OLS WITHOUT intercept (Task 4.2)[cite: 40].
    Computes weights using the Normal Equation without modifying X.
    """
    # w = (X^T X)^-1 X^T y 
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_intercept(X, w, w0):
    """
    Calculate y_hat using Xw + w0 (Task 4.3)[cite: 23, 42].
    """
    return (X @ w) + w0


def predict_no_intercept(X, w):
    """
    Calculate y_hat using Xw (Task 4.3)[cite: 43].
    """
    return X @ w


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y, y_hat):
    """
    Calculate the performance of the model (Task 4.4)[cite: 45].
    """
    # 1. Mean Squared Error (MSE): Average of squared differences[cite: 46].
    mse = np.mean((y - y_hat)**2)

    # 2. Correlation: Relationship between true and predicted values[cite: 46].
    # Use np.corrcoef to get the Pearson correlation coefficient.
    correlation = np.corrcoef(y, y_hat)[0, 1]

    # 3. Squared Correlation
    squared_correlation = correlation**2

    # 4. R^2 Score: Coefficient of determination[cite: 46].
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "MSE": mse,
        "Correlation": correlation,
        "Squared Correlation": squared_correlation,
        "R2": r2
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load CSV files and split into features (X) and target (y) (Task 4.1)[cite: 35].
    """
    # Load raw data from CSV files[cite: 35].
    train = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("test.csv", delimiter=",", skiprows=1)

    # Split: First 4 columns are features (X), last column is price (y)[cite: 8, 9].
    X_train = train[:, :4]
    y_train = train[:, 4]
    
    X_test = test[:, :4]
    y_test = test[:, 4]

    return X_train, y_train, X_test, y_test