# solve.py
# Students must complete the TODO sections

import numpy as np

# ============================================================
# OLS IMPLEMENTATIONS
# ============================================================

def ols_with_intercept(X, y):
    """
    Ordinary Least Squares WITH intercept.

    Parameters
    ----------
    X : numpy array (N,d)
        Feature matrix

    y : numpy array (N,)
        Target values

    Returns
    -------
    w  : slope vector (d,)
    w0 : intercept scalar
    """

    # TODO:
    # Step 1: Add a column of ones to X to represent intercept

    intercepts = np.ones( X.shape[0] ) 

    X = np.c_[ intercepts , X ] 

    # Step 2: Use the normal equation
    #
    #        w = (X^T X)^(-1) X^T y

    X_transpose = np.transpose(X) 

    w = (( np.linalg.inv( X_transpose @ X ) ) @ X_transpose ) @ y 

    #
    # Step 3: Separate intercept from weight vector

    w0 = w[0] 

    w = w[ 1 : ]  

    return w , w0     


    raise NotImplementedError


def ols_no_intercept(X, y):
    """
    OLS WITHOUT intercept.

    Use the normal equation:

        w = (X^T X)^(-1) X^T y
    """

    # TODO:
    # Implement closed-form solution

    X_transpose = np.transpose(X) 

    w = (( np.linalg.inv( X_transpose @ X ) ) @ X_transpose ) @ y 

    return w 

    raise NotImplementedError


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_with_intercept(X, w, w0):
    """
    Predict y = Xw + w0
    """

    # TODO:
    # return predicted values

    y_hat = (X @ w) + w0 

    return y_hat 

    raise NotImplementedError


def predict_no_intercept(X, w):
    """
    Predict y = Xw
    """

    # TODO

    y_hat = (X @ w)

    return y_hat

    raise NotImplementedError


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y, y_hat):
    """
    Compute the following metrics:

    1. Mean Squared Error (MSE)

        MSE = mean((y - y_hat)^2)

    2. Correlation

    3. Squared Correlation

    4. R^2 score
    """


    # TODO

    MSE = np.mean( (y_hat - y) ** 2 )

    corr = np.corrcoef( y , y_hat )[0, 1]

    corr_sqr = corr ** 2

    a = np.sum(( y - y_hat ) ** 2) 
    b = np.sum(( y - np.mean(y) ) ** 2) 

    R_sqr = 1 - a/b 

    return MSE , corr , corr_sqr , R_sqr 


    raise NotImplementedError


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """
    Load dataset from CSV files.

    CSV format:

    size,bedrooms,age,distance,price

    First 4 columns = features
    Last column = target
    """

    train = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    test = np.loadtxt("test.csv", delimiter=",", skiprows=1)


    # TODO:
    # Separate X and y

    # n = train.shape[0] # number of rows in the train array 

    X_train = train[ : , : -1 ] 
    y_train = train[ : , -1 ] 

    X_test = test[ : , : -1 ] 
    y_test = test[ : , -1 ] 

    return X_train , y_train , X_test , y_test 

    

    # for i in range(0,n):
    #     for j in range(0,4):
    #         X[i][j] = train[i][j] 

    # for i in range(0,n):
    #     y[i][0] = train[i][4]

    raise NotImplementedError

