import numpy as np

X = np.array([
    [1200, 3, 10, 5],
    [1500, 3, 8, 4],
    [1800, 4, 5, 6]
])

y = np.array([
    [300000],
    [360000],
    [420000]
])

N = X.shape[0]
d = X.shape[1]

print(f"Number of observations (N): {N}")
print(f"Number of features (d): {d}")
print(f"Feature Matrix X shape: {X.shape}")
print(f"Target Vector y shape: {y.shape}")