import numpy as np
from itertools import combinations_with_replacement


def normal_equations_estimate(x, y, alpha=1e-10):
    """return the estimated coefficient vector for linear regression X @ w.T + e = y"""
    n = x.shape[1]
    return np.linalg.inv(x.T @ x + alpha*np.identity(n)) @ x.T @ y


def compute_test_mse(y_test, y_prediction):

    return np.linalg.norm(y_test - y_prediction) ** 2 / (y_test.shape[0] * np.linalg.norm(y_test)**2)


def normalize(x):
    """Return the standardized matrix of x, the mean vector, and the column variances vector"""
    mean = np.sum(x, axis=0) / x.shape[0]
    variance = np.diag(x.T @ x)

    return (x - mean) / np.sqrt(variance), mean, variance


def adding_polynomial_terms(x, degree=[]):
    """Adding new terms of a polynomial in Rn w.r.t to the degree parameters"""
    n = x.shape[1]
    features = range(n)

    for i in range(len(degree)):
        j = degree[i]
        for r in combinations_with_replacement(features, j):
            new_col = np.ones(x.shape[0])
            for i in r:
                new_col = new_col * x[:, i]

            new_col = new_col.reshape(x.shape[0], 1)
            x = np.concatenate((x, new_col), axis=1)

    return x
