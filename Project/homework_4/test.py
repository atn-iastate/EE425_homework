import numpy as np
from itertools import combinations_with_replacement


a = np.transpose(np.arange(1,5,1)).reshape(1,4)
b = np.transpose(np.arange(5,9,1)).reshape(1,4)
x = np.random.randn(4).reshape(2, 2)


def normalize(x):
    """Return the standardized matrix of x"""
    mean = np.sum(x, axis=0) / x.shape[0]
    variance = np.diag(x.T @ x)
    return (x - mean) / np.sqrt(variance)


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


def data_sim(true_parameters, error, bias, nrow=1000, seed=1):
    np.random.seed(seed)

    n = true_parameters.shape[0]
    x_tot = np.random.randn(nrow, n - 10)
    x_anc = np.zeros(nrow * 10).reshape(nrow, 10)
    for i in range(x_anc.shape[1]):
        x_anc[:, i] = x_tot[:, i + 5] + i * x_tot[:, i + 5]

    x_tot = np.concatenate((x_anc, x_tot), axis=1)
    y = x_tot @ true_parameters + np.random.randn(nrow) * error + bias

    return x_tot, y


# initial setup
m = 100
t = 80  # number of training samples

true_parameters = np.concatenate((np.random.randn(60) * 20, np.zeros(40)))

error = 1e-2 * np.linalg.norm(true_parameters)
x_sim, y_sim = data_sim(true_parameters, error, bias=10000, nrow=m)

print(x_sim.shape)