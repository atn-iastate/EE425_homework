import numpy as np


def data_sim(true_parameters, mean_x, error, nrow=80, seed=1):

    n = true_parameters.shape[0]
    np.random.seed(seed)
    tmp = np.random.randn(n, n)
    evals = np.concatenate((100 * np.random.randn(round(n/8)), np.random.randn(n - round(n/8)))) ** 2
    covariance_matrix = tmp @ np.diag(evals) @ np.transpose(tmp)

    x_tot = np.zeros((nrow, n), dtype=float)
    for i in range(nrow):
        x_tot[i, :] = np.random.multivariate_normal(mean_x, covariance_matrix)

    y = x_tot @ true_parameters + np.random.randn(nrow) * error

    return x_tot, y
