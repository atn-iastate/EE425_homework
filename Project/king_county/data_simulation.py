import numpy as np


def data_sim(true_parameters, error, bias, nrow=1000, seed=1):

    np.random.seed(seed)

    n = true_parameters.shape[0]
    x_tot = np.random.randn(nrow, n)

    y = x_tot @ true_parameters + np.random.randn(nrow) * error + bias

    return x_tot, y
