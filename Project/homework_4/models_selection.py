import numpy as np
import matplotlib.pyplot as plt


def compute_test_mse(y_hat, y_test):

    if y_hat.shape[0] != y_test.shape[0]:
        raise Exception("y_hat and y_test must have the same dimension")

    return np.sum((y_test - y_hat) ** 2) / y_test.shape[0]


def choose_first_r_columns(x, y, x_test, y_test):

    """
    return an optimal number of features (for multivariate linear regression) based on the test MSE and a plot of
    log(test MSE) against the number of features

    :param x: training independent observations
    :param y: training dependent observations
    :param x_test: testing independent observations
    :param y_test: testing dependent observations
    :return: r is the number of optimal columns of X to be included in the model (0:r)
    """
    if x.shape[0] != y.shape[0]:
        raise Exception("x and y must have the same number of rows")
    elif x_test.shape[0] != y_test.shape[0]:
        raise Exception("x_test and y_test must have the same number of rows")

    test_mse = []

    # adding the bias term to the design matrix
    x = np.concatenate((np.ones(x.shape[0]).reshape(x.shape[0], 1), x), axis=1)
    x_test = np.concatenate((np.ones(x_test.shape[0]).reshape(x_test.shape[0], 1), x_test), axis=1)

    for i in range(x.shape[1]):
        estimate_parameters = np.linalg.inv(np.transpose(x[:, 0:i]) @ x[:, 0:i]) @ np.transpose(x[:, 0:i]) @ y
        y_hat = x_test[:, 0:i] @ estimate_parameters
        test_mse.append(compute_test_mse(y_hat, y_test))

    r = np.argmin(test_mse) + 1
    plot_content = (np.arange(0, len(test_mse), 1), np.log(test_mse))

    return r, plot_content


def choose_rank_r(x, y, x_test, y_test):

    """
    return an optimal rank of the approximation matrix (to be used for multivariate linear regression)
    based on the test MSE and a plot of test MSE against the number of features

    :param x: training independent observations
    :param y: training dependent observations
    :param x_test: testing independent observations
    :param y_test: testing dependent observations
    :return: r is the optimal rank of the approximation matrix
    """
    if x.shape[0] != y.shape[0]:
        raise Exception("x and y must have the same dimension")
    elif x_test.shape[0] != y_test.shape[0]:
        raise Exception("x_test and y_test must have the same dimension")

    u_full, s_full, vh_full = np.linalg.svd(x)

    test_mse = []

    for i in range(1, x.shape[1]):
        v = np.transpose(vh_full)[:, 0:i]
        B = x @ v

        B = np.concatenate((np.ones(B.shape[0]).reshape(B.shape[0], 1), B), axis=1)  # adding bias term

        estimate_parameters = np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B) @ y

        y_hat = x_test @ v @ estimate_parameters[0: i] + estimate_parameters[i]

        test_mse.append(compute_test_mse(y_hat, y_test))

    r = np.argmin(test_mse) + 1
    plot_content = (np.arange(0, len(test_mse), 1), np.log(test_mse))

    return r, plot_content
