import numpy as np
from sklearn.kernel_ridge import KernelRidge
from king_county.data_simulation import data_sim
from king_county.utils import normal_equations_estimate, compute_test_mse, normalize, adding_polynomial_terms
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt


def main_simulation():

    # initial setup
    m = 100
    t = 80  # number of training samples

    true_parameters = np.concatenate((np.random.randn(60) * 20, np.zeros(40)))

    error = 1e-2 * np.linalg.norm(true_parameters)
    x_sim, y_sim = data_sim(true_parameters, error, bias=10000, nrow=m)

    # linear model: plot test-mse per set of features
    test_mse = []
    for i in range(true_parameters.shape[0]):
        train_w = normal_equations_estimate(x_sim[0:t, 0:i+1], y_sim[0:t])
        y_predict = x_sim[t:m, 0:i+1] @ train_w
        test_mse.append(compute_test_mse(y_predict, y_sim[t:m]))

    plt.plot(np.arange(1, true_parameters.shape[0] + 1, 1), np.log(test_mse))
    plt.xlabel("Number of features")
    plt.ylabel("Normalized Test-MSE")
    plt.show()

    # ridge regression: plot test-mse per shrinkage parameter
    test_mse_ridge = []
    x_train, mean_x_train, variance_x_train = normalize(x_sim[0:t, :])
    y_train = y_sim[0:t] - np.mean(y_sim[0:t])
    for i in np.arange(0.01, 10, 0.01):

        train_w = normal_equations_estimate(x_train[:, 0:30], y_train, alpha=i)
        y_predict = ((x_sim[t:m, 0:30] - mean_x_train[0:30]) / np.sqrt(variance_x_train[0:30])) @ train_w + \
                    np.mean(y_sim[0:t])
        test_mse_ridge.append(compute_test_mse(y_predict, y_sim[t:m]))

    plt.plot(np.arange(0.01, 10, 0.01), np.log(test_mse_ridge))
    plt.xlabel("Shrinkage parameter (lambda)")
    plt.ylabel("Log of Normalized Test-MSE")
    plt.show()

    # lasso regression: plot test-mse per shrinkage parameter
    test_mse_lasso = []
    for i in np.arange(0.1, 5, 0.05):
        model = Lasso(alpha=i, max_iter=100000, tol=0.1, random_state=1).fit(x_sim[0:t, :], y_sim[0:t])
        y_predict = model.predict(x_sim[t:m])
        test_mse_lasso.append(compute_test_mse(y_predict, y_sim[t:m]))

    plt.plot(np.arange(0.1, 5, 0.05), np.log(test_mse_lasso))
    plt.xlabel("Shrinkage parameter (lambda)")
    plt.ylabel("Log of Normalized Test-MSE")
    plt.show()

    # Ridge Regression, Lasso Regression, and vanilla regression
    x_train, mean_x_train, variance_x_train = normalize(x_sim[0:t, :])
    y_train = y_sim[0:t] - np.mean(y_sim[0:t])
    alpha = 0.5
    test_mse = []
    test_mse_ridge = []
    test_mse_lasso = []

    for i in range(true_parameters.shape[0]):  # compute test-mse for each set of features
        # linear regression
        train_w = normal_equations_estimate(x_sim[0:t, 0:i+1], y_sim[0:t])
        y_predict = x_sim[t:m, 0:i+1] @ train_w
        test_mse.append(compute_test_mse(y_predict, y_sim[t:m]))

        # ridge regression
        ridge_w = normal_equations_estimate(x_train[0:t, 0:i+1], y_train[0:t], alpha=alpha)
        y_predict_ridge = ((x_sim[t:m, 0:i+1] - mean_x_train[0:i+1]) / np.sqrt(variance_x_train[0:i+1])) @ ridge_w + \
                            np.mean(y_sim[0:t])
        test_mse_ridge.append(compute_test_mse(y_predict_ridge, y_sim[t:m]))

        # lasso regression
        model = Lasso(alpha=2.1, max_iter=100000, tol=0.1).fit(x_sim[0:t, 0:i+1], y_sim[0:t])
        y_predict_lasso = model.predict(x_sim[t:m, 0:i+1])
        test_mse_lasso.append(compute_test_mse(y_predict_lasso, y_sim[t:m]))

    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse), label='Linear Regression')
    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse_ridge), color='red', label='Ridge Linear Regression')
    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse_lasso), color='green', label='Lasso Linear Regression')

    plt.xlabel("Number of features")
    plt.ylabel("Log of Normalized Test-MSE")
    plt.legend(loc='upper center', shadow=True, fontsize='x-small')
    plt.xticks(np.arange(1, len(test_mse), 10))
    plt.show()

    # PCA

    v = np.linalg.svd(x_sim[0:t, :], full_matrices=False)[2].T

    test_mse_pca = []
    test_mse_ridge_pca = []
    test_mse_lasso_pca = []

    for i in range(true_parameters.shape[0]):
        train_w = normal_equations_estimate(x_sim[0:t, :] @ v[:, 0:i+1], y_sim[0:t])
        y_predict = x_sim[t:m, :] @ v[:, 0:i+1] @ train_w
        test_mse_pca.append(compute_test_mse(y_predict, y_sim[t:m]))

        # lasso regression
        model = Lasso(alpha=2.1, max_iter=100000, tol=0.1).fit(x_sim[0:t, :] @ v[:, 0:i+1], y_sim[0:t])
        y_predict_lasso = model.predict(x_sim[t:m, :] @ v[:, 0:i+1])
        test_mse_lasso.append(compute_test_mse(y_predict_lasso, y_sim[t:m]))

        # ridge regression
        model = Ridge(alpha=0.5, normalize=True, max_iter=100000, tol=0.1).fit(x_sim[0:t, :] @ v[:, 0:i+1], y_sim[0:t])
        y_predict_ridge = model.predict(x_sim[t:m, :] @ v[:, 0:i+1])
        test_mse_ridge_pca.append(compute_test_mse(y_predict_ridge, y_sim[t:m]))

    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse_ridge), color='blue', label='Ridge Linear Regression')
    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse_ridge_pca), color='orange',
             label='Ridge Linear Regression - PCA')
    plt.xlabel("Number of features / principal components")
    plt.ylabel("Log of Normalized Test-MSE")
    plt.legend(loc='upper center', shadow=True, fontsize='x-small')
    plt.xticks(np.arange(1, len(test_mse), 10))
    plt.show()


def main_real_data():
    data = np.loadtxt('Data Updated with Removed Variables.csv', delimiter=',', skiprows=1)

    y = data[0:700, 14]
    x = data[0:700, 0:14]
    t = 600
    m = 700

    # adding new features up to 3rd order
    x = adding_polynomial_terms(x, degree=[2, 3])

    x_train, mean_x_train, variance_x_train = normalize(x[0:t, :])
    y_train = y[0:t] - np.mean(y[0:t])
    alpha = 1

    test_mse = []
    test_mse_ridge = []

    for i in range(x.shape[1]):

        train_w = normal_equations_estimate(x[0:t, 0:i+1], y[0:t])
        y_predict = x[t:m, 0:i+1] @ train_w
        test_mse.append(compute_test_mse(y_predict, y[t:m]))

        ridge_w = normal_equations_estimate(x_train[0:t, 0:i+1], y_train[0:t], alpha=alpha)
        y_predict_ridge = ((x[t:m, 0:i+1] - mean_x_train[0:i+1]) / np.sqrt(variance_x_train[0:i+1])) @ ridge_w + \
                            np.mean(y[0:t])
        test_mse_ridge.append(compute_test_mse(y_predict_ridge, y[t:m]))

    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse), label='Linear Regression')
    plt.plot(np.arange(1, len(test_mse) + 1, 1), np.log(test_mse_ridge), color='red', label='Ridge Linear Regression')

    plt.xlabel("Number of features")
    plt.ylabel("Log of Normalized Test-MSE")
    plt.legend(loc='upper center', shadow=True, fontsize='x-small')
    plt.xticks(np.arange(1, len(test_mse) + 1, 50))
    plt.show()

    print(np.argmin(test_mse_ridge))
    print(np.argmin(test_mse))


if __name__ == '__main__':
    main_simulation()
    # main_real_data()
