import numpy as np
from homework_2.models import GDAModel, LogisticModel


def compute_accuracy(y, y_predict):

    indicator = np.where(y == y_predict, 1, 0)
    accuracy = np.sum(indicator) / y.shape[0]
    return accuracy


def remove_invariance_cols(numpy_array):
    zeros_index = []
    for i in range(numpy_array.shape[1]):
        if np.var(numpy_array[:, i]) == 0:
            zeros_index.append(i)

    return np.delete(numpy_array, zeros_index, 1), zeros_index


def normalize_all_features(x):

    mu = np.sum(x, axis=0) / x.shape[0]  # means of each column

    shifted_x = np.subtract(x, mu)  # shift location of x
    covariance = np.dot(np.transpose(shifted_x), shifted_x)
    sd_vector = np.sqrt(np.diag(covariance))  # standard deviation of each column
    sd_vector = np.where(sd_vector == 0, 1, sd_vector)  # for columns with 0 variance

    scaled_x = np.dot(shifted_x, np.diag(np.reciprocal(sd_vector.astype(float))))

    return scaled_x


def choose_pca_rank(train_x, train_y, test_x, test_y, method='GDA', step_size=1):

    v_full = np.transpose(np.linalg.svd(train_x, full_matrices=False)[2])

    accuracy = []
    for i in range(1, v_full.shape[1] + 1, step_size):
        v = v_full[:, 0:i]
        b = train_x @ v

        if method == 'GDA':
            model = GDAModel.gda_estimate(b, train_y)
        if method == 'Logistic':
            model = LogisticModel.logistic_estimate(b, train_y, max_iter=5000)

        y_hat = model.predict(test_x @ v)
        accuracy.append(compute_accuracy(test_y, y_hat))

    best_r = np.argmax(accuracy) * step_size + 1
    plot_content = (np.arange(1, v_full.shape[1] + 1, step_size), accuracy)

    return best_r, plot_content


class GDAModel:

    def __init__(self, mu0, mu1, phi, covariance):
        self.mu0 = mu0
        self.mu1 = mu1
        self.phi = phi
        self.cov = covariance
        self.no_of_feature = mu0.shape[0]

    def predict(self, x):

        y_predict = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if self.compute_exp_term(x[i, :], self.mu0) < self.compute_exp_term(x[i, :], self.mu1):
                y_predict[i] = 1

        return y_predict

    def compute_exp_term(self, x, mu):
        return -1 / 2 * np.dot(np.dot(np.transpose(x - mu), np.linalg.inv(self.cov)), (x - mu))

    @staticmethod
    def compute_variance(x, y, mu0, mu1):

        if y.shape[0] > 0:
            var = (np.sum((x[np.where(y == 0)[0]] - mu0) ** 2) + np.sum((x[np.where(y == 1)[0]] - mu1) ** 2))/y.shape[0]
            return var
        else:
            return None

    @staticmethod
    def gda_estimate(x, y):

        if y.shape[0] == 0:
            return GDAModel(None, None, None)
        else:
            estimate_phi = np.sum(y) / y.shape[0]

        if np.sum(y) < y.shape[0]:  # will only work if y is Bernoulli distributed
            estimate_mu0 = np.sum(x[np.where(y == 0)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu0 = None

        if np.sum(y) > 0:
            estimate_mu1 = np.sum(x[np.where(y == 1)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu1 = None

        # initialize covariance matrix with zeroes entries of dimension n x n
        estimate_cov = np.zeros(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1])
        for i in range(x.shape[1]):
            estimate_cov[i, i] = GDAModel.compute_variance(x[:, i], y, estimate_mu0[i], estimate_mu1[i])
        model = GDAModel(estimate_mu0, estimate_mu1, estimate_phi, estimate_cov)

        return model


class LogisticModel:

    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def logistic_estimate(x, y, max_iter):
        learn_rate = (1e-3) / y.shape[0]
        theta_hat = np.ones(x.shape[1])

        for t in range(max_iter):
            hx = (1 / (1 + np.exp(-np.dot(x, theta_hat))))
            theta_hat = theta_hat - learn_rate * (np.dot(np.transpose(x), (hx - y)))

        model = LogisticModel(theta_hat)
        return model

    def predict(self, x):

        hx = (1 / (1 + np.exp(-np.dot(x, self.parameters))))
        y_hat = np.zeros(hx.shape[0])

        for i in range(0, len(hx)):
            if hx[i] >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0
        return y_hat
