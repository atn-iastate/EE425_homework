import scipy.stats as st
import numpy as np
import math


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
        learn_rate = (1e-2) / y.shape[0]
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


def compute_accuracy(y_test, y_hat):
    accuracy = 1 - ((1 / y_test.shape[0]) * np.sum(np.abs(y_test - y_hat)))
    return accuracy

