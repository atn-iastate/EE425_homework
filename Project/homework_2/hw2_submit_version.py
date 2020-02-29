import gzip
import numpy as np


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


class MnistFileInfo:

    def __init__(self, mb=0, no_of_images=0, height=0, width=0, file_type="images"):
        self.magic_number = mb
        self.no_of_images = no_of_images
        self.height = height
        self.width = width
        self.file_type = file_type

    def get_mnist_file_info(self, buffer, file_type="images"):

        self.file_type = file_type
        convert_vector = np.asarray([256**3, 256**2, 256, 1])

        if self.file_type == "images":
            info = np.frombuffer(buffer.read(16), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)
            self.height = np.dot(info[8:12], convert_vector)
            self.width = np.dot(info[12:16], convert_vector)

        if self.file_type == "labels":
            info = np.frombuffer(buffer.read(8), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)

    def get_bytes(self):
        '''Get the number of bytes containing data'''
        if self.file_type == "images":
            return self.no_of_images * self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images

    def get_dimension(self):
        '''Get the dimension of the data to be reshape in numpy'''
        if self.file_type == "images":
            return self.no_of_images, self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images


def to_numpy_dataframe(bytestream, bytestream_info):
    '''Convert the byte stream to a numpy array based on the corresponding information matrix'''

    all_bytes = np.frombuffer(bytestream.read(bytestream_info.get_bytes()), dtype=np.uint8)
    data_frame = np.asarray(all_bytes).reshape(bytestream_info.get_dimension())

    return data_frame


def remove_zero_columns(numpy_array):
    zeros_index = []
    for i in range(numpy_array.shape[1]):
        if np.sum(numpy_array[:, i]) == 0:
            zeros_index.append(i)

    return np.delete(numpy_array, zeros_index, 1), zeros_index


def remove_middle_rows(images, labels):
    remove_index = []
    for i in range(images.shape[0]):
        if labels[i] not in [0, 9]:
            remove_index.append(i)

    images = np.delete(images, remove_index, 0)
    labels = np.delete(labels, remove_index, 0)

    return images, labels


def compute_accuracy(y, y_predict):

    indicator = np.where(y == y_predict, 1, 0)
    accuracy = np.sum(indicator) / y.shape[0]
    return accuracy


def main():

    files = {
        "test_images": "./mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "./mnist/t10k-labels-idx1-ubyte.gz",
        "train_images": "./mnist/train-images-idx3-ubyte.gz",
        "train_labels": "./mnist/train-labels-idx1-ubyte.gz"
    }

    with gzip.open(files['train_images'], 'rb') as train_images, \
            gzip.open(files['train_labels'], 'rb') as train_labels, \
            gzip.open(files['test_images'], 'rb') as test_images, \
            gzip.open(files['test_labels'], 'rb') as test_labels:

        # Getting the information header of each file
        train_images_info = MnistFileInfo()
        train_images_info.get_mnist_file_info(train_images)

        train_labels_info = MnistFileInfo()
        train_labels_info.get_mnist_file_info(train_labels, file_type="labels")

        test_images_info = MnistFileInfo()
        test_images_info.get_mnist_file_info(test_images)

        test_labels_info = MnistFileInfo()
        test_labels_info.get_mnist_file_info(test_labels, file_type="labels")

        # convert the bytestream to numpy arrays
        train_images = to_numpy_dataframe(train_images, train_images_info)
        train_labels = to_numpy_dataframe(train_labels, train_labels_info)
        test_images = to_numpy_dataframe(test_images, test_images_info)
        test_labels = to_numpy_dataframe(test_labels, test_labels_info)

        # remove pictures where labels are not 0 or 9
        train_images, train_labels = remove_middle_rows(train_images, train_labels)
        test_images, test_labels = remove_middle_rows(test_images, test_labels)

        # Applying GDA Model
        # remove all zeros columns in images files
        train_images, removed_cols = remove_zero_columns(train_images)
        test_images = np.delete(test_images, removed_cols, 1)

        train_labels = np.where(train_labels == 9, 1, 0)  # since the coded models only work for levels 0, 1
        gda_model = GDAModel.gda_estimate(train_images, train_labels)  # estimate
        gda_predict_labels = np.where(gda_model.predict(test_images) == 1, 9, 0)  # predict
        gda_accuracy = compute_accuracy(test_labels, gda_predict_labels)  # compute accuracy

        # Applying the Logistic Regression Model
        logistic_model = LogisticModel.logistic_estimate(train_images, train_labels, max_iter=1000)
        lr_predict = np.where(logistic_model.predict(test_images) == 1, 9, 0)
        lr_accuracy = compute_accuracy(test_labels, lr_predict)

        # result
        print("Accuracy of the GDA Model: %.3f" % gda_accuracy)
        print("Accuracy of the Logistic Regression Model: %.3f" % lr_accuracy)


if __name__ == "__main__":
    main()

