from homework_2.models import GDAModel, LogisticModel
import gzip
import numpy as np

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
        gda_train_images, removed_cols = remove_zero_columns(train_images)
        gda_test_images = np.delete(test_images, removed_cols, 1)

        train_labels = np.where(train_labels == 9, 1, 0)
        gda_model = GDAModel.gda_estimate(gda_train_images, train_labels)  # estimate
        gda_predict_labels = np.where(gda_model.predict(gda_test_images) == 1, 9, 0)  # predict
        gda_accuracy = compute_accuracy(test_labels, gda_predict_labels)  # compute accuracy

        # Applying the Logistic Regression Model
        logistic_model = LogisticModel.logistic_estimate(train_images, train_labels, max_iter=1000)
        lr_predict = np.where(logistic_model.predict(test_images) == 1, 9, 0)
        lr_accuracy = compute_accuracy(test_labels, lr_predict)

        # testing
        print("Accuracy of the GDA Model: %.3f" % gda_accuracy)
        print("Accuracy of the Logistic Regression Model: %.3f" % lr_accuracy)


if __name__ == "__main__":
    main()

