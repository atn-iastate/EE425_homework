from homework_2.hw2 import MnistFileInfo
from homework_2.hw2 import to_numpy_dataframe
from homework_2.hw2 import remove_middle_rows
from homework_2.hw2 import remove_zero_columns
import numpy as np
import gzip


def main():
    files = {
        "test_images": "../mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "../mnist/t10k-labels-idx1-ubyte.gz",
        "train_images": "../mnist/train-images-idx3-ubyte.gz",
        "train_labels": "../mnist/train-labels-idx1-ubyte.gz"
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


if __name__ == '__main__':
    main()