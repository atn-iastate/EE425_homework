from homework_2.hw2 import MnistFileInfo
from homework_2.hw2 import to_numpy_dataframe
from homework_2.hw2 import remove_middle_rows
from homework_3.package.utils import update_alpha, compute_w, compute_intercept, find_second_index, get_unbound_index, compute_accuracy
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

        # change train labels to 1 and -1
        train_labels = np.where(train_labels == 0, 1, -1)

        # start training
        max_iter = 100
        C = 0.1
        tol = 0.001
        alpha = np.zeros_like(train_labels)

        # testing
        # w = compute_w(alpha, train_images, train_labels)
        # b = compute_intercept(w, train_images, train_labels)
        #
        # unbound_index = get_unbound_index(train_images, train_labels, w, alpha, b, C, tol)
        # index = (100, 200)
        # x_reduced = np.delete(train_images, list(index), axis=0)
        # p = np.random.choice(unbound_index)
        # q = find_second_index(train_images, train_labels, w, b, p, unbound_index)
        #
        # print(x_reduced.shape)
        # print(q)

        for i in range(max_iter):
            # compute w and b
            w = compute_w(alpha, train_images, train_labels)
            b = compute_intercept(w, train_images, train_labels)

            unbound_index = get_unbound_index(train_images, train_labels, w, alpha, b, C, tol)
            if len(unbound_index) == 0:
                break
            else:
                p = np.random.choice(unbound_index)
                q = find_second_index(train_images, train_labels, w, b, p, unbound_index)

                alpha = update_alpha(train_images, train_labels, alpha, C, (p, q))

        # predict test data
        z = np.dot(test_images, w) + b * np.ones_like(test_labels)
        y_predict = np.where(z >= 0, 0, 9)
        accuracy = compute_accuracy(test_labels, y_predict)
        
        print("Accuracy of the SVM: %2f" % accuracy)
        print(w[0:10])
        print(unbound_index)


if __name__ == '__main__':
    main()