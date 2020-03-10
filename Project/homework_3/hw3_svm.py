from homework_3.utils import compute_accuracy
from homework_3.MnistFileHandling import MnistFileInfo, to_numpy_dataframe, remove_middle_rows
import sklearn.svm as svm
import gzip
import time


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

        # setting up the parameters
        c_set = [0.001, 0.1, 1, 10]
        tol_set = [0.01, 0.001, 0.0001]
        kernel_set = ['rbf', 'poly']

        # estimate using the SVM library
        for kernel in kernel_set:
            for tol in tol_set:
                for c in c_set:
                    start_time = time.perf_counter()
                    model = svm.SVC(C=c, kernel=kernel, tol=tol)
                    model.fit(train_images, train_labels)

                    predict_labels = model.predict(test_images)

                    accuracy = compute_accuracy(test_labels, predict_labels)

                    print("Parameters: C = %f, tol = %f, kernel = %s" %(c, tol, kernel))
                    print("Accuracy: %.6f" % accuracy)
                    print("Execution time: %.4f" % (time.perf_counter() - start_time))


if __name__ == '__main__':
    main()