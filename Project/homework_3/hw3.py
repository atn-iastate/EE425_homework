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


def update_alpha(x, y, alpha, c, index):

    p = index[0]
    q = index[1]

    h = alpha[p] * y[p] + alpha[q] * y[q]

    # compute the constant term and coefficient of alpha[p] in the derivative of W(alpha[p])

    alpha_y = np.asarray([y[i] * alpha[i] for i in range(len(alpha))])
    numerator = y[p] * (np.dot(x[p, :] - x[q, :], np.dot(np.transpose(x), alpha_y)) + y[p] - y[q])
    denominator = np.dot(np.transpose(x[p]), x[p]) - 2 * y[p] * y[q] + np.dot(np.transpose(x[q]), x[q])

    alpha_p_new = alpha[p] + numerator / denominator

    lower_bound, upper_bound = get_bounds(p, q, y, alpha, c)

    if alpha_p_new > upper_bound:
        alpha_p = upper_bound
    elif alpha_p_new < lower_bound:
        alpha_p = lower_bound
    else:
        alpha_p = alpha_p_new

    alpha_q = h * y[q] - alpha[p] * y[p] * y[q]

    return alpha_p, alpha_q


def get_bounds(p, q, y, alpha, c):
    '''Return the lower and upper bound for alpha_p'''

    if c < 0:
        raise ValueError("c must be a non-negative number")

    if y[p] == y[q]:
        lower_bound = max(alpha[p] + alpha[q] - c, 0)
        upper_bound = min(c, alpha[p] + alpha[q])
    else:
        lower_bound = max(alpha[p] - alpha[q], 0)
        upper_bound = min(alpha[p] - alpha[q] + c, c)

    return lower_bound, upper_bound


def find_second_index(x, y, w, b, first_index, unbound_index):

    '''Find the index that
        - belongs to one of the unbound example
        - maximize the step size the objective function'''
    E0 = np.dot(x[first_index, :], w) + b - y[first_index]

    errors = [np.dot(x[i, :], w) + b - y[i] for i in unbound_index]
    step_size = abs(errors - E0)
    second_index = np.random.choice(unbound_index[np.where(step_size == max(step_size))[0]])

    return second_index


def get_unbound_index(x, y, w, alpha, b, c, tol):

    '''Return the indexes of observations where the KKT conditions do not hold'''

    boolean_vector = np.ones_like(alpha)
    for i in range(len(alpha)):
        if alpha[i] == 0:
            boolean_vector[i] = ((np.dot(x[i, :], w) + b) * y[i] >= (1 - tol))
        elif alpha[i] == c:
            boolean_vector[i] = ((np.dot(x[i, :], w) + b) * y[i] <= (1 + tol))
        else:
            boolean_vector[i] = ((np.dot(x[i, :], w) + b) * y[i] <= (1 + tol) | ((np.dot(x[i, :], w) + b) * y[i] >= (1 - tol)))

    return np.where(boolean_vector == 0)[0]


def compute_w(alpha, x, y):
    alpha_y = np.asarray([y[i] * alpha[i] for i in range(len(alpha))])
    return np.dot(np.transpose(x), alpha_y)


def compute_threshold(c, x, y, alpha, old_alpha, index):
    p = index[0]
    q = index[1]
    alpha_y = np.asarray([y[i] * alpha[i] for i in range(len(alpha))])
    bp = np.dot(x[p, :], np.dot(np.transpose(x), alpha_y)) - \
         y[p] * (alpha[p] - old_alpha[0]) * np.dot(x[p, :], np.transpose(x[p, :])) - \
         y[q] * (alpha[q] - old_alpha[1]) * np.dot(x[p, :], np.transpose(x[q, :]))

    bq = np.dot(x[q, :], np.dot(np.transpose(x), alpha_y)) - \
        y[p] * (alpha[p] - old_alpha[0]) * np.dot(x[p, :], np.transpose(x[q, :])) - \
        y[q] * (alpha[q] - old_alpha[1]) * np.dot(x[q, :], np.transpose(x[q, :]))

    if 0 < alpha[p] < c:
        b = bp
    elif 0 < alpha[q] < c:
        b = bq
    else:
        b = (bp + bq) / 2

    return b


def compute_accuracy(y, y_predict):

    indicator = np.where(y == y_predict, 1, 0)
    accuracy = np.sum(indicator) / y.shape[0]
    return accuracy


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
        max_iter = 10000
        C = 0.001
        tol = 0.01
        alpha = np.zeros_like(train_labels, dtype=float)
        b = 0

        for i in range(max_iter):
            # compute w and b
            w = compute_w(alpha, train_images, train_labels)

            unbound_index = get_unbound_index(train_images, train_labels, w, alpha, b, C, tol)
            if len(unbound_index) == 0:
                break
            else:
                p = np.random.choice(unbound_index)
                q = find_second_index(train_images, train_labels, w, b, p, unbound_index)
                old_alpha = [alpha[i] for i in [p, q]]
                alpha[p], alpha[q] = update_alpha(train_images, train_labels, alpha, C, (p, q))
                alpha[p], alpha[q] = update_alpha(train_images, train_labels, alpha, C, (p, q))
                b = compute_threshold(C, train_images, train_labels, alpha, old_alpha, (p, q))

        # predict test data
        z = np.dot(test_images, w) + b * np.ones_like(test_labels)
        y_predict = np.where(z >= 0, 0, 9)
        accuracy = compute_accuracy(test_labels, y_predict)
        
        print("Accuracy of the SVM: %2f" % accuracy)


if __name__ == '__main__':
    main()