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
