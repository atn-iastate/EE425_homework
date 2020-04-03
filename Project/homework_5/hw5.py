from homework_5.mnist_handling import MnistFileInfo, remove_middle_rows, to_numpy_dataframe
from homework_5.ml_utils import compute_accuracy, remove_invariance_cols, normalize_all_features, choose_pca_rank, \
                                GDAModel, LogisticModel
import gzip
import numpy as np
import scipy.stats as st
from matplotlib.pyplot import plot, show, title


def main_simulation():

    np.random.seed(1)

    # parameters for the model
    phi = 0.4
    m = 40
    m_test = 20
    n = 100
    mu0 = np.random.uniform(0, 7, n)
    mu1 = np.random.uniform(0, 7, n)

    # generate training data
    y = st.bernoulli.rvs(phi, size=m)
    tmp = np.random.randn(n, n)
    evals = np.concatenate((100 * np.random.randn(round(n/8)), np.random.randn(n - round(n/8)))) ** 2
    cov = tmp @ np.diag(evals) @ np.transpose(tmp)
    x = np.zeros(m*n).reshape(m, n)
    for i in range(m):
        mu = mu0 * (1 - y[i]).item() + mu1 * y[i].item()
        x[i, :] = np.random.multivariate_normal(mu, cov)

    # generate test data
    y_test = st.bernoulli.rvs(phi, size=m_test)

    x_test = np.zeros(m_test * n).reshape(m_test, n)
    for i in range(m_test):
        mu = mu0 * (1 - y_test[i]).item() + mu1 * y_test[i].item()
        x_test[i, :] = np.random.multivariate_normal(mu, cov)

    # WITHOUT PCA
    # Estimate using GDA and compute accuracy
    gda_model = GDAModel.gda_estimate(x, y)
    y_hat = gda_model.predict(x_test)
    gda_accuracy = compute_accuracy(y_test, y_hat)

    print("GDA Model (without PCA) accuracy is: %r" % gda_accuracy)

    # Estimate using Logistic regression and compute accuracy
    logistic_model = LogisticModel.logistic_estimate(x, y, m)
    y_hat = logistic_model.predict(x_test)
    logistic_accuracy = compute_accuracy(y_test, y_hat)

    print("Logistic Regression Model (without PCA) is: %r" % logistic_accuracy)

    # APPLYING PCA:
    best_r_gda, plot1 = choose_pca_rank(x, y, x_test, y_test, method='GDA')
    best_r_log, plot2 = choose_pca_rank(x, y, x_test, y_test, method='Logistic')

    print("Best number of PC for GDA Model: %i \nAccuracy: %.2f" % (best_r_gda, np.max(plot1[1])))
    print("Best number of PC for Logistic Regression Model: %i \nAccuracy: %.2f" % (best_r_log,  np.max(plot2[1])))

    plot(best_r_gda, np.max(plot1[1]), marker='o', color="red", alpha=1.5)
    plot(plot1[0], plot1[1], linestyle='--', marker='o', alpha=0.3)
    title("GDA Model: accuracy per number of Principal Components")
    show()

    plot(best_r_log, np.max(plot2[1]),marker='o', color="red", alpha=1.5)
    plot(plot2[0], plot2[1], linestyle='--', marker='o', alpha=0.3)
    title("Logistic Regression Model: accuracy per number of Principal Components")
    show()


def main_real_data():

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

        # remove pictures where labels are not 0 or 1
        train_images, train_labels = remove_middle_rows(train_images, train_labels)
        test_images, test_labels = remove_middle_rows(test_images, test_labels)

        # applying PCA for 200 observations
        reduced_train_images = train_images[0:200, :]
        reduced_train_labels = train_labels[0:200]

        best_r_gda, plot1 = choose_pca_rank(reduced_train_images, reduced_train_labels,
                                                     test_images, test_labels,
                                                     method='GDA', step_size=10)
        best_r_log, plot2 = choose_pca_rank(reduced_train_images, reduced_train_labels,
                                                     test_images, test_labels,
                                                     method='Logistic', step_size=10)

        print("Best number of PC for GDA Model (reduced): %i \nAccuracy: %.5f" % (best_r_gda, np.max(plot1[1])))
        print("Best number of PC for Logistic Regression Model (reduced): %i \nAccuracy: %.5f"
              % (best_r_log,  np.max(plot2[1])))

        plot(best_r_gda, np.max(plot1[1]), marker='o', color="red", alpha=1.5)
        plot(plot1[0], plot1[1], linestyle='--', marker='o', alpha=0.3)
        title("GDA Model: accuracy per number of Principal Components")
        show()

        plot(best_r_log, np.max(plot2[1]), marker='o', color="red", alpha=1.5)
        plot(plot2[0], plot2[1], linestyle='--', marker='o', alpha=0.3)
        title("Logistic Regression Model: accuracy per number of Principal Components")
        show()

        # Applying PCA for the full data sets
        best_r_gda, plot1 = choose_pca_rank(train_images, train_labels,
                                            test_images, test_labels,
                                            method='GDA', step_size=10)
        best_r_log, plot2 = choose_pca_rank(train_images, train_labels,
                                            test_images, test_labels,
                                            method='Logistic', step_size=10)

        print("Best number of PC for GDA Model (full): %i \nAccuracy: %.5f" % (best_r_gda, np.max(plot1[1])))
        print("Best number of PC for Logistic Regression Model (full): %i \nAccuracy: %.5f"
               % (best_r_log,  np.max(plot2[1])))

        plot(best_r_gda, np.max(plot1[1]), marker='o', color="red", alpha=1.5)
        plot(plot1[0], plot1[1], linestyle='--', marker='o', alpha=0.3)
        title("GDA Model: accuracy per number of Principal Components")
        show()

        plot(best_r_log, np.max(plot2[1]), marker='o', color="red", alpha=1.5)
        plot(plot2[0], plot2[1], linestyle='--', marker='o', alpha=0.3)
        title("Logistic Regression Model: accuracy per number of Principal Components")
        show()


if __name__ == '__main__':

    np.seterr(over='ignore')

    main_simulation()
    main_real_data()
