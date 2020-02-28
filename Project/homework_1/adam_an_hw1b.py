import numpy as np
import matplotlib.pyplot as plt


def normal_equations_estimate(x, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)


def linear_regression_sim(m, n, var_error, true_w, m_test):

    if n != len(true_w):
        raise Exception("Length of parameters vector must match the number of variables")
    
    # generating data
    x = np.random.randn(m + m_test, n)  # m independent vectors of n variables
    error =  np.linalg.norm(true_w) **2 * var_error * np.random.randn(m + m_test)
    y = np.dot(x, true_w) + error  # response variable

    test_x = x[m+1:m+m_test, :]
    test_y = y[m+1:m+m_test]

    train_w = normal_equations_estimate(x[0:m, :], y[0:m])

    # compute normalized error
    n_error = (np.linalg.norm(true_w - train_w) / np.linalg.norm(true_w)) ** 2

    # compute expected value of normalized test MSE
    normalized_test_mse = (np.linalg.norm(np.dot(test_x, train_w) - test_y) / np.linalg.norm(test_y)) ** 2

    return n_error, normalized_test_mse, train_w


def main():

    np.random.seed(1)
    # setting up parameters for the simulation
    true_parameters = np.arange(100, 0, -1)
    for i in range(100):
        true_parameters[i] *= (-1)**i 

    n = 100
    m = [80, 100, 120, 200, 300, 400]
    var_error = [0.01, 0.1]

    # number of test data
    m_test = 200

    # Question 1:
    # for each error variance and m, compute normalized error (n_error) and expected MSE  (mse)
    n_error_1 = []
    mse_1 = []
    for i in m:
        sim_var_1 = linear_regression_sim(i, n, var_error[0], true_parameters, m_test)
        n_error_1.append(sim_var_1[0])
        mse_1.append(sim_var_1[1])

    n_error_2 = []
    mse_2 = []
    for i in m:
        sim_var_2 = linear_regression_sim(i, n, var_error[1], true_parameters, m_test)
        n_error_2.append(sim_var_2[0])
        mse_2.append(sim_var_2[1])

    plot, n_error_plots = plt.subplots(2, 1)

    n_error_plots[0].plot(m, n_error_1, linestyle='--', marker='o')
    n_error_plots[0].set_title("(v = 0.01)")
    n_error_plots[1].plot(m, n_error_2,  linestyle='--', marker='o')
    n_error_plots[1].set_title("(v = 0.1)")
    plt.xlabel('Number of observations')
    plt.ylabel('Normalized estimation error')

    plot, n_mse_plots = plt.subplots(2, 1)

    n_mse_plots[0].plot(m, mse_1,  linestyle='--', marker='o')
    n_mse_plots[0].set_title("(v = 0.01)")
    n_mse_plots[1].plot(m, mse_2,  linestyle='--', marker='o')
    n_mse_plots[1].set_title("(v = 0.1)")
    plt.xlabel('Number of observations')
    plt.ylabel('Mean squares error')

    # Question 2:
    m = 80
    x = np.random.randn(m + m_test, 100)
    test_mse = [[], []]


    for i in range(2):
        
        test_mse[i] = []
        y = np.dot(x, true_parameters) + np.linalg.norm(true_parameters)** 2 * var_error[i] * np.random.randn(m + m_test)
        # computes mse and estimation error for each value of n
        for n_small in range(100):

            train_w = normal_equations_estimate(x[0:m, 0:n_small + 1], y[0:m])
            test_x = x[m:m + m_test, 0:n_small + 1]
            test_y = y[m:m + m_test]

            prediction_error = np.dot(test_x, train_w) - test_y
            mse_n_small = np.linalg.norm(prediction_error)**2 / np.linalg.norm(test_y)**2

            test_mse[i].append(mse_n_small)

    plot, test_mse_plots = plt.subplots(2, 1)
    test_mse_plots[0].plot(np.arange(1, 101, 1), test_mse[0],  linestyle='--', marker='o')
    test_mse_plots[0].set_title("(v = 0.01)")
    test_mse_plots[1].plot(np.arange(1, 101, 1), test_mse[1],  linestyle='--', marker='o')
    test_mse_plots[1].set_title("(v = 0.1)")
    plt.xlabel('Number of features')
    plt.ylabel('Normalized prediction mean squares error')  

    # print result for choosing n_small

    print("Number of n_small with the lowest mean squares error for v = 0.01: %d" % (np.argmin(test_mse[0]) + 1)) 
    print("Number of n_small with the lowest mean squares error for v = 0.1: %d" % (np.argmin(test_mse[1]) + 1))

    plt.show()

if __name__ == "__main__":
    main()

