import numpy as np
from matplotlib.pyplot import show, subplots, ylabel
from homework_4.data_simulation import data_sim
from homework_4.models_selection import choose_first_r_columns, choose_rank_r
from pandas import read_csv


def remove_zero_columns(numpy_array):
    zeros_index = []
    for i in range(numpy_array.shape[1]):
        if np.sum(numpy_array[:, i]) == 0:
            zeros_index.append(i)

    return np.delete(numpy_array, zeros_index, 1), zeros_index


def main():

    # initial setup
    true_parameters = np.arange(100, 50, -0.5)
    mean_x = np.arange(5, 15, 0.1)
    error = 1e-2 * np.linalg.norm(true_parameters)

    x_tot, y = data_sim(true_parameters, mean_x, error, nrow=80)

    # model_selection
    r1, plot1 = choose_first_r_columns(x_tot[0:50, :], y[0:50], x_test=x_tot[50:80], y_test=y[50:80])
    r2, plot2 = choose_rank_r(x_tot[0:50, :], y[0:50], x_test=x_tot[50:80], y_test=y[50:80])

    # showing results for simulation
    plot, mse_plots = subplots(2, 1)

    mse_plots[0].plot(plot1[0], plot1[1], linestyle='--', marker='o')
    mse_plots[0].set_title("MSE per number of first columns")
    mse_plots[1].plot(plot2[0], plot2[1], linestyle='--', marker='o')
    mse_plots[1].set_title("MSE per number of Principal Components")
    ylabel('Log( MSE )')

    print("Simulation:")
    print("The optimal number of features to be included is %i first features" % r1)
    print("The optimal number of principal components to be included is %i components" % r2)
    print("")

    # blogData
    data = {
        "train_data_1": "blogData_test-2012.02.01.00_00.csv",
        "train_data_2":  "blogData_test-2012.02.02.00_00.csv",
        "test_data": "blogData_test-2012.02.03.00_00.csv"
    }

    url = "https://raw.githubusercontent.com/atn-iastate/EE425_homework/master/Project/homework_4/data/"

    train_data_1 = read_csv("{}{}".format(url, data['train_data_1']), delimiter=",").to_numpy()
    train_data_2 = read_csv("{}{}".format(url, data['train_data_2']), delimiter=",").to_numpy()
    train_data = np.concatenate((train_data_1, train_data_2), axis=0)

    test_data = read_csv("{}{}".format(url, data['test_data']), delimiter=",").to_numpy()

    train_data, removed_cols = remove_zero_columns(train_data)
    test_data = np.delete(test_data, removed_cols, 1)

    # print(train_data.shape)
    # print(test_data.shape)
    #
    # b = np.linalg.inv(np.transpose(train_data[:, 0:121]) @ train_data[:, 0:121])
    # print(b)

    print("blogData:")

    n = train_data.shape[1] - 1
    # model_selection
    r3, plot3 = choose_first_r_columns(train_data[:, 0:n], train_data[:, n], test_data[:, 0:n], test_data[:, n])
    r4, plot4 = choose_rank_r(train_data[:, 0:n], train_data[:, n], test_data[:, 0:n], test_data[:, n])

    # showing results for simulation
    plot, another_mse_plots = subplots(2, 1)

    another_mse_plots[0].plot(plot3[0], plot3[1], linestyle='--', marker='o')
    another_mse_plots[0].set_title("MSE per number of first columns")
    another_mse_plots[1].plot(plot4[0], plot4[1], linestyle='--', marker='o')
    another_mse_plots[1].set_title("MSE per number of Principal Components")
    ylabel('Log( MSE )')

    print("blogData:")
    print("The optimal number of features to be included is %i first features" % r3)
    print("The optimal number of principal components to be included is %i components" % r4)
    print("")

    show()


if __name__ == '__main__':
    main()


