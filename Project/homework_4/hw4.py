import numpy as np
from matplotlib.pyplot import show, subplots, ylabel
from homework_4.data_simulation import data_sim
from homework_4.models_selection import choose_first_r_columns, choose_rank_r
from pandas import read_csv


def main():

    # initial setup
    true_parameters = np.arange(100, 50, -0.5)
    mean_x = np.arange(5, 15, 0.1)
    error = 1e-2 * np.linalg.norm(true_parameters)

    x_tot, y = data_sim(true_parameters, mean_x, error, nrow=80)

    # model_selection
    r1, plot1 = choose_first_r_columns(x_tot[0:50, :], y[0:50], x_test=x_tot[50:80], y_test=y[50:80])
    r2, plot2 = choose_rank_r(x_tot[0:50, :], y[0:50], x_test=x_tot[50:80], y_test=y[50:80])

    plot, mse_plots = subplots(2, 1)

    mse_plots[0].plot(plot1[0], plot1[1], linestyle='--', marker='o')
    mse_plots[0].set_title("MSE per number of first columns")
    mse_plots[1].plot(plot2[0], plot2[1], linestyle='--', marker='o')
    mse_plots[1].set_title("MSE per number of Principal Components")
    ylabel('Log( MSE )')

    print("The optimal number of features to be included is %i first features" % r1)
    print("The optimal number of principal components to be included is %i components" % r2)

    show()

    # real data
    


if __name__ == '__main__':
    main()
