import numpy as np

def find_optimal_b(y, w, x):
    #minimum and maximum values of w^Tx^(i)
    w_T_x_i_max = 0
    w_T_x_i_min = 100

    #indices of y that maximize and minimize w^Tx^(i)
    y_maximizing_indice = 0
    y_minimizing_indice = 0
    w_T = np.linalg.transpose(w)

    #iterate through x_is to find i values maximizing & minimizing w^Tx^(i)
    #conditioned on the value of y[i] being -1 or 1 respectively
    for i in range(0, sizeof(y)):
        x_i = x[:,i]
        w_T_x_i = np.dot(w_T, x_i)

        #update max value and indice if a new max is encountered
        if((w_T_x_i > w_T_x_i_max) and (y[i] == -1)):
            w_T_x_i_max = w_T_x_i
            y_maximizing_indice = i

        #update min value and indice if a new min is encountered
        if((w_T_x_i < w_T_x_i_min) and (y[i] == 1)):
            w_T_x_i_min = w_T_x_i
            y_minimizing_indice = i

    #evaluate b* (optimal b) - eq. 13 from Andrew Ng's notes
    b = (-0.5 * y_minimizing_indice * y_maximizing_indice *
        np.dot(w_T, x[:,y_maximizing_indice]) *
        np.dot(w_T, x[:,y_minimizing_indice]))

    return b