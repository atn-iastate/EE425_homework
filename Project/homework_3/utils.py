import numpy as np


def update_alpha(x, y, alpha, c, index=(0, 1)):
    if index[0] == index[1]:
        raise ValueError("Input must contains 2 different indexes")

    p = index[0]
    q = index[1]

    h = alpha[p] * y[p] + alpha[q] * y[q]

    # x_reduced, y_reduced, alpha_reduced are x, y, and alpha without the input rows p, q
    x_reduced = np.delete(x, [p, q], axis=0)
    y_reduced = np.delete(y, [p, q], axis=0)
    alpha_reduced = np.delete(alpha, [p, q], axis=0)
    D = np.diag(alpha_reduced)

    # compute the constant term and coefficient of alpha[p] in the derivative of W(alpha[p])
    constant_term = 1 - y[p] * y[q] \
                    - np.dot((y[p] * x[p] + y[q] * x[q]), np.dot(np.transpose(x_reduced), np.dot(D, y_reduced))) \
                    - h * y[q] + h * y[p] * np.dot(np.transpose(x[q]), x[q])

    coefficient = np.dot(np.transpose(x[p]), x[p]) - 2 * y[p] * y[q] + np.dot(np.transpose(x[q]), x[q])

    alpha_p_new = constant_term / coefficient

    lower_bound, upper_bound = get_bounds(y[p], y[q], h, c)

    if alpha_p_new > upper_bound:
        alpha[p] = upper_bound
    elif alpha_p_new < lower_bound:
        alpha[p] = lower_bound
    else:
        alpha[p] = alpha_p_new

    alpha[q] = h * y[q] - alpha[p] * y[p] * y[q]

    return alpha


def get_bounds(y_p, y_q, h, c):
    '''Return the lower and upper bound for alpha_p'''

    if c < 0:
        raise ValueError("c must be a non-negative number")
    elif abs(y_q * y_q) != 1:
        raise ValueError("Only accept values of -1 or 1 for labels")

    if y_p == y_q:
        lower_bound = max(h-c, 0)
        upper_bound = min(c, h)
    else:
        lower_bound = max(h, 0)
        upper_bound = min(h + c, c)

    return lower_bound, upper_bound


def check_kkt_convergence(x, y, alpha, w, b, c, tol = 0.01):

    '''Check if the KKT conditions are satisfied, return a boolean value with an index of the first violation or -1
    (if there is no violation)'''
    result = 0

    zero_index = np.where(alpha == 0)[0]
    c_index = np.where(alpha == c)[0]
    middle_index = np.where(0 < alpha < c)[0]

    # for alpha == 0
    bool_vector = np.dot(np.diag(y[zero_index]), np.dot(x[zero_index, :], w) + np.ones(len(zero_index)) * b) >= (1 - tol)
    if all(bool_vector is True):
        result += 1
    else:
        out_index = zero_index[min(np.where(bool_vector is False)[0])]

    # for alpha == c
    bool_vector = np.dot(np.diag(y[c_index]), np.dot(x[c_index, :], w) + np.ones(len(c_index)) * b) <= (1 + tol)
    if all(bool_vector is True):
        result += 1
    else:
        out_index = c_index[min(np.where(bool_vector is False)[0])]

    # for 0 < alpha < c
    bool_vector = (1-tol) <= np.dot(np.diag(y[middle_index]), np.dot(x[middle_index, :], w) + np.ones(len(middle_index)) * b) <= (1 + tol)
    if all(bool_vector is True):
        result += 1
    else:
        out_index = middle_index[min(np.where(bool_vector is False)[0])]

    if result == 3:
        return True, -1
    else:
        return False, out_index
