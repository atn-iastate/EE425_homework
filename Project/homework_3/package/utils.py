import numpy as np


def update_alpha(x, y, alpha, c, index):

    p = index[0]
    q = index[1]

    h = alpha[p] * y[p] + alpha[q] * y[q]

    # x_reduced, y_reduced, alpha_reduced are x, y, and alpha without the input rows p, q
    x_reduced = np.delete(x, list(index), axis=0)
    y_reduced = np.delete(y, list(index), axis=0)
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


def check_kkt_convergence(unbound_index):

    '''Check if the KKT conditions are satisfied, return a boolean value with an index of the first violation or -1
    (if there is no violation)'''

    out_index = np.choice(unbound_index)

    if len(unbound_index) == 0:
        return True, None
    else:
        return False, out_index


def find_second_index(x, y, w, b, first_index, unbound_index):

    '''Find the index that
        - belongs to one of the unbound example
        - maximize the step size the objective function'''

    errors = [np.dot(x[i, :], w) + b - y[i] for i in unbound_index]
    step_size = abs(errors - errors[first_index])
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
    return np.dot(np.transpose(x), np.dot(np.diag(alpha), y))


def compute_intercept(w, x, y):
    one_index = np.where(y == 1)[0]
    minus_one_index = np.where(y == -1)[0]

    return (max(np.dot(x[minus_one_index, :], w)) + min(np.dot(x[one_index, :], w))) / 2


def compute_accuracy(y, y_predict):

    indicator = np.where(y == y_predict, 1, 0)
    accuracy = np.sum(indicator) / y.shape[0]
    return accuracy
