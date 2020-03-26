import numpy as np


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
