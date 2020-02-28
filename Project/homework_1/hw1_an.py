import numpy as np
import time

# Question 1:


def lreg_sim(m, n, verror, true_w):

    if n != len(true_w):
        raise Exception("Length of parameters vector must match the number of variables")

    # generating data
    X = np.random.randn(m, n)  # m independent vectors of n variables
    error = verror * np.random.randn(m)
    y = np.dot(X, true_w) + error  # response variable

    # Pseudo-Inverse
    start_time = time.perf_counter()
    train_w_pi = np.dot(np.linalg.pinv(X), y)
    n_error = np.linalg.norm(true_w - train_w_pi) / np.linalg.norm(true_w)

    print("--- Pseudo Inverse ---")
    print("Normalized error: %.10f" % n_error)
    print("Execution time: %.10f" % (time.perf_counter() - start_time))
    print(" ")

    # Normal Equations
    start_time = time.perf_counter()
    train_w_norm = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
    n_error = np.linalg.norm(true_w - train_w_norm) / np.linalg.norm(true_w)

    print("--- Solving Normal Equations ---")
    print("Normalized error: %.10f" % n_error)
    print("Execution time: %.10f" % (time.perf_counter() - start_time))
    print(" ")

    # Gradient Descent
    start_time = time.perf_counter()
    mu = 0.001
    max_iter = 1000
    train_w_gd = np.zeros_like(true_w)

    for t in range(max_iter):
        train_w_gd = train_w_gd - mu * (
                    np.dot(np.dot(np.transpose(X), X), train_w_gd) - np.dot(np.transpose(X), y - error));

    n_error = np.linalg.norm(true_w - train_w_gd) / np.linalg.norm(true_w)

    print("--- Gradient Descent ---")
    print("Normalized error: %.10f" % n_error)
    print("Execution time: %.10f" % (time.perf_counter() - start_time))
    print(" ")


# a) m = 30, n = 5, var_error = 0, true_w = [1,4,2,10,23]
lreg_sim(30, 5, 0, [1, 4, 2, 10, 23])
# b) m = 30, n = 5, var_error = e^-6, true_w = [1,4,2,10,23]
lreg_sim(30, 5, 1e-6, [1, 4, 2, 10, 23])
# c) m = 100, n = 5, var_error = e^-6, true_w = [1,4,2,10,23]
lreg_sim(100, 5, 1e-6, [1, 4, 2, 10, 23])
# d) m = 1000, n = 5, var_error = e^-6, true_w = [1,4,2,10,23]
lreg_sim(1000, 5, 1e-6, [1, 4, 2, 10, 23])
# e) m = 1000, n = 5, var_error = e^-4, true_w = [1,4,2,10,23]
lreg_sim(100, 5, 1e-4, [1, 4, 2, 10, 23])


# Question 2: Working with real data
raw = np.genfromtxt(r"./airfoil_self_noise.dat",
                    dtype=None,
                    delimiter="\t")
airfoil = []
for t in raw:
    airfoil.append(list(t))

airfoil = np.array(airfoil)
x = airfoil[:, 0:5]
intercept = np.ones(len(airfoil)).reshape(len(airfoil), 1)
x = np.hstack((x, intercept))
y = airfoil[:, 5].reshape(len(airfoil), )

# training

train_w = [0, 0, 0, 0, 0, 0]


def gd_lreg(x, y, initial_w, max_iter, learning_rate):

    train_w = initial_w
    for i in range(max_iter):
        gradient = np.dot(np.dot(np.transpose(x), x), train_w) - np.dot(np.transpose(x), y)
        train_w = train_w - learning_rate * gradient

    error = np.linalg.norm(y - np.dot(x, train_w))**2 / len(airfoil)

    print("Estimates of parameters: ", train_w)
    print("Mean square error:", error)


gd_lreg(x, y, train_w, max_iter=100000, learning_rate=1e-11)

# Extra credit: standardizing the features

standard_x = []
for t in range(5):
    col = x[:, t]
    col = col - np.average(col) * np.ones_like(col)
    col = col / np.std(col, ddof=1)
    standard_x.append(col)

standard_x = np.asarray(standard_x)
standard_x = np.transpose(standard_x)

# repeat the gradient descent experiment

train_w = np.zeros(6)
standard_x = np.hstack((standard_x, intercept))
max_iter = 1000
learning_rate = 0.0001

gd_lreg(standard_x, y, train_w, max_iter, learning_rate)

# Extra credit: applying Stochastic Gradient Descent


def stochastic_gd_lreg(x, y, initial_w, batch_size, learning_rate):

    train_w = initial_w
    i = 0
    while i < len(x):
        last_index = min(i + batch_size, len(x))
        batch_x = x[i:last_index, :]
        batch_y = y[i:last_index]
        gradient = np.dot(np.dot(np.transpose(batch_x), batch_x), train_w) - np.dot(np.transpose(batch_x), batch_y)
        train_w = train_w - learning_rate * gradient
        i = i + batch_size
    error = np.linalg.norm(y - np.dot(x, train_w))**2 / len(airfoil)

    print("Estimates of parameters: ", train_w)
    print("Mean square error:", error)


train_w = np.zeros(6)

stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=1, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=10, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=50, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=200, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=400, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=600, learning_rate=0.001)
stochastic_gd_lreg(standard_x, y, initial_w=train_w, batch_size=800, learning_rate=0.001)