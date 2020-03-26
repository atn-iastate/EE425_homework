import numpy as np


n = 100

evals = np.concatenate((100 * np.random.randn(round(n / 8)), np.random.randn(n - round(n / 8)))) ** 2

print(evals)