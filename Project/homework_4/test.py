import numpy as np

x = np.arange(0, 9, 1).reshape(3, 3)

x = np.concatenate((np.ones(x.shape[0]).reshape(x.shape[0], 1), x), axis=1)

print(x.shape)
print(x)