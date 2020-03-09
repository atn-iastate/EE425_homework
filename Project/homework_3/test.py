import numpy as np


a = np.asarray([1,2,3,4])
b = np.asarray([2,2,3,5])

c = [1,2,3]

bool_vector = (c == 2)
out_index = a[min(np.where(bool_vector is False)[0])]

print(out_index)