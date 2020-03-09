import numpy as np

index = (1, 3)
p = index[0]
q = index[1]
a = np.arange(0, 15).reshape(5, 3)

print(np.delete(a, list(index), axis=0))
print(a)