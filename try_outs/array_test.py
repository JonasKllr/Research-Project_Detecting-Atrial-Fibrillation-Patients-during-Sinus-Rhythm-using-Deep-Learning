import numpy as np

a = np.array([[1, 2],
             [3, 4],
             [5, 6],
             [7, 8]])

print(np.shape(a))

b = np.array([1, 2])

c = a[b]
print(c)