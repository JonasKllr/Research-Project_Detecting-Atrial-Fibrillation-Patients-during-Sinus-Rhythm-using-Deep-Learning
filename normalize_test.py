import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import normalize

a = np.array([[4.2, 0, -0.1,],
         [0, 0, 0,],
         [0, 0, 0,],
         [-5, 0, 0]])

# find min and max value in every row
min_val = a.min(axis=1)
max_val = a.max(axis=1)

# add another axis to enable broadcasting
min_val = min_val[:, np.newaxis]
max_val = max_val[:, np.newaxis]

print(min_val)
print(max_val)

# normalize
a_norm = (a - min_val) / (max_val - min_val)

print(a_norm)

