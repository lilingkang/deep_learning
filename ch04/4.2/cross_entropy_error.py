import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / t.shape[0]


# t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
# y1 = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
# y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
# print(cross_entropy_error(y1, t))
# print(cross_entropy_error(y2, t))
