from matplotlib import pyplot as plt
import numpy as np
from save_figure import save_figure


def relu(x):
    return np.maximum(0, x)

# x = np.arange(-5, 5, 0.1)
# y = []
# for item in x:
#     y.append(relu(item))
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# save_figure(__file__)