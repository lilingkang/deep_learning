import os
from matplotlib import pyplot as plt
import numpy as np
from save_figure import save_figure


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# x = np.arange(-5, 5, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# save_figure(__file__)
# plt.savefig(os.path.splitext(__file__)[0] + "_fig.jpg")
