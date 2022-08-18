import numpy as np
import matplotlib.pylab as plt
import os
from save_figure import save_figure


def step_function(x):
    return np.array(x > 0, dtype=np.int)


# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # 指定y轴的范围
# save_figure(__file__)
# plt.savefig(os.path.splitext(__file__)[0] + "_fig.jpg")
# print(os.path.splitext(__file__)[0])
