import os
from matplotlib import pyplot as plt


def save_figure(file):
    plt.savefig(os.path.splitext(file)[0] + "_fig.jpg")
