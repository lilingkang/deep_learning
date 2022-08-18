import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.tile(np.sum(exp_a, axis=1).reshape(a.shape[0], 1), a.shape[1])
    y = exp_a / sum_exp_a
    
    return y
