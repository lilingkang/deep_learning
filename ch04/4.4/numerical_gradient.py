import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    for idx in range(x.shape[0]):
        if len(x.shape) > 1:
            grad[idx] = numerical_gradient(f, x[idx])
        else:
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)

            grad[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmp_val  # 还原值

    return grad


# def f2(x):
#     return x[0]**2+x[1]**2

# print(numerical_gradient(f2,np.array([3.0,4.0])))