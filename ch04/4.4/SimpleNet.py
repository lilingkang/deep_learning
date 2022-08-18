import sys
sys.path.append('./ch03/3.5')
sys.path.append('./ch04/4.2')
from numerical_gradient import numerical_gradient
import numpy as np
from cross_entropy_error import cross_entropy_error
from softmax import softmax


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
        # self.W = np.array([[0.47, 0.99, 0.84], [0.85, 0.03, 0.69]])

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = SimpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p, np.argmax(p))
t = np.array([0, 0, 1])
print(net.loss(x, t))


def f(W):
    # print(W)
    # net.W = W
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
