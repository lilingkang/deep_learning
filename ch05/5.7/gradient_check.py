import numpy as np
from new_two_layer_net import *
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


# 读取数据
(x_train, t_train), (x_test, t_test) = mnist.load_data()
enc = OneHotEncoder(sparse=False)
t_train = enc.fit_transform(t_train.reshape(-1, 1))  # 标签转独热编码
x_train = x_train.reshape(-1, 784) / float(256)

network = NewTwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_backprop = network.gradient(x_batch, t_batch)
grad_numerical = network.numerical_gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
