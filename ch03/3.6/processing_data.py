import sys
sys.path.append('./ch03/3.2')
sys.path.append('./ch03/3.5')
from softmax import softmax
from sigmoid import sigmoid
import numpy as np
from keras.datasets import mnist


# load data
def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    x_test_norm = x_test.reshape(10000, 784) / float(256)
    return x_test_norm, t_test


# init network(输入层：784，隐藏层1：50，隐藏层2：100，输出层：10)
def init_network():
    network = {}
    network['W1'] = np.random.random(784 * 50).reshape(784, 50)
    network['W2'] = np.random.random(50 * 100).reshape(50, 100)
    network['W3'] = np.random.random(100 * 10).reshape(100, 10)
    network['b1'] = np.random.random(50)
    network['b2'] = np.random.random(100)
    network['b3'] = np.random.random(10)

    return network


# predict
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100  # 批数量
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 获取概率最高的元素索引
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Process:" + str(i) + "/" + str(len(x)) + " " + str(accuracy_cnt))

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
