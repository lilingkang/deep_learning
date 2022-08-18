import numpy as np
from two_layer_net import TwoLayerNet
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import sys
sys.path.append("./ch03/3.2")
from save_figure import save_figure


(x_train, t_train), (x_test, t_test) = mnist.load_data()
enc = OneHotEncoder(sparse=False)
one_hot_train_label = enc.fit_transform(t_train.reshape(-1, 1))  # 标签转独热编码
x_train_norm = x_train.reshape(-1, 784) / float(256)

train_loss_list = []

# 超参数
iters_num = 10
train_size = x_train_norm.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    
    x_batch = x_train_norm[batch_mask]
    t_batch = one_hot_train_label[batch_mask]
    
    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print("iter " + str(i) + ": loss=" + str(loss))
    
plt.plot(train_loss_list)
save_figure(__file__)
