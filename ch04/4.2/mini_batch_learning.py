import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder


(x_train, t_train), (x_test, t_test) = mnist.load_data()
enc = OneHotEncoder(sparse=False)
one_hot_train_label = enc.fit_transform(t_train.reshape(-1, 1))  # 标签转独热编码
x_train_norm = x_train.reshape(-1, 784) / float(256)
print(x_train_norm.shape)
print(one_hot_train_label.shape)

train_size = x_train_norm.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train_norm[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape)
print(t_batch)
