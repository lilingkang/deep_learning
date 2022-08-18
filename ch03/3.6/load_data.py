import numpy as np
from keras.datasets import mnist
from PIL import Image
import os


(x_train, t_train), (x_test, t_test) = mnist.load_data()
# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)

pil_img = Image.fromarray(np.uint8(img))
pil_img.save(os.path.split(__file__)[0] + "/first_img.jpg")
