from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,AveragePooling2D,Reshape
from keras.models import Model
from keras import backend as K

def preprocess():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train),28,28, 1))
    x_test = np.reshape(x_test, (len(x_test),28,28, 1))
    y_train = to_categorical(y_train,10)
    y_test = to_categorical(y_test,10)

    return (x_train,y_train,x_test,y_test)

