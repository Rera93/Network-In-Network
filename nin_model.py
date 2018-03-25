from keras.datasets import mnist
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D,AveragePooling2D,Reshape
from keras.models import Model
from keras import backend as K