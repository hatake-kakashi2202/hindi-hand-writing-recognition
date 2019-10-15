import numpy as np
from keras import layers
from keras.layers import Input,Dense,Activation,ZeroPadding2D,BatchNormalization,Flatten,Conv2D
from keras.layers import AveragePooling2D,MaxPooling2D,Dropout,GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.utils import np_utils,print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
data=pd.read_csv("data.csv")
dataset=np.array(data)
np.random.shuffle(dataset)
X = dataset[:,:-1]
X=X.astype('int')
Y = dataset[:,-1]
x_train=X[0:70000, :]
x_train=x_train/255
x_test=X[70000:72000, :]
x_test=x_test/255

Y=Y.reshape(Y.shape[0],1)
y_train=Y[0:70000, :]
y_train=y_train.T
y_test=Y[70000:72000, :]
y_test=y_test.T
//normalization
image_x=32
image_y=32

train_y=np_utils.to_categorical(y_train)
test_y=np_utils.to_categorical(y_test)
train_y=train_y.reshape(train_y.shape[1],test_y.shape[2])
test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
x_train=x_train.reshape(x_train.shape[0],image_x,image_y, 1)
x_test=x_test.reshape(x_test.shape[0],image_x,image_y, 1)
