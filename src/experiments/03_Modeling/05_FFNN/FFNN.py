import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from xgboost import XGBRegressor
import xgboost as xgb

from keras.layers import Dense, InputLayer
from keras.layers import SimpleRNN, LSTM
from keras.models import Sequential
import keras.backend as K

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Get Training Data
# All Stores
train_x = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
train_y = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')

test = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test.pkl')

scaler = MinMaxScaler()

train_x[train_x.columns] = scaler.fit_transform(train_x[train_x.columns])
test[train_x.columns] = scaler.transform(test[train_x.columns])

train_x = train_x.values
train_y = train_y.values
test = test.values

test = test.reshape(test.shape[0], test.shape[1], 1)


x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=.2)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# test.drop(['Id'], axis=1, inplace=True)
# dtrain = xgb.DMatrix(data=x_train[train_x_col], label = y_train[train_y_col])
# dval = xgb.DMatrix(data=x_val[train_x_col], label=y_val[train_y_col])
# dtest = xgb.DMatrix(data=test[train_x_col])


my_batch = 64
my_epoch = 200
my_past = 12
my_split = 0.5
my_neuron = 500  # RNN, LSTM  parameter
my_shape = (my_past, 1)


K.clear_session()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(train_x.shape[1], 1)))
# model.add(SimpleRNN(my_neuron))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(my_neuron, go_backwards=True))
model.add(tf.keras.layers.Dense(my_neuron, activation='relu'))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(my_neuron, activation='relu'))
model.add(tf.keras.layers.Dropout(.3))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.summary()


model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
model.fit(x_train, y_train, batch_size=my_batch, validation_split=.2, epochs=my_epoch, use_multiprocessing=True, verbose=0)


# RNN Evaluate
score = model.evaluate(x_val, y_val, verbose=1)
print('Loss:' + format(score[0], "1.3f"))

pred = model.predict(test)
pred

