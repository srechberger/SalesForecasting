from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dense, LSTM, Flatten, Dropout, Input
from keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

# Get Training Data
# Store 198
from tensorflow.python.keras.models import Model

X_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y.pkl')

# Get Test Data
# Store 198
X_test = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_3M.pkl')
y_test = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_3M.pkl')

# Drop not important features
X_train = X_train.drop([
    'Store',
    'StoreType',
    'Assortment',
    'CompetitionDistance',
    'Promo2',
    'DayOfMonth',
    'IsPromoMonth'],
    axis=1)

X_valid = X_test.drop([
    'Store',
    'StoreType',
    'Assortment',
    'CompetitionDistance',
    'Promo2',
    'DayOfMonth',
    'IsPromoMonth'],
    axis=1)

print(X_train.dtypes)
features = ['DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'Month', 'Year', 'WeekOfYear']

# Scale values (performance relevant)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])
X_valid[X_valid.columns] = scaler.transform(X_valid[X_valid.columns])

# reformatting the datasets
x_train_array = np.copy(X_train[features].values)
x_valid_array = np.copy(X_test[features].values)

format_sub_train_x = x_train_array.reshape((x_train_array.shape[0], 1, x_train_array.shape[1]))
format_sub_val_x = x_valid_array.reshape((x_valid_array.shape[0], 1, x_valid_array.shape[1]))


model = Sequential()
model.add(LSTM(50, input_shape=(format_sub_train_x.shape[1], format_sub_train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

mod_1 = model.fit(format_sub_train_x, np.copy(y_train),
                  epochs=100,
                  batch_size=100,
                  verbose=2,
                  shuffle=False)
