import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# Gather the data
train_file_path = "data/train_store198.csv"
df = pd.read_csv(train_file_path)

# Parse date column, set index, drop redundant col
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
# df = df.set_index(df.Date)

# Making sure there are no duplicated data
# If there are some duplicates we average the data during those duplicated days
df = df.groupby('Date', as_index=False)['Sales'].mean()

# Sorting the values
df.sort_values('Date', inplace=True)


# Set Params
data = df
Y_var = 'Sales'
lag = 24
LSTM_layer_depth = 100
batch_size = 256
epochs = 20
train_test_split = 0.8


def create_train_test(use_last_n=None):
    # Extracting the main variable we want to model/forecast
    y = data[Y_var].tolist()

    # Subseting the time series if needed
    if use_last_n is not None:
        y = y[-use_last_n:]

    # The X matrix will hold the lags of Y
    X, Y = [], []

    if len(y) - lag <= 0:
        X.append(y)
    else:
        for i in range(len(y) - lag):
            Y.append(y[i + lag])
            X.append(y[i:(i + lag)])

    # Transform to numpy array
    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an LSTM input shape
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Creating training and test sets
    X_train = X
    X_test = []

    Y_train = Y
    Y_test = []

    if train_test_split > 0:
        index = round(len(X) * train_test_split)
        X_train = X[:(len(X) - index)]
        X_test = X[-index:]

        Y_train = Y[:(len(X) - index)]
        Y_test = Y[-index:]

    return X_train, X_test, Y_train, Y_test


# Getting the data
X_train, X_test, Y_train, Y_test = create_train_test()

# Defining the model
model = Sequential()
model.add(LSTM(LSTM_layer_depth, activation='relu', return_sequences=True, input_shape=(lag, 1)))
# model.add(LSTM(100, return_sequences=False))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Defining the model parameter dict
keras_dict = {
    'x': X_train,
    'y': Y_train,
    'batch_size': batch_size,
    'epochs': epochs,
    'shuffle': False
}

if train_test_split > 0:
    keras_dict.update({
        'validation_data': (X_test, Y_test)
    })

# Fitting the model
model.fit(
    **keras_dict
)

yhat = []

if (train_test_split > 0):
    # Getting the last n time series
    _, X_test, _, _ = create_train_test()

    # Making the prediction list
    yhat = [y[0] for y in model.predict(X_test)]


if len(yhat) > 0:

    # Constructing the forecast dataframe
    fc = data.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    # Ploting the forecasts
    plt.figure(figsize=(12, 8))
    for dtype in ['Sales', 'forecast']:
        plt.plot(
            'Date',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()
