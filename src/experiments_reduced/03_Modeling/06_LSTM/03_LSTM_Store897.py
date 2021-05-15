import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import datetime

# Gather the data
file_path = "data/sales_897_3M.csv"
df = pd.read_csv(file_path)

# Parse date column, set index, drop redundant col
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# Making sure there are no duplicated data
# If there are some duplicates we average the data during those duplicated days
df = df.groupby('Date', as_index=False)['Sales'].mean()

# Sorting the values
df.sort_values('Date', inplace=True)

# Set Params
data = df               # Daten

Y_var = 'Sales'         # Zielvariable

lag = 7                 # Die Anzahl der Verzögerungen, die für die Modellierung verwendet werden

LSTM_layer_depth = 100  # Anzahl der Neuronen in der LSTM-Schicht

batch_size = 72         # Die Größe der Datenstichprobe für den Gradientenabstieg,
                        # der beim Ermitteln der Parameter durch das Deep-Learning-Modell verwendet wird.
                        # Alle Daten werden in Blöcke mit Batch-Größen unterteilt und über das Netzwerk eingespeist.
                        # Die internen Parameter des Modells werden aktualisiert,
                        # nachdem jede Stapelgröße von Daten im Modell vorwärts und rückwärts geht.

epochs = 100            # Anzahl der Trainingsschleifen (Vorwärtsausbreitung zu Rückwärtsausbreitungszyklen)

train_test_split = 0.11070111  # Prozentueller Anteil Testdaten
                               # Gesamtanzahl Datensätze = 813 (820 - 7 Verzögerungen)
                               # Testdatensätze = 90 (01/01/2015 - 31/03/2015)
                               # 90 / 813 = 0.11070111

# Prediction horizon
train_date_start = datetime.datetime(2014, 11, 1)
fc_date_start = datetime.datetime(2015, 1, 1)
fc_date_2W = datetime.datetime(2015, 1, 14)
fc_date_1M = datetime.datetime(2015, 1, 31)
fc_date_3M = datetime.datetime(2015, 3, 31)

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
    print(X_test)


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


# Constructing the forecast dataframe
fc = data.tail(len(yhat)).copy()
fc.reset_index(inplace=True)
fc['forecast'] = yhat
fc = fc.set_index(fc.Date)
fc['forecast'] = fc['forecast'].astype(float)
fc = fc['forecast']
fc = fc.sort_index(ascending=False)

# Store Predictions
fc_train = fc.loc[(fc.index >= train_date_start) & (fc.index < fc_date_start)]
fc_train.to_pickle('predictions/pred_897_train.pkl')
fc_2W = fc.loc[(fc.index >= fc_date_start) & (fc.index <= fc_date_2W)]
fc_2W.to_pickle('predictions/pred_897_2W.pkl')
fc_1M = fc.loc[(fc.index >= fc_date_start) & (fc.index <= fc_date_1M)]
fc_1M.to_pickle('predictions/pred_897_1M.pkl')
fc_3M = fc.loc[(fc.index >= fc_date_start) & (fc.index <= fc_date_3M)]
fc_3M.to_pickle('predictions/pred_897_3M.pkl')
