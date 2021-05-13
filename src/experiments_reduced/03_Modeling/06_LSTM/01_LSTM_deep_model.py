# Ploting packages
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Date wrangling
from datetime import datetime, timedelta

# Data wrangling
import pandas as pd

# The deep learning class
from deep_model import DeepModelTS


# Reading the data
# Store 198
d = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198.pkl')
d['Datetime'] = d.index
d = d.loc[:, ['Sales', 'Datetime', 'DayOfWeek', 'Promo', 'StateHoliday']]

# Making sure there are no duplicated data
# If there are some duplicates we average the data during those duplicated days
d = d.groupby('Datetime', as_index=False)['Sales'].mean()

# Sorting the values
d.sort_values('Datetime', inplace=True)

# Set Params
# 0.1
train_test_split = 0.1  # Prozentueller Anteil Testdaten
# 7
lag = 24  # Die Anzahl der Verzögerungen, die für die Modellierung verwendet werden
# 40
LSTM_layer_depth = 40  # Anzahl der Neuronen in der LSTM-Schicht
# 10
epochs = 10  # Anzahl der Trainingsschleifen (Vorwärtsausbreitung zu Rückwärtsausbreitungszyklen)
# 7
batch_size = 20 # Die Größe der Datenstichprobe für den Gradientenabstieg,
                # der beim Ermitteln der Parameter durch das Deep-Learning-Modell verwendet wird.
                # Alle Daten werden in Blöcke mit Batch-Größen unterteilt und über das Netzwerk eingespeist.
                # Die internen Parameter des Modells werden aktualisiert,
                # nachdem jede Stapelgröße von Daten im Modell vorwärts und rückwärts geht.

# Initiating the class
deep_learner = DeepModelTS(
    data=d,  # Daten für Modellerstellung
    Y_var='Sales',  # Y_var - Zielvariable
    lag=lag,
    LSTM_layer_depth=LSTM_layer_depth,
    epochs=epochs,
    batch_size=batch_size,
    train_test_split=train_test_split
)

# Fitting the model
model = deep_learner.LSTModel()

# Making the prediction on the validation set
# Only applicable if train_test_split > 0
yhat = deep_learner.predict()

if len(yhat) > 0:

    # Constructing the forecast dataframe
    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    # Ploting the forecasts
    plt.figure(figsize=(12, 8))
    for dtype in ['Sales', 'forecast']:
        plt.plot(
            'Datetime',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    plt.legend()
    plt.grid()
    plt.show()

# Forecasting n steps ahead

# Creating the model using full data and forecasting n steps ahead
deep_learner = DeepModelTS(
    data=d,
    Y_var='Sales',
    lag=lag,
    LSTM_layer_depth=LSTM_layer_depth,
    epochs=epochs,
    train_test_split=0
)

# Fitting the model
model = deep_learner.LSTModel()

# Forecasting n steps ahead
n_ahead = 90
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Constructing the forecast dataframe
fc = d.tail(400).copy()
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(days=x + 1) for x in range(n_ahead)],
    'Sales': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

# Ploting the forecasts
plt.figure(figsize=(12, 8))
for col_type in ['original', 'forecast']:
    plt.plot(
        'Datetime',
        'Sales',
        data=fc[fc['type'] == col_type],
        label=col_type
    )

plt.legend()
plt.grid()
plt.show()
