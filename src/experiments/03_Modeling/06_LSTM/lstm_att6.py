import numpy as np
import pandas as pd
from datetime import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns
# sklearn für Überwachtes Lernen
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
# keras für Neuronale Netze
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


# Display dataframe
def display_dataframe(df, rows=6, cols=None):
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols):
        print(df)


# Reading the data
# Store 198
opsd_df = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198.pkl')

# Drop not important features
opsd_df = opsd_df.drop([
    'Store',
    'StoreType',
    'Assortment',
    'CompetitionDistance',
    'Promo2',
    'DayOfMonth',
    'IsPromoMonth'],
    axis=1)

display_dataframe(opsd_df, 6)

series = opsd_df['Sales']
anzZ = opsd_df.shape[0]  # Anzahl Zeilen

# ------ Scale the Data -------
# Sales als 2D numpy-Array
sales = series.values.reshape(anzZ, 1)
# Skaliere die Daten auf den Wertebereich (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
print("Sales (unskaliert)")
print(sales[0:3])
sales = scaler.fit_transform(sales)
print("Sales (skaliert)")
print(sales[0:3])

# ------ Split the Data in train and test -------
# Anteil der Trainingsdaten für Validierung
VALIDATION_SPLIT = 0.1

# Letzter Index der Trainingsdaten
train_size = int(anzZ*(1-VALIDATION_SPLIT))

# Erster Index der Testdaten
test_size = anzZ - train_size
train = sales[0:train_size, :]
test = sales[train_size:anzZ, :]
print("Trainingsdaten:")
print(train.shape[0])
print("Testdaten:")
print(test.shape[0])


# ------ Transform the Data into unsupervised learning base -------
def erzeuge_bewertung(data, timesteps=7):
   X, Y = [], []
   for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, 0])
        Y.append(data[i, 0])
   return np.array(X), np.array(Y)


# X = Merkmale, Y = Zielvariable
X_Train, Y_Train = erzeuge_bewertung(train)
X_Test, Y_Test = erzeuge_bewertung(test)

# Function to build the FFNN model with a variable number of nodes
def build_model(neurons):
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(X_Train.shape[1], X_Train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Construct the Regressor model
regressor = KerasRegressor(build_fn=build_model, verbose=0)
parameters = {'batch_size': [5, 10, 20],
              'epochs': [300],
              'neurons': [20, 25, 30, 40]}

grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           cv=10)

# Fit the various models using the object defined above
grid_search = grid_search.fit(X_Train, Y_Train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print('Best Parameters:', best_parameters)
# Best Parameters: {'batch_size': 5, 'epochs': 300, 'neurons': 40}

# -----------------------------------------------------------------------------------------------
# Build model with best params of GridSearchCV

# Now we build and train the optimal model
ffnn_model = build_model(neurons=best_parameters['neurons'])

# Store the history of loss of the optimal function
history = ffnn_model.fit(X_Train,
                         Y_Train,
                         epochs=best_parameters['epochs'],
                         batch_size=best_parameters['batch_size'],
                         verbose=0)

# Save the model
ffnn_model.save('../../04_Evaluation/00_Models/ffnn_model_198_gs')

# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.plot()
path = "../../../../data/rossmann/output/ffnn_learning_curve_198"
plt.savefig(path)
# plt.show()

