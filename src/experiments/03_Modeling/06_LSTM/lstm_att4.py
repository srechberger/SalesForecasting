from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd

# Get Training Data
# Store 198
train_df = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198.pkl')

# Get Test Data
# Store 198
test_df = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198.pkl')

# Transform data (X_*) - drop features + min_max_scaler
# Drop not important features
features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 'DayOfMonth', 'IsPromoMonth']
train_df = train_df.drop(features, axis=1)


# Split the data
x = train_df.drop(['Sales'], axis=1)
y = train_df['Sales']


x = np.array(x)
y = np.array(y)

print('Getting data ...')
# x = np.concatenate(np.load('bigx.npy'))
# y = np.concatenate(np.load('bigy.npy'))

#print(x.shape[2])
#in_neurons = x.shape[2]
hidden_neurons = 500
hidden_neurons_2 = 500
out_neurons = 1
nb_epoch = 10
evaluation = []

print('Creating simple DLSTM ...')
model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=hidden_neurons, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(out_neurons, input_dim=hidden_neurons))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

print('Fitting model ...')
print(model.evaluate(x, y, verbose=0))
model.fit(x, y, validation_split=0.05, batch_size=50, shuffle=True, epochs=10, verbose=2)

print('Done ...')
