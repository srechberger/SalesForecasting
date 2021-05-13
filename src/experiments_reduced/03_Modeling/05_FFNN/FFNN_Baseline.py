import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# Get Training Data
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_y.pkl')

# Get Test Data
# Store 708
X_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_X_3M.pkl')
y_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_X_3M.pkl')
y_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_X_3M.pkl')
y_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_3M.pkl')

# Transform data (X_*) - drop features + min_max_scaler
# Drop not important features
features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 'DayOfMonth', 'IsPromoMonth']
X_train_708 = X_train_708.drop(features, axis=1)
X_test_3M_708 = X_test_3M_708.drop(features, axis=1)
X_train_198 = X_train_198.drop(features, axis=1)
X_test_3M_198 = X_test_3M_198.drop(features, axis=1)
X_train_897 = X_train_897.drop(features, axis=1)
X_test_3M_897 = X_test_3M_897.drop(features, axis=1)


# Scale values (performance relevant)
scaler = MinMaxScaler()
X_train_708[X_train_708.columns] = scaler.fit_transform(X_train_708[X_train_708.columns])
X_test_3M_708[X_test_3M_708.columns] = scaler.transform(X_test_3M_708[X_test_3M_708.columns])
X_train_198[X_train_198.columns] = scaler.fit_transform(X_train_198[X_train_198.columns])
X_test_3M_198[X_test_3M_198.columns] = scaler.transform(X_test_3M_198[X_test_3M_198.columns])
X_train_897[X_train_897.columns] = scaler.fit_transform(X_train_897[X_train_897.columns])
X_test_3M_897[X_test_3M_897.columns] = scaler.transform(X_test_3M_897[X_test_3M_897.columns])


# ---------------------------- Fit Model - Store 708 ------------------------------------------

# When adding dropout, you may need to increase the number of units in your Dense layers.
ffnn_model_708_bl = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[8]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

# There's nothing to change this time in how we set up the training.
ffnn_model_708_bl.compile(
    optimizer='adam',
    loss='mean_squared_error',
)

# After defining the callback, add it as an argument in fit (you can have several, so put it in a list).
# Choose a large number of epochs when using early stopping, more than you'll need.
history_708 = ffnn_model_708_bl.fit(
    X_train_708, y_train_708,
    validation_data=(X_test_3M_708, y_test_3M_708),
    batch_size=256,
    epochs=100,
    verbose=0,  # turn off training log
)

# Show the learning curves
history_df = pd.DataFrame(history_708.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
# plt.show()

# Save the model
ffnn_model_708_bl.save('../../04_Evaluation/00_Models/ffnn_model_708_bl')


# ---------------------------- Fit Model - Store 198 ------------------------------------------

# When adding dropout, you may need to increase the number of units in your Dense layers.
ffnn_model_198_bl = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[8]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

# There's nothing to change this time in how we set up the training.
ffnn_model_198_bl.compile(
    optimizer='adam',
    loss='mean_squared_error',
)

# After defining the callback, add it as an argument in fit (you can have several, so put it in a list).
# Choose a large number of epochs when using early stopping, more than you'll need.
history_198 = ffnn_model_198_bl.fit(
    X_train_198, y_train_198,
    validation_data=(X_test_3M_198, y_test_3M_198),
    batch_size=256,
    epochs=100,
    verbose=0,  # turn off training log
)

# Show the learning curves
history_df = pd.DataFrame(history_198.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
# plt.show()

# Save the model
ffnn_model_198_bl.save('../../04_Evaluation/00_Models/ffnn_model_198_bl')
    

# ---------------------------- Fit Model - Store 897 ------------------------------------------

# When adding dropout, you may need to increase the number of units in your Dense layers.
ffnn_model_897_bl = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[8]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

# There's nothing to change this time in how we set up the training.
ffnn_model_897_bl.compile(
    optimizer='adam',
    loss='mean_squared_error',
)

# After defining the callback, add it as an argument in fit (you can have several, so put it in a list).
# Choose a large number of epochs when using early stopping, more than you'll need.
history_897 = ffnn_model_897_bl.fit(
    X_train_897, y_train_897,
    validation_data=(X_test_3M_897, y_test_3M_897),
    batch_size=256,
    epochs=100,
    verbose=0,  # turn off training log
)

# Show the learning curves
history_df = pd.DataFrame(history_897.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))
# plt.show()

# Save the model
ffnn_model_897_bl.save('../../04_Evaluation/00_Models/ffnn_model_897_bl')
