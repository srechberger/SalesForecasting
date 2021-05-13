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
df = df.set_index(df.Date)
df = df.drop(['Date'], axis=1)


# ----- Future of Days ---------
x = 90


# Target var
df_eval = df['Sales']


# ------------ Transform Data --------------

# Get values of target
data = df.filter(['Sales'])
dataset = data.values


# Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# ------------ Training Data --------------

# Get length of train data
training_data_len = math.ceil(len(dataset) * .8)


# ----- Get train data ----------------
train_data = scaled_data[0:training_data_len, :]

y_train_data = pd.DataFrame(train_data).shift(x)
y_train_data = y_train_data.dropna()

x_train = []
for i in range(x, len(train_data)):
    x_train.append(train_data[i-x:i, 0])


# Convert to numpy array
x_train, y_train = np.array(x_train), np.array(y_train_data)


# Reshape data from 2D to 3D
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)


# +++++++++++++++++++++ ------------------------------ ############################
# ---------------------            LSTM MODEL          ----------------------------
# +++++++++++++++++++++ ------------------------------ ############################

# Create Model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(70))
model.add(Dense(30))
model.add(Dense(1))


# Complie Model
model.compile(optimizer="adam", loss="mse")


# Fit Model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# Create the test data set
test_data = scaled_data[training_data_len - x:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(x, len(test_data)):
    x_test.append(test_data[i-x:i, 0])


# Convert Data to numpy array
x_test = np.array(x_test)


# Reshape the test data from 2D to 3D
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# +++++++++++++++++++++ ------------------------------ ############################
# ---------------------           PREDICTIONS          ----------------------------
# +++++++++++++++++++++ ------------------------------ ############################

# Predict
predictions = model.predict(x_test)
# Transform normalized predictions to real values
predictions = scaler.inverse_transform(predictions)


# Get RMSE
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)


# Prepare data for plots
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# Plot
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.plot(train['Sales'])
plt.plot(valid[['Sales', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()


# Forecast x=90
df_new_last_x_days = data[-x:].values
last_x_days_scaled = scaler.transform(df_new_last_x_days)

X_test = []
X_test.append(last_x_days_scaled)

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_sales = model.predict(X_test)

pred_sales = scaler.inverse_transform(pred_sales)
print(pred_sales)
