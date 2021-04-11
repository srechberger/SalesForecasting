import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

# In the Fuel Economy dataset your task is to predict the fuel economy of an automobile given features
# like its type of engine or the year it was made.

fuel = pd.read_csv('../../../data/kaggleTutorials/input/fuel.csv')

X = fuel.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Show original data
print(fuel.head())

# Show processed features
print(pd.DataFrame(X[:10,:]).head())

# Define the network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

### Add Loss and Optimizer
model.compile(
    optimizer='adam',
    loss='mae'
)

### Train Model
# Train the network for 200 epochs with a batch size of 128. The input data is X with target y.
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)

# Plot training loss
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot()
plt.show()

# With the learning rate and the batch size, you have some control over:
#
#     How long it takes to train a model
#     How noisy the learning curves are
#     How small the loss becomes

# Change the values for learning_rate, batch_size, and num_examples
# Experiment with different values for the learning rate, batch size, and number of examples
learning_rate = 0.05
batch_size = 32
num_examples = 256

# animate_sgd(
#     learning_rate=learning_rate,
#     batch_size=batch_size,
#     num_examples=num_examples,
    # You can also change these, if you like
#     steps=50, # total training steps (batches seen)
#     true_w=3.0, # the slope of the data
#    true_b=2.0, # the bias of the data
# )

### Learning Rate and Batch Size
# What effect did changing these parameters have?

# You probably saw that smaller batch sizes gave noisier weight updates and loss curves.
# This is because each batch is a small sample of data and smaller samples tend to give noisier estimates.
# Smaller batches can have an "averaging" effect though which can be beneficial.
#
# Smaller learning rates make the updates smaller and the training takes longer to converge.
# Large learning rates can speed up training, but don't "settle in" to a minimum as well.
# When the learning rate is too large, the training can fail completely.
# (Try setting the learning rate to a large value like 0.99 to see this.)
