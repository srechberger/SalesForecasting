import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Load Data
concrete = pd.read_csv('../../../data/kaggleTutorials/input/concrete.csv')
print(concrete.head())

# Define feature count
input_shape = [8]


# Define a model with hidden layers
# Now create a model with three hidden layers, each having 512 units and the ReLU activation.
# Be sure to include an output layer of one unit and no activation,
# and also input_shape as an argument to the first layer.

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# Activation Layers
# The usual way of attaching an activation function to a Dense layer is to include it as part of the definition
# with the activation argument.
# Sometimes though you'll want to put some other layer between the Dense layer and its activation function.
# (We'll see an example of this in Lesson 5 with batch normalization.)
# In this case, we can define the activation in its own Activation layer, like so:

# layers.Dense(units=8),
# layers.Activation('relu')

model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])

# Alternatives to ReLU
# There is a whole family of variants of the 'relu' activation -- 'elu', 'selu', and 'swish', among others
# -- all of which you can use in Keras.
# Sometimes one activation will perform better than another on a given task,
# so you could consider experimenting with activations as you develop a model.
# The ReLU activation tends to do well on most problems, so it's a good one to start with.

# Change 'relu' to 'elu', 'selu', 'swish'... or something else
# https://www.tensorflow.org/api_docs/python/tf/keras/activations
activation_layer = layers.Activation('relu')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
