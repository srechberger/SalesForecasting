# Imports
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# Setup plotting
plt.style.use('seaborn-whitegrid')

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Load data
red_wine = pd.read_csv('../../../data/kaggleTutorials/input/red-wine.csv')

# Show head of data
print(red_wine.head())

# Get shape of dataframe
# (rows, columns) --> (1599, 12)
print(red_wine.shape)

# Set up input shape for keras model
# The target is 'quality', and the remaining columns are the features.
input_shape = [11]

# Define a linear model
model = keras.Sequential([
    layers.Dense(units=1, input_shape=input_shape)
])

# Internally, Keras represents the weights of a neural network with tensors.
# Tensors are basically TensorFlow's version of a Numpy array with a few differences
# that make them better suited to deep learning.
#
# One of the most important is that tensors are compatible with GPU and TPU) accelerators.
# TPUs, in fact, are designed specifically for tensor computations.

# A model's weights are kept in its weights attribute as a list of tensors.
# Get the weights of the model you defined above.
w, b = model.weights

print("Weights\n{}\n\nBias\n{}".format(w, b))

# Plot the output of an untrained linear model

# The kinds of problems we'll work on through Lesson 5 will be regression problems,
# where the goal is to predict some numeric target. Regression problems are like "curve-fitting" problems:
# we're trying to find a curve that best fits the data.
# Let's take a look at the "curve" produced by a linear model. (You've probably guessed that it's a line!)

# We mentioned that before training a model's weights are set randomly.
# Run the cell below a few times to see the different lines produced with a random initialization.

model2 = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model2.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model2.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()
