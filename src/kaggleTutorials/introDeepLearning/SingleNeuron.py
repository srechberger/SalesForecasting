# Linear Units in Keras

# The easiest way to create a model in Keras is through keras.Sequential,
# which creates a neural network as a stack of layers.
# We can create models using a dense layer.

# We could define a linear model accepting three input features ('sugars', 'fiber', and 'protein') and
# producing a single output ('calories') like so:

from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

# UNITS
# With the first argument, units, we define how many outputs we want.
# In this case we are just predicting 'calories', so we'll use units=1.

# With the second argument, input_shape, we tell Keras the dimensions of the inputs.
# Setting input_shape=[3] ensures the model will accept three features as input ('sugars', 'fiber', and 'protein').
