# Many Kinds of Layers
# A "layer" in Keras is a very general kind of thing.
# A layer can be, essentially, any kind of data transformation.
# Many layers, like the convolutional and recurrent layers,
# transform data through use of neurons and differ primarily in the pattern of connections they form.
# Others though are used for feature engineering or just simple arithmetic.
# There's a whole world of layers to discover -- check them out!

# https://www.tensorflow.org/api_docs/python/tf/keras/layers

# The Activation Function
# It turns out, however, that two dense layers with nothing in between
# are no better than a single dense layer by itself.
# Dense layers by themselves can never move us out of the world of lines and planes.
# What we need is something nonlinear. What we need are activation functions.

# Without activation functions, neural networks can only learn linear relationships.
# In order to fit curves, we'll need to use activation functions

from tensorflow import keras
from tensorflow.keras import layers

### Building Sequential Models
# The Sequential model we've been using will connect together a list of layers in order from first to last:
# the first layer gets the input, the last layer produces the output. This creates the model in the figure above:

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer
    layers.Dense(units=1),
])

# Be sure to pass all the layers together in a list, like [layer, layer, layer, ...],
# instead of as separate arguments.
# To add an activation function to a layer, just give its name in the activation argument.
