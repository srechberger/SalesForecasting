# In this lesson, we'll learn about a two kinds of special layers, not containing any neurons themselves,
# but that add some functionality that can sometimes benefit a model in various ways.
# Both are commonly used in modern architectures.

### Dropout and Batch Normalization

### Dropout
# The first of these is the "dropout layer", which can help correct overfitting.

# To break up these conspiracies, we randomly drop out some fraction of a layer's input units every step of training,
# making it much harder for the network to learn those spurious patterns in the training data.
# Instead, it has to search for broad, general patterns, whose weight patterns tend to be more robust.

# Adding Dropout
# In Keras, the dropout rate argument rate defines what percentage of the input units to shut off.
# Put the Dropout layer just before the layer you want the dropout applied to:

# keras.Sequential([
#     # ...
#     layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
#     layers.Dense(16),
#     # ...
# ])

### Batch Normalization
# The next special layer we'll look at performs "batch normalization" (or "batchnorm"),
# which can help correct training that is slow or unstable.

# With neural networks, it's generally a good idea to put all of your data on a common scale,
# perhaps with something like scikit-learn's StandardScaler or MinMaxScaler.
# Features that tend to produce activations of very different sizes can make for unstable training behavior.

# Now, if it's good to normalize the data before it goes into the network,
# maybe also normalizing inside the network would be better!
# In fact, we have a special kind of layer that can do this, the batch normalization layer.

# A batch normalization layer looks at each batch as it comes in,
# first normalizing the batch with its own mean and standard deviation,
# and then also putting the data on a new scale with two trainable rescaling parameters. Batchnorm, in effect,
# performs a kind of coordinated rescaling of its inputs.

# Models with batchnorm tend to need fewer epochs to complete training.
# Moreover, batchnorm can also fix various problems that can cause the training to get "stuck".
# Consider adding batch normalization to your models, especially if you're having trouble during training.

### Adding Batch Normalization
# It seems that batch normalization can be used at almost any point in a network. You can put it after a layer...

# layers.Dense(16, activation='relu'),
# layers.BatchNormalization(),

# ... or between a layer and its activation function:

# layers.Dense(16),
# layers.BatchNormalization(),
# layers.Activation('relu'),

# And if you add it as the first layer of your network it can act as a kind of adaptive preprocessor,
# standing in for something like Sci-Kit Learn's StandardScaler.

### Examples
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

red_wine = pd.read_csv('../../../data/kaggleTutorials/input/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# When adding dropout, you may need to increase the number of units in your Dense layers.
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
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
model.compile(
    optimizer='adam',
    loss='mae',
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    verbose=0,
)


# Show the learning curves
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.show()

# You'll typically get better performance if you standardize your data before using it for training.
# That we were able to use the raw data at all,
# however, shows how effective batch normalization can be on more difficult datasets.
