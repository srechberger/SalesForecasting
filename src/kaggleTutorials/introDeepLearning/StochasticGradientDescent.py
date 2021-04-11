# In this lesson we're going to see how to train a neural network;
# we're going to see how neural networks learn.

# As with all machine learning tasks, we begin with a set of training data.
# Each example in the training data consists of some features (the inputs)
# together with an expected target (the output).

# Training the network means adjusting its weights in such a way that it can transform the features into the target.

# If we can successfully train a network to do that,
# its weights must represent in some way the relationship between those features and
# that target as expressed in the training data.

# In addition to the training data, we need two more things:
#
#     A "loss function" that measures how good the network's predictions are.
#     An "optimizer" that can tell the network how to change its weights.

##############################################################################
########################### The Loss Function ################################
##############################################################################
# The loss function measures the disparity between the the target's true value and the value the model predicts.
# A common loss function for regression problems is the mean absolute error or MAE.
# Besides MAE, other loss functions you might see for regression problems are
# the mean-squared error (MSE) or the Huber loss (both available in Keras).

##############################################################################
############# The Optimizer - Stochastic Gradient Descent (SGD) ##############
##############################################################################
# We've described the problem we want the network to solve, but now we need to say how to solve it.
# This is the job of the optimizer.
# The optimizer is an algorithm that adjusts the weights to minimize the loss.

# Virtually all of the optimization algorithms used in deep learning belong to a family called
# stochastic gradient descent.
# They are iterative algorithms that train a network in steps.
# One step of training goes like this:
#    1) Sample some training data and run it through the network to make predictions.
#    2) Measure the loss between the predictions and the true values.
#    3) Finally, adjust the weights in a direction that makes the loss smaller.

# Batch Size
# Each iteration's sample of training data is called a minibatch (or often just "batch"),
# while a complete round of the training data is called an epoch.

# Learning Rate
# Notice that the line only makes a small shift in the direction of each batch (instead of moving all the way).
# The size of these shifts is determined by the learning rate.
# A smaller learning rate means the network needs to see
# more minibatches before its weights converge to their best values.

# Adam-Optimizer
# Fortunately, for most work it won't be necessary to do an extensive hyperparameter search to get satisfactory results.
# Adam is an SGD algorithm that has an adaptive learning rate that makes it suitable for most problems
# without any parameter tuning (it is "self tuning", in a sense). Adam is a great general-purpose optimizer.

# model.compile(
#     optimizer="adam",
#     loss="mae",
# )

# Notice that we are able to specify the loss and optimizer with just a string.
# You can also access these directly through the Keras API -- if you wanted to tune parameters,
# for instance -- but for us, the defaults will work fine.

# The gradient is a vector that tells us in what direction the weights need to go.
# More precisely, it tells us how to change the weights to make the loss change fastest.
# We call our process gradient descent because it uses the gradient to descend the loss curve towards a minimum.
# Stochastic means "determined by chance."
# Our training is stochastic because the minibatches are random samples from the dataset.
# And that's why it's called SGD!

### Example
# One thing you might note for now though is that we've rescaled each feature to lie in the interval [0,1].
# As we'll discuss more in Lesson 5, neural networks tend to perform best when their inputs are on a common scale.

import pandas as pd
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

red_wine = pd.read_csv('../../../data/kaggleTutorials/input/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# Print shape of dataframe
print(X_train.shape)

# We've chosen a three-layer network with over 1500 neurons.
# This network should be capable of learning fairly complex relationships in the data.
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# Deciding the architecture of your model should be part of a process.
# Start simple and use the validation loss as your guide.

# After defining the model, we compile in the optimizer and loss function.
model.compile(
    optimizer='adam',
    loss='mae',
)

# Now we're ready to start the training!
# We've told Keras to feed the optimizer 256 rows of the training data at a time (the batch_size) and
# to do that 10 times all the way through the dataset (the epochs).
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)

# use Pandas native plot method
history_df['loss'].plot()
plt.show()

# Notice how the loss levels off as the epochs go by.
# When the loss curve becomes horizontal like that,
# it means the model has learned all it can and there would be no reason continue for additional epochs.
