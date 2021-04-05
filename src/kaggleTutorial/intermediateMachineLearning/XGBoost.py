import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('../../../data/housingPrices/input/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Create XGBoost model
my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# Make predictions
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

################################################################################
#################### Parameter Tuning ##########################################
################################################################################

# n_estimators
# n_estimators specifies how many times to go through the modeling cycle described above.
# It is equal to the number of models that we include in the ensemble.

# Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
# Too high a value causes overfitting, which causes accurate predictions on training data,
# but inaccurate predictions on test data (which is what we care about).

# Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train)

# early_stopping_rounds
# early_stopping_rounds offers a way to automatically find the ideal value for n_estimators.
# Early stopping causes the model to stop iterating when the validation score stops improving,
# even if we aren't at the hard stop for n_estimators.
# It's smart to set a high value for n_estimators and
# then use early_stopping_rounds to find the optimal time to stop iterating.

# Since random chance sometimes causes a single round where validation scores don't improve,
# you need to specify a number for how many rounds of straight deterioration to allow before stopping.
# Setting early_stopping_rounds=5 is a reasonable choice.
# In this case, we stop after 5 straight rounds of deteriorating validation scores.

# When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores -
# this is done by setting the eval_set parameter.

my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# learning_rate
# Instead of getting predictions by simply adding up the predictions from each component model,
# we can multiply the predictions from each model by a small number (known as the learning rate) before adding them in.

# This means each tree we add to the ensemble helps us less.
# So, we can set a higher value for n_estimators without overfitting.
# If we use early stopping, the appropriate number of trees will be determined automatically.

# In general, a small learning rate and large number of estimators will yield more accurate XGBoost models,
# though it will also take the model longer to train since it does more iterations through the cycle.
# As default, XGBoost sets learning_rate=0.1.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)

# n_jobs
# On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.
# It's common to set the parameter n_jobs equal to the number of cores on your machine.
# On smaller datasets, this won't help.

# The resulting model won't be any better, so micro-optimizing for fitting time is typically nothing but a distraction.
# But, it's useful in large datasets where you would otherwise spend a long time waiting during the fit command.

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
