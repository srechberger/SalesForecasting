import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize'] = (12.0, 10.0)

##### Get data
# save filepath to variable for easier access
rossmann_file_path = "../../../data/rossmann/input/train.csv"

# read the data and store data in DataFrame titled melbourne_data
types = {'StateHoliday': np.dtype(str)}
rossmann_data = pd.read_csv(rossmann_file_path, dtype=types, parse_dates=[2], nrows=70000)
# rossmann_data = pd.read_csv(rossmann_file_path, low_memory=False)
# store = pd.read_csv("store.csv")

##### Split data into training and test data
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# print a summary of the data in Melbourne data
rossmann_data.describe()

# show data metrics
print(rossmann_data.describe())

# show data columns
print(rossmann_data.columns)

# dropna drops missing values (think of na as "not available")
rossmann_data = rossmann_data.dropna(axis=0)

# y = prediction target (endogene Variable)
y = rossmann_data.Sales

# choosing features for prediction (exogene Variablen)
rossmann_features = ['Promo', 'StateHoliday', 'SchoolHoliday']

# assign features to X
X = rossmann_data[rossmann_features]

# show data metrics for selected features
print(X.describe())

# show example data for first 5 rows
print(X.head())

##### Build first prediction model

# Define model. Specify a number for random_state to ensure same results each run
rossmann_model = DecisionTreeRegressor(random_state=1)

# Fit model
rossmann_model.fit(X, y)

# Prediction
print("Making predictions for the 5 rows:")
print(X.head())
print("The predictions are")
print(rossmann_model.predict(X.head()))

predictions = rossmann_model.predict(X)

##### Model validation
mean_absolute_error(y, predictions)


##### Overfitting and Underfitting

# function for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# find best tree size
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = min(scores, key=scores.get)

