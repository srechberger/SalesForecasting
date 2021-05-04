from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pickle

# Get Training Data
# All Stores
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')
# Store 708
X_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_y.pkl')


# Fit Model All Stores
rf_model_all = RandomForestRegressor(n_estimators=30)
rf_model_all.fit(X_train, y_train)

# Save Model All Stores
model_all_filename = "../../04_Evaluation/00_Models/rf_model_all.pkl"
with open(model_all_filename, 'wb') as file:
    pickle.dump(rf_model_all, file)

# Fit Model Store 708
rf_model_708 = RandomForestRegressor(n_estimators=30)
rf_model_708.fit(X_train_708, y_train_708)

# Save Model Store 708
model_708_filename = "../../04_Evaluation/00_Models/rf_model_708.pkl"
with open(model_708_filename, 'wb') as file:
    pickle.dump(rf_model_708, file)

# Fit Model Store 198
rf_model_198 = RandomForestRegressor(n_estimators=30)
rf_model_198.fit(X_train_198, y_train_198)

# Save Model Store 198
model_198_filename = "../../04_Evaluation/00_Models/rf_model_198.pkl"
with open(model_198_filename, 'wb') as file:
    pickle.dump(rf_model_198, file)

# Fit Model Store 897
rf_model_897 = RandomForestRegressor(n_estimators=30)
rf_model_897.fit(X_train_897, y_train_897)

# Save Model Store 897
model_897_filename = "../../04_Evaluation/00_Models/rf_model_897.pkl"
with open(model_897_filename, 'wb') as file:
    pickle.dump(rf_model_897, file)


# ----------------------
print("finished")
# ------------------------ Hyperparameter Tuning ----------------------------------

# Grid-Search

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
      estimator=rf,
      param_distributions=random_grid,
      n_iter=100, # iterations - number of different combinations to try
      cv=3, # cross validation - number of folds to use for cross validation
      verbose=2,
      random_state=42,
      n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train, y_train)

# View best parameters
print(rf_random.best_params_)

best_random_model = rf_random.best_estimator_


