from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Get Training Data
# All Stores
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/01_AllStores/train_y.pkl')
# Store 708
X_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/train_store897_y.pkl')

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


# ------------------------ Hyperparameter Tuning ----------------------------------

def plot_validation_curve(X_train_data, y_train_data, param_range, param_name):
    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(RandomForestRegressor(),
                                                 X=X_train_data,
                                                 y=y_train_data,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=3)

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training Score", color="darkgoldenrod")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="indigo")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="palegoldenrod")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="mediumpurple")

    # Create plot
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Accuracy Score")
    plt.tight_layout()
    plt.legend(loc="best")
    # plt.show()


# +++++ Store 708 +++++
# Validation Curve - n_estimators
range_n_estimators = np.arange(1, 30, 2)
plot_validation_curve(X_train_708, y_train_708, range_n_estimators, 'n_estimators')
# result: n_estimators = 21

# Validation Curve - max_depth
range_max_depth = np.arange(2, 30, 1)
plot_validation_curve(X_train_708, y_train_708, range_max_depth, 'max_depth')
# result: max_depth = 7

# Validation Curve - min_samples_split
range_min_samples_split = np.arange(2, 30, 1)
plot_validation_curve(X_train_708, y_train_708, range_min_samples_split, 'min_samples_split')
# result: min_samples_split = 8

# Validation Curve - min_samples_leaf
range_min_samples_leaf = np.arange(2, 30, 1)
plot_validation_curve(X_train_708, y_train_708, range_min_samples_leaf, 'min_samples_leaf')
# result: min_samples_leaf = 7

# Fit Model Store 708
rf_model_708_vc = RandomForestRegressor(n_estimators=21, max_depth=7, min_samples_split=8, min_samples_leaf=7)
rf_model_708_vc.fit(X_train_708, y_train_708)

# Save Model Store 708
model_708_vc_filename = "../../04_Evaluation/00_Models/rf_model_708_vc.pkl"
with open(model_708_vc_filename, 'wb') as file:
    pickle.dump(rf_model_708_vc, file)

# +++++ Store 198 +++++
# Validation Curve - n_estimators
range_n_estimators = np.arange(1, 30, 2)
plot_validation_curve(X_train_198, y_train_198, range_n_estimators, 'n_estimators')
# result: n_estimators = 25

# Validation Curve - max_depth
range_max_depth = np.arange(2, 30, 1)
plot_validation_curve(X_train_198, y_train_198, range_max_depth, 'max_depth')
# result: max_depth = 7

# Validation Curve - min_samples_split
range_min_samples_split = np.arange(2, 30, 1)
plot_validation_curve(X_train_198, y_train_198, range_min_samples_split, 'min_samples_split')
# result: min_samples_split = 18

# Validation Curve - min_samples_leaf
range_min_samples_leaf = np.arange(2, 30, 1)
plot_validation_curve(X_train_198, y_train_198, range_min_samples_leaf, 'min_samples_leaf')
# result: min_samples_leaf = 2

# Fit Model Store 198
rf_model_198_vc = RandomForestRegressor(n_estimators=21, max_depth=7, min_samples_split=8, min_samples_leaf=7)
rf_model_198_vc.fit(X_train_198, y_train_198)

# Save Model Store 198
model_198_vc_filename = "../../04_Evaluation/00_Models/rf_model_198_vc.pkl"
with open(model_198_vc_filename, 'wb') as file:
    pickle.dump(rf_model_198_vc, file)

# +++++ Store 897 +++++
# Validation Curve - n_estimators
range_n_estimators = np.arange(1, 50, 2)
plot_validation_curve(X_train_897, y_train_897, range_n_estimators, 'n_estimators')
# result: n_estimators = 32

# Validation Curve - max_depth
range_max_depth = np.arange(2, 30, 1)
plot_validation_curve(X_train_897, y_train_897, range_max_depth, 'max_depth')
# result: max_depth = 2

# Validation Curve - min_samples_split
range_min_samples_split = np.arange(2, 50, 1)
plot_validation_curve(X_train_897, y_train_897, range_min_samples_split, 'min_samples_split')
# result: min_samples_split = 20

# Validation Curve - min_samples_leaf
range_min_samples_leaf = np.arange(2, 30, 1)
plot_validation_curve(X_train_897, y_train_897, range_min_samples_leaf, 'min_samples_leaf')
# result: min_samples_leaf = 7

# Fit Model Store 897
rf_model_897_vc = RandomForestRegressor(n_estimators=21, max_depth=7, min_samples_split=8, min_samples_leaf=7)
rf_model_897_vc.fit(X_train_897, y_train_897)

# Save Model Store 897
model_897_vc_filename = "../../04_Evaluation/00_Models/rf_model_897_vc.pkl"
with open(model_897_vc_filename, 'wb') as file:
    pickle.dump(rf_model_897_vc, file)


# -------------- Grid-Search ------------------

# Estimators
n_estimators = [10, 15, 20, 25, 30, 40]
# Maximum number of levels in tree
max_depth = [5, 8, 10]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 5, 8]

# Create a grid to search for best hyperparameters
grid = {'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf}

# Create a base model to tune
rf = RandomForestRegressor()

# Apply GridSearch
gridRF_708 = GridSearchCV(rf, grid, cv=3, verbose=1, n_jobs=-1)
gridRF_198 = GridSearchCV(rf, grid, cv=3, verbose=1, n_jobs=-1)
gridRF_897 = GridSearchCV(rf, grid, cv=3, verbose=1, n_jobs=-1)

# Fit Models
bestRF_708 = gridRF_708.fit(X_train_708, y_train_708)
bestRF_198 = gridRF_198.fit(X_train_198, y_train_198)
bestRF_897 = gridRF_897.fit(X_train_897, y_train_897)

# Print best params
print('Store 708: ', bestRF_708.best_params_)
# Store 708:  {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 5, 'n_estimators': 15}

# Fit Model Store 708
rf_model_708_gs = RandomForestRegressor(n_estimators=15, max_depth=10, min_samples_split=5, min_samples_leaf=5)
rf_model_708_gs.fit(X_train_708, y_train_708)

# Save Model Store 708
model_708_gs_filename = "../../04_Evaluation/00_Models/rf_model_708_gs.pkl"
with open(model_708_gs_filename, 'wb') as file:
    pickle.dump(rf_model_708_gs, file)


print('Store 198: ', bestRF_198.best_params_)
# Store 198:  {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 10}

# Fit Model Store 198
rf_model_198_gs = RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=2)
rf_model_198_gs.fit(X_train_198, y_train_198)

# Save Model Store 198
model_198_gs_filename = "../../04_Evaluation/00_Models/rf_model_198_gs.pkl"
with open(model_198_gs_filename, 'wb') as file:
    pickle.dump(rf_model_198_gs, file)


print('Store 897: ', bestRF_897.best_params_)
# Store 897:  {'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 20}

# Fit Model Store 897
rf_model_897_gs = RandomForestRegressor(n_estimators=20, max_depth=10, min_samples_split=2, min_samples_leaf=5)
rf_model_897_gs.fit(X_train_897, y_train_897)

# Save Model Store 897
model_897_gs_filename = "../../04_Evaluation/00_Models/rf_model_897_gs.pkl"
with open(model_897_gs_filename, 'wb') as file:
    pickle.dump(rf_model_897_gs, file)
