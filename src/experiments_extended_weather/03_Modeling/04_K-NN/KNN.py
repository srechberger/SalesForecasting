from sklearn import neighbors
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV


# Get Training Data
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/train_store897_y.pkl')

# --------------------------- Baseline Models  ------------------------------------------

# Store 708
knn_model_708_bl = neighbors.KNeighborsRegressor()
knn_model_708_bl.fit(X_train_708, y_train_708)

# Save Model
knn_model_708_bl_filename = "../../04_Evaluation/00_Models/knn_model_708_bl.pkl"
with open(knn_model_708_bl_filename, 'wb') as file:
    pickle.dump(knn_model_708_bl, file)


# Store 198
knn_model_198_bl = neighbors.KNeighborsRegressor()
knn_model_198_bl.fit(X_train_198, y_train_198)

# Save Model
knn_model_198_bl_filename = "../../04_Evaluation/00_Models/knn_model_198_bl.pkl"
with open(knn_model_198_bl_filename, 'wb') as file:
    pickle.dump(knn_model_198_bl, file)


# Store 897
knn_model_897_bl = neighbors.KNeighborsRegressor()
knn_model_897_bl.fit(X_train_897, y_train_897)

# Save Model
knn_model_897_bl_filename = "../../04_Evaluation/00_Models/knn_model_897_bl.pkl"
with open(knn_model_897_bl_filename, 'wb') as file:
    pickle.dump(knn_model_897_bl, file)


# --------------------------- Hyperparameter Tuning -------------------------------------

# Hyperparameters (Grid Search
leaf_size = list(range(1, 50))
n_neighbors = list(range(1, 30))
p = [1, 2]

# Convert to dictionary
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)


# --------------------------- Store 708 -------------------------------------------------

# Create KNN-Model
knn_clf_708 = neighbors.KNeighborsRegressor()

# GridSearch
clf = GridSearchCV(knn_clf_708, hyperparameters, cv=10)

# Find best params with Grid Search
best_model_708 = clf.fit(X_train_708, y_train_708)

# Best Hyperparameters
print('Store 708')
print('Best leaf_size:', best_model_708.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model_708.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model_708.best_estimator_.get_params()['n_neighbors'])
# Best leaf_size: 42
# Best p: 1
# Best n_neighbors: 26

# Fit Model
knn_708 = neighbors.KNeighborsRegressor(leaf_size=42, p=1, n_neighbors=26)
knn_708.fit(X_train_708, y_train_708)

# Save Model
knn_model_708_filename = "../../04_Evaluation/00_Models/knn_model_708.pkl"
with open(knn_model_708_filename, 'wb') as file:
    pickle.dump(knn_708, file)


# --------------------------- Store 198 -------------------------------------------------

# Create KNN-Model
knn_clf_198 = neighbors.KNeighborsRegressor()

# GridSearch
clf = GridSearchCV(knn_clf_198, hyperparameters, cv=10)

# Find best params with Grid Search
best_model_198 = clf.fit(X_train_198, y_train_198)

# Best Hyperparameters
print('Store 198')
print('Best leaf_size:', best_model_198.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model_198.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model_198.best_estimator_.get_params()['n_neighbors'])
# Best leaf_size: 1
# Best p: 1
# Best n_neighbors: 7

# Fit model
knn_198 = neighbors.KNeighborsRegressor(leaf_size=1, p=1, n_neighbors=7)
knn_198.fit(X_train_198, y_train_198)

# Save Model
knn_model_198_filename = "../../04_Evaluation/00_Models/knn_model_198.pkl"
with open(knn_model_198_filename, 'wb') as file:
    pickle.dump(knn_198, file)


# --------------------------- Store 897 -------------------------------------------------

# Create KNN-Model
knn_clf_897 = neighbors.KNeighborsRegressor()

# GridSearch
clf = GridSearchCV(knn_clf_897, hyperparameters, cv=10)

# Find best params with Grid Search
best_model_897 = clf.fit(X_train_897, y_train_897)

# Best Hyperparameters
print('Store 897')
print('Best leaf_size:', best_model_897.best_estimator_.get_params()['leaf_size'])
print('Best p:', best_model_897.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model_897.best_estimator_.get_params()['n_neighbors'])
# Best leaf_size: 3
# Best p: 1
# Best n_neighbors: 29

# Fit model
knn_897 = neighbors.KNeighborsRegressor(leaf_size=3, p=1, n_neighbors=29)
knn_897.fit(X_train_897, y_train_897)

# Save Model
knn_model_897_filename = "../../04_Evaluation/00_Models/knn_model_897.pkl"
with open(knn_model_897_filename, 'wb') as file:
    pickle.dump(knn_897, file)
