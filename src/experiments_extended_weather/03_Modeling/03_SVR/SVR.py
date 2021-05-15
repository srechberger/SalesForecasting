from sklearn.svm import SVR
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

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

# --------- Store 708 -----------

# Choose best kernel for data
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = SVR(kernel=k)
    clf.fit(X_train_708, y_train_708)
    confidence = clf.score(X_train_708, y_train_708)
    print(k, confidence)

# linear 0.1290668132333268
# poly -0.006135371366612574
# rbf -0.006135417671555787
# sigmoid -0.006135462504934575

# Fit model
svr_model_708_bl = SVR(kernel='linear', C=1, gamma=0.5)
svr_model_708_bl.fit(X_train_708, y_train_708)

# Save Model
svr_model_708_bl_filename = "../../04_Evaluation/00_Models/svr_model_708_bl.pkl"
with open(svr_model_708_bl_filename, 'wb') as file:
    pickle.dump(svr_model_708_bl, file)


# --------- Store 198 -----------

# Choose best kernel for data
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = SVR(kernel=k)
    clf.fit(X_train_198, y_train_198)
    confidence = clf.score(X_train_198, y_train_198)
    print(k, confidence)

# linear 0.1290668132333268
# poly -0.006135371366612574
# rbf -0.006135417671555787
# sigmoid -0.006135462504934575

# Fit model
svr_model_198_bl = SVR(kernel='linear', C=1, gamma=0.5)
svr_model_198_bl.fit(X_train_198, y_train_198)

# Save Model
svr_model_198_bl_filename = "../../04_Evaluation/00_Models/svr_model_198_bl.pkl"
with open(svr_model_198_bl_filename, 'wb') as file:
    pickle.dump(svr_model_198_bl, file)
    
    
# --------- Store 897 -----------

# Choose best kernel for data
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
    clf = SVR(kernel=k)
    clf.fit(X_train_897, y_train_897)
    confidence = clf.score(X_train_897, y_train_897)
    print(k, confidence)

# linear 0.1290668132333268
# poly -0.006135371366612574
# rbf -0.006135417671555787
# sigmoid -0.006135462504934575

# Fit model
svr_model_897_bl = SVR(kernel='linear', C=1, gamma=0.5)
svr_model_897_bl.fit(X_train_897, y_train_897)

# Save Model
svr_model_897_bl_filename = "../../04_Evaluation/00_Models/svr_model_897_bl.pkl"
with open(svr_model_897_bl_filename, 'wb') as file:
    pickle.dump(svr_model_897_bl, file)


# -------------------------- Hyperparameter Tuning ---------------------------------

# ----------- Store 708 -----------

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
              'gamma': [1, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

K = 5
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_708 = GridSearchCV(SVR(epsilon=0.01), param_grid, cv=K, scoring=scorer)

# fitting the model for grid search
grid_708.fit(X_train_708, y_train_708)

# Print best hyperparameters
print('Grid Search Store 708')
print(grid_708.best_params_)
print(grid_708.best_estimator_)
print('----------------------------------')
# {'C': 10000, 'gamma': 0.001, 'kernel': 'rbf'}
# SVR(C=10000, epsilon=0.01, gamma=0.001)

# Fit Model
svr_model_708_gs = SVR(kernel='rbf', C=10000, epsilon=0.01, gamma=0.001)
svr_model_708_gs.fit(X_train_708, y_train_708)

# Save Model
svr_model_708_gs_filename = "../../04_Evaluation/00_Models/svr_model_708_gs.pkl"
with open(svr_model_708_gs_filename, 'wb') as file:
    pickle.dump(svr_model_708_gs, file)


# ----------- Store 198 -----------

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
              'gamma': [1, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

K = 5
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_198 = GridSearchCV(SVR(epsilon=0.01), param_grid, cv=K, scoring=scorer)

# fitting the model for grid search
grid_198.fit(X_train_198, y_train_198)

# Print best hyperparameters
print('Grid Search Store 198')
print(grid_198.best_params_)
print(grid_198.best_estimator_)
print('----------------------------------')
# {'C': 10000, 'gamma': 0.001, 'kernel': 'rbf'}
# SVR(C=10000, epsilon=0.01, gamma=0.001)

# Fit Model
svr_model_198_gs = SVR(kernel='rbf', C=10000, epsilon=0.01, gamma=0.001)
svr_model_198_gs.fit(X_train_198, y_train_198)

# Save Model
svr_model_198_gs_filename = "../../04_Evaluation/00_Models/svr_model_198_gs.pkl"
with open(svr_model_198_gs_filename, 'wb') as file:
    pickle.dump(svr_model_198_gs, file)


# ----------- Store 897 -----------

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
              'gamma': [1, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

K = 5
scorer = make_scorer(mean_squared_error, greater_is_better=False)
grid_897 = GridSearchCV(SVR(epsilon=0.01), param_grid, cv=K, scoring=scorer)

# fitting the model for grid search
grid_897.fit(X_train_897, y_train_897)

# Print best hyperparameters
print('Grid Search Store 897')
print(grid_897.best_params_)
print(grid_897.best_estimator_)
print('----------------------------------')
# {'C': 10000, 'gamma': 0.001, 'kernel': 'rbf'}
# SVR(C=10000, epsilon=0.01, gamma=0.001)

# Fit Model
svr_model_897_gs = SVR(kernel='rbf', C=10000, epsilon=0.01, gamma=0.001)
svr_model_897_gs.fit(X_train_897, y_train_897)

# Save Model
svr_model_897_gs_filename = "../../04_Evaluation/00_Models/svr_model_897_gs.pkl"
with open(svr_model_897_gs_filename, 'wb') as file:
    pickle.dump(svr_model_897_gs, file)


# ----------------------------------------------------------------------------------


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Prediction
y_train_predicted_708_gs = svr_model_708_gs.predict(X_train_708)

# Evaluation
print("Training Evaluation - SVR Baseline")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708_gs))


def plot_stores(actual, predictions, pred_horizont, store_id):
    title = 'Sales Predictions Store ' + store_id + ' - ' + pred_horizont
    act = plt.plot(actual, color='turquoise', label='Actual')
    pred = plt.plot(predictions, color='darkgoldenrod', label='Predictions')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

y_train_predicted_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_train_708.index, y_train_predicted_708_gs)], columns=["date", "pred"])
y_train_predicted_708_gs = y_train_predicted_708_gs.set_index('date')

plot_stores(y_train_708, y_train_predicted_708_gs, 'Training', '708')
