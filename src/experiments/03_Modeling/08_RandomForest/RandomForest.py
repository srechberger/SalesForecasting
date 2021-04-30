from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Get Data
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')
X_test = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_X.pkl')
y_test = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_y.pkl.pkl')

# Fit Model
rf = RandomForestRegressor(n_estimators=30)
rf_reg = rf.fit(X_train, y_train)

print("Regresion Model Score", ":", rf_reg.score(X_train, y_train), ",",
      "Out of Sample Test Score", ":", rf_reg.score(X_test, y_test))

# Prediction
y_train_predicted = rf_reg.predict(X_train)
y_test_predicted = rf_reg.predict(X_test)

# Evaluation
print("Training RMSE", ":", rmse(y_train, y_train_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))
