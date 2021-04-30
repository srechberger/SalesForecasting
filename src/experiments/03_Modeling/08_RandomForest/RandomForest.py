from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Get Training Data
# All Stores
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y.pkl')


# Get Test Data
# All Stores
X_test = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_X.pkl')
y_test = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_y.pkl')
# Store 708
X_test_2W_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_2W.pkl')
X_test_1M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_1M.pkl')
X_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_3M.pkl')
y_test_2W_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_2W.pkl')
y_test_1M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_1M.pkl')
y_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_2W_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_2W.pkl')
X_test_1M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_1M.pkl')
X_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_3M.pkl')
y_test_2W_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_2W.pkl')
y_test_1M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_1M.pkl')
y_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_2W_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_2W.pkl')
X_test_1M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_1M.pkl')
X_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_3M.pkl')
y_test_2W_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_2W.pkl')
y_test_1M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_1M.pkl')
y_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_3M.pkl')

# Fit Model
rf_model = RandomForestRegressor(n_estimators=30)
rf_model.fit(X_train, y_train)

print("Regresion Model Score", ":", rf_model.score(X_train, y_train), ",",
      "Out of Sample Test Score", ":", rf_model.score(X_test, y_test))

# -------------------------- All Stores ----------------------------------------

# Prediction
y_train_predicted = rf_model.predict(X_train)
y_test_predicted = rf_model.predict(X_test)

# Evaluation
print("All Stores")
print("Training RMSE", ":", rmse(y_train, y_train_predicted),
      "Testing RMSE", ":", rmse(y_test, y_test_predicted))

# ----- ++++ separate training attempt ++++ -----

# Fit Model
rf_model_708 = RandomForestRegressor(n_estimators=30)
rf_model_708.fit(X_train, y_train)

print("Regresion Model Score", ":", rf_model_708.score(X_train_708, y_train_708), ",",
      "Out of Sample Test Score", ":", rf_model_708.score(X_test_3M_708, y_test_3M_708))

# Prediction
y_train_pre_708 = rf_model_708.predict(X_train_708)
y_test_pre_2W_708 = rf_model_708.predict(X_test_2W_708)
y_test_pre_1M_708 = rf_model_708.predict(X_test_1M_708)
y_test_pre_3M_708 = rf_model_708.predict(X_test_3M_708)

# Evaluation
print("Predictions with Single Store Information")
print("Training RMSE", ":", rmse(y_train_708, y_train_pre_708))
print("Testing 2W RMSE", ":", rmse(y_test_2W_708, y_test_pre_2W_708))
print("Testing 1M RMSE", ":", rmse(y_test_1M_708, y_test_pre_1M_708))
print("Testing 3M RMSE", ":", rmse(y_test_3M_708, y_test_pre_3M_708))

# -------------------------- Store 708 ------------------------------------------

# Prediction
y_test_pred_2W_708 = rf_model.predict(X_test_2W_708)
# Evaluation
print("Store 708 - 2 Weeks")
print("Testing RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708))

# Prediction
y_test_pred_1M_708 = rf_model.predict(X_test_1M_708)
# Evaluation
print("Store 708 - 1 Month")
print("Testing RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708))

# Prediction
y_test_pred_3M_708 = rf_model.predict(X_test_3M_708)
# Evaluation
print("Store 708 - 3 Months")
print("Testing RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708))


# -------------------------- Store 198 ------------------------------------------

# Prediction
y_test_pred_2W_198 = rf_model.predict(X_test_2W_198)
# Evaluation
print("Store 198 - 2 Weeks")
print("Testing RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198))

# Prediction
y_test_pred_1M_198 = rf_model.predict(X_test_1M_198)
# Evaluation
print("Store 198 - 1 Month")
print("Testing RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198))

# Prediction
y_test_pred_3M_198 = rf_model.predict(X_test_3M_198)
# Evaluation
print("Store 198 - 3 Months")
print("Testing RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198))


# -------------------------- Store 897 ------------------------------------------

# Prediction
y_test_pred_2W_897 = rf_model.predict(X_test_2W_897)
# Evaluation
print("Store 897 - 2 Weeks")
print("Testing RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897))

# Prediction
y_test_pred_1M_897 = rf_model.predict(X_test_1M_897)
# Evaluation
print("Store 897 - 1 Month")
print("Testing RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897))

# Prediction
y_test_pred_3M_897 = rf_model.predict(X_test_3M_897)
# Evaluation
print("Store 897 - 3 Months")
print("Testing RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897))
