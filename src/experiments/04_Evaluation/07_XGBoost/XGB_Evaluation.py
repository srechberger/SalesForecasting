from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import pandas as pd


# Get Training Data
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_y.pkl')


# Get Test Data
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


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Load Models
model_all_filename = "../00_Models/xgb_model_all.pkl"
with open(model_all_filename, 'rb') as file:
    xgb_model_all = pickle.load(file)

model_708_filename = "../00_Models/xgb_model_708.pkl"
with open(model_708_filename, 'rb') as file:
    xgb_model_708 = pickle.load(file)

model_198_filename = "../00_Models/xgb_model_198.pkl"
with open(model_198_filename, 'rb') as file:
    xgb_model_198 = pickle.load(file)

model_897_filename = "../00_Models/xgb_model_897.pkl"
with open(model_897_filename, 'rb') as file:
    xgb_model_897 = pickle.load(file)


# -------------------------- Training Fit ----------------------------------------

# Prediction
y_train_predicted_708_all = xgb_model_all.predict(X_train_708)
y_train_predicted_198_all = xgb_model_all.predict(X_train_198)
y_train_predicted_897_all = xgb_model_all.predict(X_train_897)

# Evaluation
print("Training Evaluation - All Stores Model")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708_all))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198_all))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897_all))

# Prediction
y_train_predicted_708 = xgb_model_708.predict(X_train_708)
y_train_predicted_198 = xgb_model_198.predict(X_train_198)
y_train_predicted_897 = xgb_model_897.predict(X_train_897)

# Evaluation
print("Training Evaluation - Single Store Model")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897))

print("-----------------------------------------------------------")


# -------------------------- Store 708 ------------------------------------------

# Prediction
y_test_pred_2W_708_all = xgb_model_all.predict(X_test_2W_708)
y_test_pred_2W_708 = xgb_model_708.predict(X_test_2W_708)
# Evaluation
print("Store 708 - 2 Weeks")
print("All Stores Model - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708_all))
print("Single Store Model - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708))

# Prediction
y_test_pred_1M_708_all = xgb_model_all.predict(X_test_1M_708)
y_test_pred_1M_708 = xgb_model_708.predict(X_test_1M_708)
# Evaluation
print("Store 708 - 1 Month")
print("All Stores Model - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708_all))
print("Single Store Model - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708))

# Prediction
y_test_pred_3M_708_all = xgb_model_all.predict(X_test_3M_708)
y_test_pred_3M_708 = xgb_model_708.predict(X_test_3M_708)
# Evaluation
print("Store 708 - 3 Months")
print("All Stores Model - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708_all))
print("Single Store Model - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708))

print("-----------------------------------------------------------")


# -------------------------- Store 198 ------------------------------------------

# Prediction
y_test_pred_2W_198_all = xgb_model_all.predict(X_test_2W_198)
y_test_pred_2W_198 = xgb_model_198.predict(X_test_2W_198)
# Evaluation
print("Store 198 - 2 Weeks")
print("All Stores Model - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198_all))
print("Single Store Model - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198))

# Prediction
y_test_pred_1M_198_all = xgb_model_all.predict(X_test_1M_198)
y_test_pred_1M_198 = xgb_model_198.predict(X_test_1M_198)
# Evaluation
print("Store 198 - 1 Month")
print("All Stores Model - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198_all))
print("Single Store Model - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198))

# Prediction
y_test_pred_3M_198_all = xgb_model_all.predict(X_test_3M_198)
y_test_pred_3M_198 = xgb_model_198.predict(X_test_3M_198)
# Evaluation
print("Store 198 - 3 Months")
print("All Stores Model - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198_all))
print("Single Store Model - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198))

print("-----------------------------------------------------------")


# -------------------------- Store 897 ------------------------------------------

# Prediction
y_test_pred_2W_897_all = xgb_model_all.predict(X_test_2W_897)
y_test_pred_2W_897 = xgb_model_897.predict(X_test_2W_897)
# Evaluation
print("Store 897 - 2 Weeks")
print("All Stores Model - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897_all))
print("Single Store Model - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897))

# Prediction
y_test_pred_1M_897_all = xgb_model_all.predict(X_test_1M_897)
y_test_pred_1M_897 = xgb_model_897.predict(X_test_1M_897)
# Evaluation
print("Store 897 - 1 Month")
print("All Stores Model - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897_all))
print("Single Store Model - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897))

# Prediction
y_test_pred_3M_897_all = xgb_model_all.predict(X_test_3M_897)
y_test_pred_3M_897 = xgb_model_897.predict(X_test_3M_897)
# Evaluation
print("Store 897 - 3 Months")
print("All Stores Model - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897_all))
print("Single Store Model - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897))
