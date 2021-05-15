from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import pandas as pd
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


# Get Test Data
# Store 708
X_test_2W_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_2W.pkl')
X_test_1M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_1M.pkl')
X_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_3M.pkl')
y_test_2W_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_2W.pkl')
y_test_1M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_1M.pkl')
y_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_2W_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_2W.pkl')
X_test_1M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_1M.pkl')
X_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_3M.pkl')
y_test_2W_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_2W.pkl')
y_test_1M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_1M.pkl')
y_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_2W_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_2W.pkl')
X_test_1M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_1M.pkl')
X_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_3M.pkl')
y_test_2W_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_2W.pkl')
y_test_1M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_1M.pkl')
y_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_3M.pkl')


# Load Baseline Models
model_708_bl_filename = "../00_Models/knn_model_708_bl.pkl"
with open(model_708_bl_filename, 'rb') as file:
    knn_model_708_bl = pickle.load(file)

model_198_bl_filename = "../00_Models/knn_model_198_bl.pkl"
with open(model_198_bl_filename, 'rb') as file:
    knn_model_198_bl = pickle.load(file)
    
model_897_bl_filename = "../00_Models/knn_model_897_bl.pkl"
with open(model_897_bl_filename, 'rb') as file:
    knn_model_897_bl = pickle.load(file)

# Load Grid Search Models
model_708_filename = "../00_Models/knn_model_708.pkl"
with open(model_708_filename, 'rb') as file:
    knn_model_708 = pickle.load(file)

model_198_filename = "../00_Models/knn_model_198.pkl"
with open(model_198_filename, 'rb') as file:
    knn_model_198 = pickle.load(file)

model_897_filename = "../00_Models/knn_model_897.pkl"
with open(model_897_filename, 'rb') as file:
    knn_model_897 = pickle.load(file)


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# -------------------------- Training Fit ----------------------------------------

# Prediction
y_train_predicted_708_bl = knn_model_708_bl.predict(X_train_708)
y_train_predicted_198_bl = knn_model_198_bl.predict(X_train_198)
y_train_predicted_897_bl = knn_model_897_bl.predict(X_train_897)

# Evaluation
print("Training Evaluation - KNN Baseline")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708_bl))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198_bl))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897_bl))

# Prediction
y_train_predicted_708 = knn_model_708.predict(X_train_708)
y_train_predicted_198 = knn_model_198.predict(X_train_198)
y_train_predicted_897 = knn_model_897.predict(X_train_897)

# Evaluation
print("Training Evaluation - KNN Grid Search")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897))


# -------------------------- Store 708 ------------------------------------------

# Prediction
y_test_pred_2W_708_bl = knn_model_708_bl.predict(X_test_2W_708)
y_test_pred_2W_708 = knn_model_708.predict(X_test_2W_708)
# Evaluation
print("Store 708 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708))

# Prediction
y_test_pred_1M_708_bl = knn_model_708_bl.predict(X_test_1M_708)
y_test_pred_1M_708 = knn_model_708.predict(X_test_1M_708)
# Evaluation
print("Store 708 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708))

# Prediction
y_test_pred_3M_708_bl = knn_model_708_bl.predict(X_test_3M_708)
y_test_pred_3M_708 = knn_model_708.predict(X_test_3M_708)
# Evaluation
print("Store 708 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708))

print("-----------------------------------------------------------")


# -------------------------- Store 198 ------------------------------------------

# Prediction
y_test_pred_2W_198_bl = knn_model_198_bl.predict(X_test_2W_198)
y_test_pred_2W_198 = knn_model_198.predict(X_test_2W_198)
# Evaluation
print("Store 198 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198))

# Prediction
y_test_pred_1M_198_bl = knn_model_198_bl.predict(X_test_1M_198)
y_test_pred_1M_198 = knn_model_198.predict(X_test_1M_198)
# Evaluation
print("Store 198 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198))

# Prediction
y_test_pred_3M_198_bl = knn_model_198_bl.predict(X_test_3M_198)
y_test_pred_3M_198 = knn_model_198.predict(X_test_3M_198)
# Evaluation
print("Store 198 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198))

print("-----------------------------------------------------------")


# -------------------------- Store 897 ------------------------------------------

# Prediction
y_test_pred_2W_897_bl = knn_model_897_bl.predict(X_test_2W_897)
y_test_pred_2W_897 = knn_model_897.predict(X_test_2W_897)
# Evaluation
print("Store 897 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897))

# Prediction
y_test_pred_1M_897_bl = knn_model_897_bl.predict(X_test_1M_897)
y_test_pred_1M_897 = knn_model_897.predict(X_test_1M_897)
# Evaluation
print("Store 897 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897))

# Prediction
y_test_pred_3M_897_bl = knn_model_897_bl.predict(X_test_3M_897)
y_test_pred_3M_897 = knn_model_897.predict(X_test_3M_897)
# Evaluation
print("Store 897 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897))

print("-----------------------------------------------------------")


# ---------------------------------- PLOT PREDICTIONS -----------------------------------

def plot_stores(actual, predictions, pred_horizont, store_id):
    title = 'Sales Predictions Store ' + store_id + ' - ' + pred_horizont
    act = plt.plot(actual, color='turquoise', label='Actual')
    pred = plt.plot(predictions, color='darkgoldenrod', label='Predictions')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


pred_3M_897 = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, y_test_pred_3M_897)], columns=["date", "pred"])
pred_3M_897 = pred_3M_897.set_index('date')

plot_stores(y_test_3M_897, pred_3M_897, '3 Months', '897')
