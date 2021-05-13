import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

# Get Training Data
# Store 708
y_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/train_store708_y_sarima.pkl')
# Store 198
y_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_y_sarima.pkl')
# Store 897
y_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_y_sarima.pkl')

# Get Test Data
# Store 708
y_test_2W_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_2W.pkl')
y_test_1M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_1M.pkl')
y_test_3M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_3M.pkl')

# Store 198
y_test_2W_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_2W.pkl')
y_test_1M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_1M.pkl')
y_test_3M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_3M.pkl')

# Store 897
y_test_2W_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_2W.pkl')
y_test_1M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_1M.pkl')
y_test_3M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_3M.pkl')


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Load Models
model_708_filename = "../00_Models/sarima_model_708.pkl"
with open(model_708_filename, 'rb') as file:
    sarima_model_708 = pickle.load(file)
    
model_198_filename = "../00_Models/sarima_model_198.pkl"
with open(model_198_filename, 'rb') as file:
    sarima_model_198 = pickle.load(file)
    
model_897_filename = "../00_Models/sarima_model_897.pkl"
with open(model_897_filename, 'rb') as file:
    sarima_model_897 = pickle.load(file)
    

# -------------------------- Training Fit ----------------------------------------

# Prediction
y_train_predicted_708 = sarima_model_708.get_prediction(start=pd.to_datetime('2014-11-01'),
                                                        end=pd.to_datetime('2014-12-31'),
                                                        dynamic=False)
y_train_forecasted_708 = y_train_predicted_708.predicted_mean

y_train_predicted_198 = sarima_model_198.get_prediction(start=pd.to_datetime('2014-11-01'),
                                                        end=pd.to_datetime('2014-12-31'),
                                                        dynamic=False)
y_train_forecasted_198 = y_train_predicted_198.predicted_mean

y_train_predicted_897 = sarima_model_897.get_prediction(start=pd.to_datetime('2014-11-01'),
                                                        end=pd.to_datetime('2014-12-31'),
                                                        dynamic=False)
y_train_forecasted_897 = y_train_predicted_897.predicted_mean

# Evaluation
print("Training Evaluation - SARIMA")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_forecasted_708))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_forecasted_198))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_forecasted_897))

print("-----------------------------------------------------------")

# -------------------------- Store 708 ------------------------------------------

# Prediction
y_test_pred_2W_708 = sarima_model_708.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-14'),
                                                     dynamic=False)
y_test_forecasted_2W_708 = y_test_pred_2W_708.predicted_mean

y_test_pred_1M_708 = sarima_model_708.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-31'),
                                                     dynamic=False)
y_test_forecasted_1M_708 = y_test_pred_1M_708.predicted_mean

y_test_pred_3M_708 = sarima_model_708.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-03-31'),
                                                     dynamic=False)
y_test_forecasted_3M_708 = y_test_pred_3M_708.predicted_mean

# Evaluation
print("Store 708 - 2 Weeks")
print("SARIMA - RMSE", ":", rmse(y_test_2W_708, y_test_forecasted_2W_708))

print("Store 708 - 1 Month")
print("SARIMA - RMSE", ":", rmse(y_test_1M_708, y_test_forecasted_1M_708))

print("Store 708 - 3 Months")
print("SARIMA - RMSE", ":", rmse(y_test_3M_708, y_test_forecasted_3M_708))


# -------------------------- Store 198 ------------------------------------------

# Prediction
y_test_pred_2W_198 = sarima_model_198.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-14'),
                                                     dynamic=False)
y_test_forecasted_2W_198 = y_test_pred_2W_198.predicted_mean

y_test_pred_1M_198 = sarima_model_198.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-31'),
                                                     dynamic=False)
y_test_forecasted_1M_198 = y_test_pred_1M_198.predicted_mean

y_test_pred_3M_198 = sarima_model_198.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-03-31'),
                                                     dynamic=False)
y_test_forecasted_3M_198 = y_test_pred_3M_198.predicted_mean

# Evaluation
print("Store 198 - 2 Weeks")
print("SARIMA - RMSE", ":", rmse(y_test_2W_198, y_test_forecasted_2W_198))

print("Store 198 - 1 Month")
print("SARIMA - RMSE", ":", rmse(y_test_1M_198, y_test_forecasted_1M_198))

print("Store 198 - 3 Months")
print("SARIMA - RMSE", ":", rmse(y_test_3M_198, y_test_forecasted_3M_198))


# -------------------------- Store 897 ------------------------------------------

# Prediction
y_test_pred_2W_897 = sarima_model_897.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-14'),
                                                     dynamic=False)
y_test_forecasted_2W_897 = y_test_pred_2W_897.predicted_mean

y_test_pred_1M_897 = sarima_model_897.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-01-31'),
                                                     dynamic=False)
y_test_forecasted_1M_897 = y_test_pred_1M_897.predicted_mean

y_test_pred_3M_897 = sarima_model_897.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                     end=pd.to_datetime('2015-03-31'),
                                                     dynamic=False)
y_test_forecasted_3M_897 = y_test_pred_3M_897.predicted_mean

# Evaluation
print("Store 897 - 2 Weeks")
print("SARIMA - RMSE", ":", rmse(y_test_2W_897, y_test_forecasted_2W_897))

print("Store 897 - 1 Month")
print("SARIMA - RMSE", ":", rmse(y_test_1M_897, y_test_forecasted_1M_897))

print("Store 897 - 3 Months")
print("SARIMA - RMSE", ":", rmse(y_test_3M_897, y_test_forecasted_3M_897))


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
    

plot_stores(y_test_3M_708, y_test_forecasted_3M_708, '3 Months', '708')
