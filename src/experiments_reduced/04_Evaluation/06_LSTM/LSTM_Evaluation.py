from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt


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


# -------------------------- Store 708 ------------------------------------------

# Predictions
y_test_pred_2W_708 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_708_2W.pkl')
y_test_pred_1M_708 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_708_1M.pkl')
y_test_pred_3M_708 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_708_3M.pkl')

print(y_test_2W_708)
print(y_test_pred_2W_708)

# Evaluation
print("Store 708 - LSTM")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708))
print("1 Month - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708))
print("3 Months - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708))

print("-----------------------------------------------------------")


# -------------------------- Store 198 ------------------------------------------

# Predictions
y_test_pred_2W_198 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_198_2W.pkl')
y_test_pred_1M_198 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_198_1M.pkl')
y_test_pred_3M_198 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_198_3M.pkl')

# Evaluation
print("Store 198 - LSTM")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198))
print("1 Month - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198))
print("3 Months - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198))

print("-----------------------------------------------------------")


# -------------------------- Store 897 ------------------------------------------

# Predictions
y_test_pred_2W_897 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_897_2W.pkl')
y_test_pred_1M_897 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_897_1M.pkl')
y_test_pred_3M_897 = pd.read_pickle(
    '../../03_Modeling/06_LSTM/predictions/pred_897_3M.pkl')

# Evaluation
print("Store 897 - LSTM")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897))
print("1 Month - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897))
print("3 Months - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897))

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


y_test_pred_3M_708 = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, y_test_pred_3M_708)], columns=["date", "pred"])
y_test_pred_3M_708 = y_test_pred_3M_708.set_index('date')

plot_stores(y_test_3M_708, y_test_pred_3M_708, '3 Months', '708')
