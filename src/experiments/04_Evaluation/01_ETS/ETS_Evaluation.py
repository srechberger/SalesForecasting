import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

# Get Training Data
# Store 708
y_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y_ets.pkl')
# Store 198
y_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y_ets.pkl')
# Store 897
y_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_y_ets.pkl')

# Get Test Data
# Store 708
y_test_2W_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_2W.pkl')
y_test_1M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_1M.pkl')
y_test_3M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_3M.pkl')

# Store 198
y_test_2W_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_2W.pkl')
y_test_1M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_1M.pkl')
y_test_3M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_3M.pkl')

# Store 897
y_test_2W_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_2W.pkl')
y_test_1M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_1M.pkl')
y_test_3M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_3M.pkl')


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# Load Models
ses_model_708_filename = "../00_Models/ets_ses_model_708.pkl"
with open(ses_model_708_filename, 'rb') as file:
    ses_model_708 = pickle.load(file)

holt_model_708_filename = "../00_Models/ets_holt_model_708.pkl"
with open(holt_model_708_filename, 'rb') as file:
    holt_model_708 = pickle.load(file)

holt_winters_model_708_filename = "../00_Models/ets_holt_winters_model_708.pkl"
with open(holt_winters_model_708_filename, 'rb') as file:
    holt_winters_model_708 = pickle.load(file)

ses_model_198_filename = "../00_Models/ets_ses_model_198.pkl"
with open(ses_model_198_filename, 'rb') as file:
    ses_model_198 = pickle.load(file)

holt_model_198_filename = "../00_Models/ets_holt_model_198.pkl"
with open(holt_model_198_filename, 'rb') as file:
    holt_model_198 = pickle.load(file)

holt_winters_model_198_filename = "../00_Models/ets_holt_winters_model_198.pkl"
with open(holt_winters_model_198_filename, 'rb') as file:
    holt_winters_model_198 = pickle.load(file)

ses_model_897_filename = "../00_Models/ets_ses_model_897.pkl"
with open(ses_model_897_filename, 'rb') as file:
    ses_model_897 = pickle.load(file)

holt_model_897_filename = "../00_Models/ets_holt_model_897.pkl"
with open(holt_model_897_filename, 'rb') as file:
    holt_model_897 = pickle.load(file)

holt_winters_model_897_filename = "../00_Models/ets_holt_winters_model_897.pkl"
with open(holt_winters_model_897_filename, 'rb') as file:
    holt_winters_model_897 = pickle.load(file)


# -------------------------- Training Fit ----------------------------------------

# Evaluation
print("Training Evaluation - SES")
print("Store 708 RMSE", ":", rmse(y_train_708, ses_model_708.fittedvalues))
print("Store 198 RMSE", ":", rmse(y_train_198, ses_model_198.fittedvalues))
print("Store 897 RMSE", ":", rmse(y_train_897, ses_model_897.fittedvalues))

print("Training Evaluation - Holt")
print("Store 708 RMSE", ":", rmse(y_train_708, holt_model_708.fittedvalues))
print("Store 198 RMSE", ":", rmse(y_train_198, holt_model_198.fittedvalues))
print("Store 897 RMSE", ":", rmse(y_train_897, holt_model_897.fittedvalues))

print("Training Evaluation - Holt-Winters")
print("Store 708 RMSE", ":", rmse(y_train_708, holt_winters_model_708.fittedvalues))
print("Store 198 RMSE", ":", rmse(y_train_198, holt_winters_model_198.fittedvalues))
print("Store 897 RMSE", ":", rmse(y_train_897, holt_winters_model_897.fittedvalues))

print("-----------------------------------------------------------")


# Predict SES
ses_pred_708_2W = ses_model_708.forecast(14)
ses_pred_708_1M = ses_model_708.forecast(31)
ses_pred_708_3M = ses_model_708.forecast(90)

ses_pred_198_2W = ses_model_198.forecast(14)
ses_pred_198_1M = ses_model_198.forecast(31)
ses_pred_198_3M = ses_model_198.forecast(90)

ses_pred_897_2W = ses_model_897.forecast(14)
ses_pred_897_1M = ses_model_897.forecast(31)
ses_pred_897_3M = ses_model_897.forecast(90)

# Predict Holt
holt_pred_708_2W = holt_model_708.forecast(14)
holt_pred_708_1M = holt_model_708.forecast(31)
holt_pred_708_3M = holt_model_708.forecast(90)

holt_pred_198_2W = holt_model_198.forecast(14)
holt_pred_198_1M = holt_model_198.forecast(31)
holt_pred_198_3M = holt_model_198.forecast(90)

holt_pred_897_2W = holt_model_897.forecast(14)
holt_pred_897_1M = holt_model_897.forecast(31)
holt_pred_897_3M = holt_model_897.forecast(90)

# Predict Holt-Winters
holt_winters_pred_708_2W = holt_winters_model_708.forecast(14)
holt_winters_pred_708_1M = holt_winters_model_708.forecast(31)
holt_winters_pred_708_3M = holt_winters_model_708.forecast(90)

holt_winters_pred_198_2W = holt_winters_model_198.forecast(14)
holt_winters_pred_198_1M = holt_winters_model_198.forecast(31)
holt_winters_pred_198_3M = holt_winters_model_198.forecast(90)

holt_winters_pred_897_2W = holt_winters_model_897.forecast(14)
holt_winters_pred_897_1M = holt_winters_model_897.forecast(31)
holt_winters_pred_897_3M = holt_winters_model_897.forecast(90)


# Evaluation SES

print("Store 708 - SES")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_708, ses_pred_708_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_708, ses_pred_708_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_708, ses_pred_708_3M))

print("Store 198 - SES")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_198, ses_pred_198_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_198, ses_pred_198_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_198, ses_pred_198_3M))

print("Store 897 - SES")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_897, ses_pred_897_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_897, ses_pred_897_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_897, ses_pred_897_3M))

print("-----------------------------------------------------------")


# Evaluation Holt

print("Store 708 - holt")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_708, holt_pred_708_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_708, holt_pred_708_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_708, holt_pred_708_3M))

print("Store 198 - holt")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_198, holt_pred_198_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_198, holt_pred_198_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_198, holt_pred_198_3M))

print("Store 897 - holt")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_897, holt_pred_897_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_897, holt_pred_897_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_897, holt_pred_897_3M))

print("-----------------------------------------------------------")


# Evaluation Holt-Winters

print("Store 708 - holt_winters")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_708, holt_winters_pred_708_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_708, holt_winters_pred_708_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_708, holt_winters_pred_708_3M))

print("Store 198 - holt_winters")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_198, holt_winters_pred_198_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_198, holt_winters_pred_198_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_198, holt_winters_pred_198_3M))

print("Store 897 - holt_winters")
print("2 Weeks - RMSE", ":", rmse(y_test_2W_897, holt_winters_pred_897_2W))
print("1 Month - RMSE", ":", rmse(y_test_1M_897, holt_winters_pred_897_1M))
print("3 Months - RMSE", ":", rmse(y_test_3M_897, holt_winters_pred_897_3M))

print("-----------------------------------------------------------")


# ---------------------------------- PLOT PREDICTIONS -----------------------------------

def plot_stores(historic, fitted, actual, predictions, pred_horizont, store_id, model):
    title = 'Sales Predictions Store ' + store_id + ' - ' + pred_horizont + ' - ' + model
    hist_act = plt.plot(historic, color='paleturquoise', label='Actual Historic')
    fit_val = plt.plot(fitted, color='goldenrod', label='Fitted Values')
    act = plt.plot(actual, color='turquoise', label='Actual')
    pred = plt.plot(predictions, color='darkgoldenrod', label='Predictions')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


ses_pred_2W_708 = pd.DataFrame([[x, y] for x, y in zip(y_test_2W_708.index, ses_pred_708_2W)], columns=["date", "pred"])
ses_pred_2W_708 = ses_pred_2W_708.set_index('date')
plot_stores(y_train_708, ses_model_708.fittedvalues, y_test_2W_708, ses_pred_2W_708, '2 Weeks', '708', 'SES')

holt_pred_2W_708 = pd.DataFrame([[x, y] for x, y in zip(y_test_2W_708.index, holt_pred_708_2W)],
                                columns=["date", "pred"])
holt_pred_2W_708 = holt_pred_2W_708.set_index('date')
plot_stores(y_train_708, holt_model_708.fittedvalues, y_test_2W_708, holt_pred_2W_708,
            '2 Weeks', '708', 'holt')

holt_winters_pred_2W_708 = pd.DataFrame([[x, y] for x, y in zip(y_test_2W_708.index, holt_winters_pred_708_2W)],
                                        columns=["date", "pred"])
holt_winters_pred_2W_708 = holt_winters_pred_2W_708.set_index('date')
plot_stores(y_train_708, holt_winters_model_708.fittedvalues, y_test_2W_708, holt_winters_pred_2W_708,
            '2 Weeks', '708', 'holt winters')

