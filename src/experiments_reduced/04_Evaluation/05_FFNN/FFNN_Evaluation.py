import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Get Training Data
# Store 708
X_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_y.pkl')

# Get Test Data
# Store 708
X_test_2W_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_X_2W.pkl')
X_test_1M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_X_1M.pkl')
X_test_3M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_X_3M.pkl')
y_test_2W_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_2W.pkl')
y_test_1M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_1M.pkl')
y_test_3M_708 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_2W_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_X_2W.pkl')
X_test_1M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_X_1M.pkl')
X_test_3M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_X_3M.pkl')
y_test_2W_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_2W.pkl')
y_test_1M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_1M.pkl')
y_test_3M_198 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_2W_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_X_2W.pkl')
X_test_1M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_X_1M.pkl')
X_test_3M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_X_3M.pkl')
y_test_2W_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_2W.pkl')
y_test_1M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_1M.pkl')
y_test_3M_897 = pd.read_pickle(
    '../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/test_store897_y_3M.pkl')

# Transform data (X_*) - drop features + min_max_scaler
# Drop not important features
features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 'DayOfMonth', 'IsPromoMonth']
X_train_708 = X_train_708.drop(features, axis=1)
X_test_2W_708 = X_test_2W_708.drop(features, axis=1)
X_test_1M_708 = X_test_1M_708.drop(features, axis=1)
X_test_3M_708 = X_test_3M_708.drop(features, axis=1)
X_train_198 = X_train_198.drop(features, axis=1)
X_test_2W_198 = X_test_2W_198.drop(features, axis=1)
X_test_1M_198 = X_test_1M_198.drop(features, axis=1)
X_test_3M_198 = X_test_3M_198.drop(features, axis=1)
X_train_897 = X_train_897.drop(features, axis=1)
X_test_2W_897 = X_test_2W_897.drop(features, axis=1)
X_test_1M_897 = X_test_1M_897.drop(features, axis=1)
X_test_3M_897 = X_test_3M_897.drop(features, axis=1)


# Scale values (performance relevant)
scaler = MinMaxScaler()
X_train_708[X_train_708.columns] = scaler.fit_transform(X_train_708[X_train_708.columns])
X_test_2W_708[X_test_2W_708.columns] = scaler.transform(X_test_2W_708[X_test_2W_708.columns])
X_test_1M_708[X_test_1M_708.columns] = scaler.transform(X_test_1M_708[X_test_1M_708.columns])
X_test_3M_708[X_test_3M_708.columns] = scaler.transform(X_test_3M_708[X_test_3M_708.columns])
X_train_198[X_train_198.columns] = scaler.fit_transform(X_train_198[X_train_198.columns])
X_test_2W_198[X_test_2W_198.columns] = scaler.transform(X_test_2W_198[X_test_2W_198.columns])
X_test_1M_198[X_test_1M_198.columns] = scaler.transform(X_test_1M_198[X_test_1M_198.columns])
X_test_3M_198[X_test_3M_198.columns] = scaler.transform(X_test_3M_198[X_test_3M_198.columns])
X_train_897[X_train_897.columns] = scaler.fit_transform(X_train_897[X_train_897.columns])
X_test_2W_897[X_test_2W_897.columns] = scaler.transform(X_test_2W_897[X_test_2W_897.columns])
X_test_1M_897[X_test_1M_897.columns] = scaler.transform(X_test_1M_897[X_test_1M_897.columns])
X_test_3M_897[X_test_3M_897.columns] = scaler.transform(X_test_3M_897[X_test_3M_897.columns])

# Load Baseline Models
ffnn_model_708_bl = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_708_bl')
ffnn_model_198_bl = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_198_bl')
ffnn_model_897_bl = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_897_bl')

# Load Grid Search Models
ffnn_model_708_gs = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_708_gs')
ffnn_model_198_gs = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_198_gs')
ffnn_model_897_gs = tf.keras.models.load_model('../../04_Evaluation/00_Models/ffnn_model_897_gs')


# Define Function for RMSE
def rmse(x, y):
    return sqrt(mean_squared_error(x, y))


# -------------------------- Training Fit ----------------------------------------

# Prediction
y_train_predicted_708_bl = ffnn_model_708_bl.predict(X_train_708)
y_train_predicted_198_bl = ffnn_model_198_bl.predict(X_train_198)
y_train_predicted_897_bl = ffnn_model_897_bl.predict(X_train_897)

# Evaluation
print("Training Evaluation - ffnn Baseline")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708_bl))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198_bl))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897_bl))

# Prediction
y_train_predicted_708_gs = ffnn_model_708_gs.predict(X_train_708)
y_train_predicted_198_gs = ffnn_model_198_gs.predict(X_train_198)
y_train_predicted_897_gs = ffnn_model_897_gs.predict(X_train_897)

# Evaluation
print("Training Evaluation - ffnn Grid Search")
print("Store 708 RMSE", ":", rmse(y_train_708, y_train_predicted_708_gs))
print("Store 198 RMSE", ":", rmse(y_train_198, y_train_predicted_198_gs))
print("Store 897 RMSE", ":", rmse(y_train_897, y_train_predicted_897_gs))

# -------------------------- Store 708 ------------------------------------------

# Prediction
y_test_pred_2W_708_bl = ffnn_model_708_bl.predict(X_test_2W_708)
y_test_pred_2W_708_gs = ffnn_model_708_gs.predict(X_test_2W_708)
# Evaluation
print("Store 708 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_708, y_test_pred_2W_708_gs))

# Prediction
y_test_pred_1M_708_bl = ffnn_model_708_bl.predict(X_test_1M_708)
y_test_pred_1M_708_gs = ffnn_model_708_gs.predict(X_test_1M_708)
# Evaluation
print("Store 708 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_708, y_test_pred_1M_708_gs))

# Prediction
y_test_pred_3M_708_bl = ffnn_model_708_bl.predict(X_test_3M_708)
y_test_pred_3M_708_gs = ffnn_model_708_gs.predict(X_test_3M_708)
# Evaluation
print("Store 708 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_708, y_test_pred_3M_708_gs))

print("-----------------------------------------------------------")

# -------------------------- Store 198 ------------------------------------------

# Prediction
y_test_pred_2W_198_bl = ffnn_model_198_bl.predict(X_test_2W_198)
y_test_pred_2W_198_gs = ffnn_model_198_gs.predict(X_test_2W_198)
# Evaluation
print("Store 198 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_198, y_test_pred_2W_198_gs))

# Prediction
y_test_pred_1M_198_bl = ffnn_model_198_bl.predict(X_test_1M_198)
y_test_pred_1M_198_gs = ffnn_model_198_gs.predict(X_test_1M_198)
# Evaluation
print("Store 198 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_198, y_test_pred_1M_198_gs))

# Prediction
y_test_pred_3M_198_bl = ffnn_model_198_bl.predict(X_test_3M_198)
y_test_pred_3M_198_gs = ffnn_model_198_gs.predict(X_test_3M_198)
# Evaluation
print("Store 198 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_198, y_test_pred_3M_198_gs))

print("-----------------------------------------------------------")

# -------------------------- Store 897 ------------------------------------------

# Prediction
y_test_pred_2W_897_bl = ffnn_model_897_bl.predict(X_test_2W_897)
y_test_pred_2W_897_gs = ffnn_model_897_gs.predict(X_test_2W_897)
# Evaluation
print("Store 897 - 2 Weeks")
print("Baseline - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_2W_897, y_test_pred_2W_897_gs))

# Prediction
y_test_pred_1M_897_bl = ffnn_model_897_bl.predict(X_test_1M_897)
y_test_pred_1M_897_gs = ffnn_model_897_gs.predict(X_test_1M_897)
# Evaluation
print("Store 897 - 1 Month")
print("Baseline - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_1M_897, y_test_pred_1M_897_gs))

# Prediction
y_test_pred_3M_897_bl = ffnn_model_897_bl.predict(X_test_3M_897)
y_test_pred_3M_897_gs = ffnn_model_897_gs.predict(X_test_3M_897)
# Evaluation
print("Store 897 - 3 Months")
print("Baseline - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897_bl))
print("Grid Search - RMSE", ":", rmse(y_test_3M_897, y_test_pred_3M_897_gs))

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


y_test_pred_3M_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, y_test_pred_3M_708_gs)], columns=["date", "pred"])
y_test_pred_3M_708_gs = y_test_pred_3M_708_gs.set_index('date')

plot_stores(y_test_3M_708, y_test_pred_3M_708_gs, '3 Months', '708')
