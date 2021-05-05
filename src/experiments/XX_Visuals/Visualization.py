import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Get Test Data
# Store 708
X_test_3M_708 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_3M.pkl')
y_test_3M_708 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_3M_198 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_3M.pkl')
y_test_3M_198 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_3M_897 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_3M.pkl')
y_test_3M_897 = pd.read_pickle(
    '../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_3M.pkl')

# 01_ETS

holt_winters_model_708_filename = "../04_Evaluation/00_Models/ets_holt_winters_model_708.pkl"
with open(holt_winters_model_708_filename, 'rb') as file:
    holt_winters_model_708 = pickle.load(file)

holt_winters_model_198_filename = "../04_Evaluation/00_Models/ets_holt_winters_model_198.pkl"
with open(holt_winters_model_198_filename, 'rb') as file:
    holt_winters_model_198 = pickle.load(file)

holt_winters_model_897_filename = "../04_Evaluation/00_Models/ets_holt_winters_model_897.pkl"
with open(holt_winters_model_897_filename, 'rb') as file:
    holt_winters_model_897 = pickle.load(file)

holt_winters_pred_708_3M = holt_winters_model_708.forecast(90)
holt_winters_pred_3M_708 = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, holt_winters_pred_708_3M)],
                                        columns=["date", "pred"])
holt_winters_pred_3M_708 = holt_winters_pred_3M_708.set_index('date')

holt_winters_pred_198_3M = holt_winters_model_198.forecast(90)
holt_winters_pred_3M_198 = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_198.index, holt_winters_pred_198_3M)],
                                        columns=["date", "pred"])
holt_winters_pred_3M_198 = holt_winters_pred_3M_198.set_index('date')

holt_winters_pred_897_3M = holt_winters_model_897.forecast(90)
holt_winters_pred_3M_897 = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, holt_winters_pred_897_3M)],
                                        columns=["date", "pred"])
holt_winters_pred_3M_897 = holt_winters_pred_3M_897.set_index('date')

# 02_SARIMA

model_708_filename = "../04_Evaluation/00_Models/sarima_model_708.pkl"
with open(model_708_filename, 'rb') as file:
    sarima_model_708 = pickle.load(file)

model_198_filename = "../04_Evaluation/00_Models/sarima_model_198.pkl"
with open(model_198_filename, 'rb') as file:
    sarima_model_198 = pickle.load(file)

model_897_filename = "../04_Evaluation/00_Models/sarima_model_897.pkl"
with open(model_897_filename, 'rb') as file:
    sarima_model_897 = pickle.load(file)

sarima_y_test_pred_3M_708 = sarima_model_708.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                            end=pd.to_datetime('2015-03-31'),
                                                            dynamic=False)
sarima_y_test_forecasted_3M_708 = sarima_y_test_pred_3M_708.predicted_mean

sarima_y_test_pred_3M_198 = sarima_model_198.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                            end=pd.to_datetime('2015-03-31'),
                                                            dynamic=False)
sarima_y_test_forecasted_3M_198 = sarima_y_test_pred_3M_198.predicted_mean

sarima_y_test_pred_3M_897 = sarima_model_897.get_prediction(start=pd.to_datetime('2015-01-01'),
                                                            end=pd.to_datetime('2015-03-31'),
                                                            dynamic=False)
sarima_y_test_forecasted_3M_897 = sarima_y_test_pred_3M_897.predicted_mean

# 03_SVR
model_708_gs_filename = "../04_Evaluation/00_Models/svr_model_708_gs.pkl"
with open(model_708_gs_filename, 'rb') as file:
    svr_model_708_gs = pickle.load(file)

model_198_gs_filename = "../04_Evaluation/00_Models/svr_model_198_gs.pkl"
with open(model_198_gs_filename, 'rb') as file:
    svr_model_198_gs = pickle.load(file)

model_897_gs_filename = "../04_Evaluation/00_Models/svr_model_897_gs.pkl"
with open(model_897_gs_filename, 'rb') as file:
    svr_model_897_gs = pickle.load(file)

svr_y_test_pred_3M_708_gs = svr_model_708_gs.predict(X_test_3M_708)
svr_y_test_pred_3M_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, svr_y_test_pred_3M_708_gs)],
                                         columns=["date", "pred"])
svr_y_test_pred_3M_708_gs = svr_y_test_pred_3M_708_gs.set_index('date')

svr_y_test_pred_3M_198_gs = svr_model_198_gs.predict(X_test_3M_198)
svr_y_test_pred_3M_198_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_198.index, svr_y_test_pred_3M_198_gs)],
                                         columns=["date", "pred"])
svr_y_test_pred_3M_198_gs = svr_y_test_pred_3M_198_gs.set_index('date')

svr_y_test_pred_3M_897_gs = svr_model_897_gs.predict(X_test_3M_897)
svr_y_test_pred_3M_897_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, svr_y_test_pred_3M_897_gs)],
                                         columns=["date", "pred"])
svr_y_test_pred_3M_897_gs = svr_y_test_pred_3M_897_gs.set_index('date')

# 04_KNN

model_708_gs_filename = "../04_Evaluation/00_Models/knn_model_708_bl.pkl"
with open(model_708_gs_filename, 'rb') as file:
    knn_model_708_gs = pickle.load(file)

model_198_gs_filename = "../04_Evaluation/00_Models/knn_model_198.pkl"
with open(model_198_gs_filename, 'rb') as file:
    knn_model_198_gs = pickle.load(file)

model_897_gs_filename = "../04_Evaluation/00_Models/knn_model_897.pkl"
with open(model_897_gs_filename, 'rb') as file:
    knn_model_897_gs = pickle.load(file)

knn_y_test_pred_3M_708_gs = knn_model_708_gs.predict(X_test_3M_708)
knn_y_test_pred_3M_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, knn_y_test_pred_3M_708_gs)],
                                         columns=["date", "pred"])
knn_y_test_pred_3M_708_gs = knn_y_test_pred_3M_708_gs.set_index('date')

knn_y_test_pred_3M_198_gs = knn_model_198_gs.predict(X_test_3M_198)
knn_y_test_pred_3M_198_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_198.index, knn_y_test_pred_3M_198_gs)],
                                         columns=["date", "pred"])
knn_y_test_pred_3M_198_gs = knn_y_test_pred_3M_198_gs.set_index('date')

knn_y_test_pred_3M_897_gs = knn_model_897_gs.predict(X_test_3M_897)
knn_y_test_pred_3M_897_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, knn_y_test_pred_3M_897_gs)],
                                         columns=["date", "pred"])
knn_y_test_pred_3M_897_gs = knn_y_test_pred_3M_897_gs.set_index('date')

# 07_XGB

model_all_filename = "../04_Evaluation/00_Models/xgb_model_all.pkl"
with open(model_all_filename, 'rb') as file:
    xgb_model_all = pickle.load(file)

model_198_gs_filename = "../04_Evaluation/00_Models/xgb_model_198_gs.pkl"
with open(model_198_gs_filename, 'rb') as file:
    xgb_model_198_gs = pickle.load(file)

model_897_gs_filename = "../04_Evaluation/00_Models/xgb_model_897_gs.pkl"
with open(model_897_gs_filename, 'rb') as file:
    xgb_model_897_gs = pickle.load(file)

xgb_y_test_pred_3M_708_gs = xgb_model_all.predict(X_test_3M_708)
xgb_y_test_pred_3M_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, xgb_y_test_pred_3M_708_gs)],
                                         columns=["date", "pred"])
xgb_y_test_pred_3M_708_gs = xgb_y_test_pred_3M_708_gs.set_index('date')

xgb_y_test_pred_3M_198_gs = xgb_model_198_gs.predict(X_test_3M_198)
xgb_y_test_pred_3M_198_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_198.index, xgb_y_test_pred_3M_198_gs)],
                                         columns=["date", "pred"])
xgb_y_test_pred_3M_198_gs = xgb_y_test_pred_3M_198_gs.set_index('date')

xgb_y_test_pred_3M_897_gs = xgb_model_897_gs.predict(X_test_3M_897)
xgb_y_test_pred_3M_897_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, xgb_y_test_pred_3M_897_gs)],
                                         columns=["date", "pred"])
xgb_y_test_pred_3M_897_gs = xgb_y_test_pred_3M_897_gs.set_index('date')

# 08_RF

model_all_filename = "../04_Evaluation/00_Models/rf_model_all.pkl"
with open(model_all_filename, 'rb') as file:
    rf_model_all = pickle.load(file)

model_198_gs_filename = "../04_Evaluation/00_Models/rf_model_198_gs.pkl"
with open(model_198_gs_filename, 'rb') as file:
    rf_model_198_gs = pickle.load(file)

model_897_gs_filename = "../04_Evaluation/00_Models/rf_model_897_gs.pkl"
with open(model_897_gs_filename, 'rb') as file:
    rf_model_897_gs = pickle.load(file)

rf_y_test_pred_3M_708_gs = rf_model_all.predict(X_test_3M_708)
rf_y_test_pred_3M_708_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_708.index, rf_y_test_pred_3M_708_gs)],
                                        columns=["date", "pred"])
rf_y_test_pred_3M_708_gs = rf_y_test_pred_3M_708_gs.set_index('date')

rf_y_test_pred_3M_198_gs = rf_model_198_gs.predict(X_test_3M_198)
rf_y_test_pred_3M_198_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_198.index, rf_y_test_pred_3M_198_gs)],
                                        columns=["date", "pred"])
rf_y_test_pred_3M_198_gs = rf_y_test_pred_3M_198_gs.set_index('date')

rf_y_test_pred_3M_897_gs = rf_model_897_gs.predict(X_test_3M_897)
rf_y_test_pred_3M_897_gs = pd.DataFrame([[x, y] for x, y in zip(y_test_3M_897.index, rf_y_test_pred_3M_897_gs)],
                                        columns=["date", "pred"])
rf_y_test_pred_3M_897_gs = rf_y_test_pred_3M_897_gs.set_index('date')

# -----------------------------------------------------------------------------------------------

title = 'Sales Predictions Store 708'
ets_708 = plt.plot(holt_winters_pred_3M_708, color='turquoise', label='ETS')
sarima_708 = plt.plot(sarima_y_test_forecasted_3M_708, color='darkgoldenrod', label='SARIMA')
svr_708 = plt.plot(svr_y_test_pred_3M_708_gs, color='indigo', label='SVR')
knn_708 = plt.plot(knn_y_test_pred_3M_708_gs, color='darkred', label='KNN')
xgb_708 = plt.plot(xgb_y_test_pred_3M_708_gs, color='lightgreen', label='XGB')
rf_708 = plt.plot(rf_y_test_pred_3M_708_gs, color='peru', label='RF')
act_708 = plt.plot(y_test_3M_708, color='black', label='Actual')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title(title)
plt.show()

title = 'Sales Predictions Store 198'
ets_198 = plt.plot(holt_winters_pred_3M_198, color='turquoise', label='ETS')
sarima_198 = plt.plot(sarima_y_test_forecasted_3M_198, color='darkgoldenrod', label='SARIMA')
svr_198 = plt.plot(svr_y_test_pred_3M_198_gs, color='indigo', label='SVR')
knn_198 = plt.plot(knn_y_test_pred_3M_198_gs, color='darkred', label='KNN')
xgb_198 = plt.plot(xgb_y_test_pred_3M_198_gs, color='lightgreen', label='XGB')
rf_198 = plt.plot(rf_y_test_pred_3M_198_gs, color='peru', label='RF')
act_198 = plt.plot(y_test_3M_198, color='black', label='Actual')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title(title)
plt.show()

title = 'Sales Predictions Store 897'
ets_897 = plt.plot(holt_winters_pred_3M_897, color='turquoise', label='ETS')
sarima_897 = plt.plot(sarima_y_test_forecasted_3M_897, color='darkgoldenrod', label='SARIMA')
svr_897 = plt.plot(svr_y_test_pred_3M_897_gs, color='indigo', label='SVR')
knn_897 = plt.plot(knn_y_test_pred_3M_897_gs, color='darkred', label='KNN')
xgb_897 = plt.plot(xgb_y_test_pred_3M_897_gs, color='lightgreen', label='XGB')
rf_897 = plt.plot(rf_y_test_pred_3M_897_gs, color='peru', label='RF')
act_897 = plt.plot(y_test_3M_897, color='black', label='Actual')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title(title)
plt.show()