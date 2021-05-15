from xgboost import XGBRegressor
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Get Training Data
# All Stores
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/01_AllStores/train_y.pkl')
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/train_store897_y.pkl')

# Get Test Data
# Store 708
X_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/test_store708_X_3M.pkl')
y_test_3M_708 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/02_Store708/test_store708_y_3M.pkl')
# Store 198
X_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/test_store198_X_3M.pkl')
y_test_3M_198 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/03_Store198/test_store198_y_3M.pkl')
# Store 897
X_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/test_store897_X_3M.pkl')
y_test_3M_897 = pd.read_pickle('../../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/04_Store897/test_store897_y_3M.pkl')

# Fit Model All Stores
xgb_model_all = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_all.fit(X_train, y_train)

# Save Model All Stores
model_all_filename = "../../04_Evaluation/00_Models/xgb_model_all.pkl"
with open(model_all_filename, 'wb') as file:
    pickle.dump(xgb_model_all, file)


# Fit Model Store 708
xgb_model_708 = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_708.fit(X_train_708, y_train_708)

# Save Model Store 708
model_708_filename = "../../04_Evaluation/00_Models/xgb_model_708.pkl"
with open(model_708_filename, 'wb') as file:
    pickle.dump(xgb_model_708, file)
    

# Fit Model Store 198
xgb_model_198 = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_198.fit(X_train_198, y_train_198)

# Save Model Store 198
model_198_filename = "../../04_Evaluation/00_Models/xgb_model_198.pkl"
with open(model_198_filename, 'wb') as file:
    pickle.dump(xgb_model_198, file)
    
    
# Fit Model Store 897
xgb_model_897 = XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model_897.fit(X_train_897, y_train_897)

# Save Model Store 897
model_897_filename = "../../04_Evaluation/00_Models/xgb_model_897.pkl"
with open(model_897_filename, 'wb') as file:
    pickle.dump(xgb_model_897, file)

# ----------------------------- Hyperparameter Tuning -----------------------------------

# XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()
    gsearch = GridSearchCV(estimator=xgb_model,
                           param_grid=param_tuning,
                           scoring='neg_mean_squared_error',  #MSE
                           cv=5,
                           n_jobs=-1,
                           verbose=1)
    gsearch.fit(X_train, y_train)
    return gsearch.best_params_


# ------------------ Store 708 -------------------------------------------------------------------------
print('Store 708: ', hyperParameterTuning(X_train_708, y_train_708))
# Store 708:  {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5,
#              'n_estimators': 500, 'objective': 'reg:squarederror', 'subsample': 0.7}

xgb_model_708_gs = XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.7,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        n_estimators=500,
        subsample=0.7)

xgb_model_708_gs.fit(
    X_train_708, 
    y_train_708, 
    early_stopping_rounds=5, 
    eval_set=[(X_test_3M_708, y_test_3M_708)], 
    verbose=False)

# Save Model Store 708 Grid Search
model_708_gs_filename = "../../04_Evaluation/00_Models/xgb_model_708_gs.pkl"
with open(model_708_gs_filename, 'wb') as file:
    pickle.dump(xgb_model_708_gs, file)

# ------------------ Store 198 -------------------------------------------------------------------------
print('Store 198: ', hyperParameterTuning(X_train_198, y_train_198))
# Store 198:  {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1,
#              'n_estimators': 100, 'objective': 'reg:squarederror', 'subsample': 0.7}

xgb_model_198_gs = XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.7,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        n_estimators=100,
        subsample=0.7)

xgb_model_198_gs.fit(
    X_train_198, 
    y_train_198, 
    early_stopping_rounds=5, 
    eval_set=[(X_test_3M_198, y_test_3M_198)], 
    verbose=False)

# Save Model Store 198 Grid Search
model_198_gs_filename = "../../04_Evaluation/00_Models/xgb_model_198_gs.pkl"
with open(model_198_gs_filename, 'wb') as file:
    pickle.dump(xgb_model_198_gs, file)

# ------------------ Store 897 -------------------------------------------------------------------------
print('Store 897: ', hyperParameterTuning(X_train_897, y_train_897))
# Store 897:  {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 3, 'min_child_weight': 5,
#              'n_estimators': 500, 'objective': 'reg:squarederror', 'subsample': 0.7}

xgb_model_897_gs = XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.5,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        n_estimators=500,
        subsample=0.7)

xgb_model_897_gs.fit(
    X_train_897, 
    y_train_897, 
    early_stopping_rounds=5, 
    eval_set=[(X_test_3M_897, y_test_3M_897)], 
    verbose=False)

# Save Model Store 897 Grid Search
model_897_gs_filename = "../../04_Evaluation/00_Models/xgb_model_897_gs.pkl"
with open(model_897_gs_filename, 'wb') as file:
    pickle.dump(xgb_model_897_gs, file)
