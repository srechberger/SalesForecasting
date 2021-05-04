from xgboost import XGBRegressor
import pandas as pd
import pickle


# Get Training Data
# All Stores
X_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
y_train = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')
# Store 708
X_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_X.pkl')
y_train_708 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y.pkl')
# Store 198
X_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_X.pkl')
y_train_198 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y.pkl')
# Store 897
X_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_X.pkl')
y_train_897 = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_y.pkl')


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



