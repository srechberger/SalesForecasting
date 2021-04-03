import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import pickle
plt.rcParams['figure.figsize'] = (12.0, 10.0)
from sklearn.tree import DecisionTreeRegressor

# save filepath to variable for easier access
rossmann_file_path = "dataset/train.csv"
# read the data and store data in DataFrame titled melbourne_data

# types = {'StateHoliday': np.dtype(str)}
rossmann_data = pd.read_csv(rossmann_file_path, parse_dates=[2], nrows=70000) #,dtype=types
# rossmann_data = pd.read_csv(rossmann_file_path, low_memory=False)
# store = pd.read_csv("store.csv")



# print a summary of the data in Melbourne data
rossmann_data.describe()

# show data metrics
print(rossmann_data.describe())

# show data columns
print(rossmann_data.columns)

# dropna drops missing values (think of na as "not available")
rossmann_data = rossmann_data.dropna(axis=0)

# y = prediction target (endogene Variable)
y = rossmann_data.Sales

# choosing features for prediction (exogene Variablen)
rossmann_features = ['Promo', 'StateHoliday', 'SchoolHoliday']

# assign features to X
X = rossmann_data[rossmann_features]

# show data metrics for selected features
print(X.describe())

# show example data for first 5 rows
print(X.head())

##### Build first prediction model

# Define model. Specify a number for random_state to ensure same results each run
rossmann_model = DecisionTreeRegressor(random_state=1)

# Fit model
rossmann_model.fit(X, y)

# Prediction
print("Making predictions for the 5 rows:")
print(X.head())
print("The predictions are")
print(rossmann_model.predict(X.head()))


