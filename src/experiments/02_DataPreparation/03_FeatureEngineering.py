import numpy as np
import pandas as pd, datetime
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from time import time
import os
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import  ARIMA
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Get data
sales = pd.read_pickle('../../../data/rossmann/intermediate/01_SalesDataCleaned/sales.pkl')

# Transform sales column to float
sales['Sales'] = sales['Sales'].astype(float)

# ------------------------------------ FEATURE ENGINEERING -----------------------------------------

# Null values: CompetitionDistance 3
# TODO: imputation of 0 values
# print(sales['CompetitionDistance'].loc[(sales.CompetitionDistance == 0)].count())
# result --> 2642

# ----------------------------------------------------------------------------------------------------------

# Null values: PromoInterval 544
# TODO: transform information (split into separate cols)
# print(sales['PromoInterval'].loc[(sales.PromoInterval == '')].count())
# result --> 508031
# print --> ['' 'Jan,Apr,Jul,Oct' 'Feb,May,Aug,Nov' 'Mar,Jun,Sept,Dec']

# Add new column with month from date column
sales['Month'] = sales['DateCol'].dt.month
# Add new column with year from date column
sales['Year'] = sales['DateCol'].dt.year
# Add new column with day of month from date column
sales['DayOfMonth'] = sales['DateCol'].dt.day


# Defining a function to check if the PromotionInterval corresponds to the Month
def label_is_promo_month(row):
   if ((row['PromoInterval'] == 'Jan,Apr,Jul,Oct') and (row['Month'] in [1, 4, 7, 10])):
      return 1
   if ((row['PromoInterval'] == 'Feb,May,Aug,Nov') and (row['Month'] in [2, 5, 8, 11])):
      return 1
   if ((row['PromoInterval'] == 'Mar,Jun,Sept,Dec') and (row['Month'] in [3, 6, 9, 12])):
      return 1
   return 0


sales['isPromoMonth'] = sales.apply(lambda row: label_is_promo_month(row), axis=1)


# ----------------------------------------------------------------------------------------------------------

# TODO: transform information (split into separate cols) - OneHotEncoder
# print(train['StateHoliday'].unique()) --> ['n' 'a' 'b' 'c']

# One-Hot Encoding (nominal - categorical columns)
# Apply one-hot encoder to 'StateHoliday'
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_num = pd.DataFrame(OH_encoder.fit_transform(sales[['StateHoliday']]))

# One-hot encoding removed index --> reset of index
OH_cols_num.index = sales.index

# Removement of the categorical column from sales data
sales = sales.drop('StateHoliday', axis=1)

# Add one-hot encoded column to sales data
sales = pd.concat([sales, OH_cols_num], axis=1)

# -----------------------------------------------------------------------------------------

# TODO: Transform all objects to a numeric value (OneHotEncoder or LabelEncoder)
# Categorical variables
s = (sales.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
# ['StoreType', 'Assortment', 'PromoInterval']
# --> Drop PromoInterval (already transformed to isPromoMonth

# ------------------------------- EXPLORATORY DATA ANALYSIS (EDA) ----------------------------

## Trend Analysis
# Sales trend over the months and year
sns.factorplot(data=sales, x="Month", y="Sales",
               col='Promo',
               hue='Promo2',
               row="Year")
plt.show()

# Conclusion: seasonality exists

# Sales trend over days
sns.factorplot(data=sales, x="DayOfWeek", y="Sales", hue="Promo")
plt.show()

## Stationary Analysis


# Assigning store 6
store6 = sales[sales.Store == 25]['Sales']

# Rolling mean analysis (Stationary)

# Function to test the stationarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()

    print(roll_std)

    # Plotting rolling statistics:
    orig = plt.plot(timeseries.resample('W').mean(), color='blue', label='Original')
    mean = plt.plot(roll_mean.resample('W').mean(), color='red', label='Rolling Mean')
    std = plt.plot(roll_std.resample('W').mean(), color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.show(block=False)

# Testing stationarity of store type a
test_stationarity(store6)


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Get data of single stores
sales_store708 = sales.loc[sales.Store == 708]
sales_store198 = sales.loc[sales.Store == 198]
sales_store897 = sales.loc[sales.Store == 897]

# Store data for modeling tasks
sales.to_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales.pkl')
sales_store708.to_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales_store708.pkl')
sales_store198.to_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales_store198.pkl')
sales_store897.to_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales_store897.pkl')
