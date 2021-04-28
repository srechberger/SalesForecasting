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
import warnings
warnings.filterwarnings('ignore')

# Exploratory Data Analysis (EDA)

# Get data
sales = pd.read_pickle('../../../data/rossmann/intermediate/sales.pkl')

# Transform sales to float
sales['Sales'] = sales['Sales'] * 1.0



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

u = store6.mean()
print(u)

var = store6.var()
print(var)

drift = u - (0.5 * var)
print(drift)

stdev = store6.std()
print(stdev)

# Standardabweichung und Varianz zu bestimmen, gehört zum kleinen Einmaleins der Betriebswelt.
# Während die Standardabweichung Unternehmen dabei hilft, Trends und Probleme wie hohe Kostenunterschiede zu erkennen,
# gibt die Varianzberechnung Aufschluss über die Streuung von Werten.
# Beide Parameter lassen sich durch Programme wie Excel schnell und leicht erheben.

