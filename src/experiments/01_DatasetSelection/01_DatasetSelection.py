# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Function Coefficient of Variation (CV = Variationskoeffizient)
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100


# 01 ROSSMANN (get and prepare data for analysis)
# https://www.kaggle.com/c/rossmann-store-sales
rossmann_path = "../../../data/rossmann/input/train.csv"
rossmann_data = pd.read_csv(rossmann_path)
# Parse date column from object to datetime
rossmann_data['Date'] = pd.to_datetime(rossmann_data['Date'], format="%Y-%m-%d")
# Calc CV
rossmann_sales = rossmann_data['Sales']
print('CV Rossmann: ' + str(cv(rossmann_sales)))


# 02 WALMART (get and prepare data for analysis)
# https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting
walmart_path = "../../../data/walmart/train.csv"
walmart_data = pd.read_csv(walmart_path)
# Parse date column from object to datetime
walmart_data['Date'] = pd.to_datetime(walmart_data['Date'], format="%Y-%m-%d")
# Rename Sales Column
walmart_data = walmart_data.rename(columns={'Weekly_Sales': 'Sales'}, inplace=False)
# Calc CV
walmart_sales = walmart_data['Sales']
print('CV Walmart: ' + str(cv(walmart_sales)))


# 03 Supermarket (get and prepare data for analysis)
# https://www.kaggle.com/aungpyaeap/supermarket-sales
supermarket_path = "../../../data/supermarket/supermarket_sales.csv"
supermarket_data = pd.read_csv(supermarket_path)
# Rename columns
supermarket_data = supermarket_data.rename(columns={'Total': 'Sales'}, inplace=False)
# Parse date column from object to datetime
supermarket_data['Date'] = pd.to_datetime(supermarket_data['Date'], format="%m/%d/%Y")
# Group Sales by Date
supermarket_sales = supermarket_data.groupby('Date').Sales.sum()
# Calc CV
print('CV Supermarket: ' + str(cv(supermarket_sales)))


# 04 Superstore (get and prepare data for analysis)
# https://www.kaggle.com/rohitsahoo/sales-forecasting
superstore_path = "../../../data/superstore/train.csv"
superstore_data = pd.read_csv(superstore_path)
# Rename columns
superstore_data = superstore_data.rename(columns={'Order Date': 'Date'}, inplace=False)
# Parse date column from object to datetime
superstore_data['Date'] = pd.to_datetime(superstore_data['Date'], format="%d/%m/%Y")
# Group Sales by Date
superstore_sales = superstore_data.groupby('Date').Sales.sum()
# Calc CV
print('CV Superstore: ' + str(cv(superstore_sales)))


# 05 TV-SALES (get and prepare data for analysis)
# https://www.kaggle.com/nomanvb/tv-sales-forecast?select=Date+and+model+wise+sale.csv
tv_path = "../../../data/tv-sales/tvsales.csv"
tv_data = pd.read_csv(tv_path)
# Rename columns
tv_data = tv_data.rename(columns={'Count': 'Sales'}, inplace=False)
# Parse date column from object to datetime
tv_data['Date'] = pd.to_datetime(tv_data['Date'], format="%d-%b-%y")
# Group Sales by Date
tv_sales = tv_data.groupby('Date').Sales.sum()
# Calc CV
print('CV TV-Sales: ' + str(cv(tv_sales)))


def plot_data(data, dataset_name):
    # get defined columns
    data = data.loc[:, ['Sales', 'Date']]
    # sort by date
    data = data.sort_values(by=['Date'])
    # groupby Date if multiple entries per date exist
    data = data.groupby('Date').Sales.sum()

    # get rolling (weekly) data
    weekly_mean = data.rolling(window=7).mean() # window size 7 steht für 7 Tage
    weekly_std = data.rolling(window=7).std()

    # print CV for weekly data
    print('CV rolling week ' + dataset_name + ' ' + str(cv(weekly_mean)))

    # set title
    title = 'Sales Figures ' + dataset_name

    # plot rolling statistics
    orig = plt.plot(data, color='turquoise', label='Original')
    mean = plt.plot(weekly_mean, color='darkgoldenrod', label='Weekly Rolling Mean')
    std = plt.plot(weekly_std, color='indigo', label='Weekly Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    plt.title(title)
    plt.show()

# plot selected stores
plot_data(rossmann_data, 'Rossmann')
plot_data(walmart_data, 'Walmart')
plot_data(supermarket_data, 'Supermarket')
plot_data(superstore_data, 'Superstore')
plot_data(tv_data, 'TV-Sales')

# data information rossmann
rossmann_store_path = "../../../data/rossmann/input/store.csv"
rossmann_store = pd.read_csv(rossmann_store_path)
rossmann = pd.merge(left=rossmann_data, right=rossmann_store, how='left', left_on='Store', right_on='Store')


# data information walmart
walmart_features_path = "../../../data/walmart/features.csv"
walmart_features = pd.read_csv(walmart_features_path)
walmart_features['Date'] = pd.to_datetime(walmart_features['Date'], format="%Y-%m-%d")
walmart = pd.merge(
    left=walmart_data,
    right=walmart_features,
    how='left',
    left_on=['Store', 'Date'],
    right_on=['Store', 'Date'])

# count columns of datasets
print('Number of columns Walmart: ', len(walmart.columns))
print('Number of columns Rossmann: ', len(rossmann.columns))
print('Number of columns Supermarket: ', len(supermarket_data.columns))
print('Number of columns Superstore: ', len(superstore_data.columns))
print('Number of columns TV: ', len(tv_data.columns))
