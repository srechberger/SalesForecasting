# Data Description
# You are provided with historical sales data for 1,115 Rossmann stores.
# The task is to forecast the "Sales" column for the test set.
# Note that some stores in the dataset were temporarily closed for refurbishment.

# Files
#
#     train.csv - historical data including Sales
#     test.csv - historical data excluding Sales
#     sample_submission.csv - a sample submission file in the correct format
#     store.csv - supplemental information about the stores

# Data fields
#
# Most of the fields are self-explanatory. The following are descriptions for those that aren't.
#
#     Id - an Id that represents a (Store, Date) duple within the test set
#     Store - a unique Id for each store
#     Sales - the turnover for any given day (this is what you are predicting)
#     Customers - the number of customers on a given day
#     Open - an indicator for whether the store was open: 0 = closed, 1 = open
#     StateHoliday - indicates a state holiday. Normally all stores, with few exceptions,
#           are closed on state holidays. Note that all schools are closed on public holidays and weekends.
#           a = public holiday, b = Easter holiday, c = Christmas, 0 = None
#     SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
#     StoreType - differentiates between 4 different store models: a, b, c, d
#     Assortment - describes an assortment level: a = basic, b = extra, c = extended
#     CompetitionDistance - distance in meters to the nearest competitor store
#     CompetitionOpenSince[Month/Year] -
#           gives the approximate year and month of the time the nearest competitor was opened
#     Promo - indicates whether a store is running a promo on that day
#     Promo2 - Promo2 is a continuing and consecutive promotion for some stores:
#           0 = store is not participating, 1 = store is participating
#     Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
#     PromoInterval - describes the consecutive intervals Promo2 is started,
#           naming the months the promotion is started anew.
#           E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November
#                 of any given year for that store

# -----------------------------------------------------------------------------

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# set display options
pd.set_option('display.max_columns', 20)

# -----------------------------------------------------------------------------

### Step 1: Gather the data

# Filepaths
train_file_path = "../../../data/rossmann/input/train.csv"
store_file_path = "../../../data/rossmann/input/store.csv"
# test and submission not relevant for modeling
# test_file_path = "../../../data/rossmann/input/test.csv"
# submission_file_path = "../../../data/rossmann/input/sample_submission.csv"

# Load data
train = pd.read_csv(train_file_path)
store = pd.read_csv(store_file_path)
# test and submission not relevant for modeling
# test = pd.read_csv(test_file_path)
# submission = pd.read_csv(submission_file_path)
# -----------------------------------------------------------------------------

### Step 2: Prepare the data

## train file

# Get datatypes of all cols
# print(train.dtypes)
# Store             int64
# DayOfWeek         int64
# Date             object
# Sales             int64
# Customers         int64
# Open              int64
# Promo             int64
# StateHoliday     object
# SchoolHoliday     int64

# Get the number of missing data points per column
missing_values_count = train.isnull().sum()
# print(missing_values_count)
# print --> all column sums = 0

# ---------- Store (int64) ----------
# Store - a unique Id for each store

# Check negative store IDs or ID = 0
storeIds = train['Store'].loc[(train.Store <= 0)]
# print --> no negative values or ID = 0

# ---------- DayOfWeek(int64) ----------
# Mo=1, Di=2, Mi=3, Do=4, Fr=5, Sa=6, So=7

# Check values
train['DayOfWeek'].unique()
# print --> [5 4 3 2 1 7 6]

# ---------- Date (object) ----------
# Sales date
# Parse date column from object to datetime
train['Date'] = pd.to_datetime(train['Date'], format="%Y-%m-%d")

# ---------- Sales (int64) ----------
# The turnover for any given day (this is what you are predicting)

# Check negative sales
sales = train['Sales'].loc[(train.Sales < 0)]
# print --> no negative values

# ---------- Customers (int64) ----------
# Customers - the number of customers on a given day

# Check negative values
customers = train['Customers'].loc[(train.Customers < 0)]
# print --> no negative values

# Check values
train['Customers'].describe().apply(lambda x: format(x, 'f'))
# print
# count    1017209.000000
# mean         633.145946
# std          464.411734
# min            0.000000
# 25%          405.000000
# 50%          609.000000
# 75%          837.000000
# max         7388.000000

# ---------- Open (int64) ----------
# an indicator for whether the store was open: 0 = closed, 1 = open

# Check values
train['Open'].unique()
# print --> [1 0]

# ---------- Promo (int64) ----------
# indicates whether a store is running a promo on that day

# Check values
train['Promo'].unique()
# print --> [1 0]

# ---------- StateHoliday (object) ----------
# StateHoliday - indicates a state holiday.
# Normally all stores, with few exceptions, are closed on state holidays.
# Note that all schools are closed on public holidays and weekends.
# a = public holiday, b = Easter holiday, c = Christmas, 0 = None
train['StateHoliday'].unique()
# print --> ['0' 'a' 'b' 'c' 0]
# Convert object to string
train['StateHoliday'] = train['StateHoliday'].astype('str')
# Replace '0' to 'n'
train['StateHoliday'] = train['StateHoliday'].str.replace('0', 'n')
# print(train['StateHoliday'].unique()) --> ['n' 'a' 'b' 'c']

# ---------- SchoolHoliday (int64) ----------
# indicates if the (Store, Date) was affected by the closure of public schools

# Check values
train['SchoolHoliday'].unique()
# print --> [1 0]

# -----------------------------------------------------------------------------
# --- Final proof of dataframe ---

# Check dtypes again
# print(train.dtypes)
# --> OK

# show data metrics
# print(train.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
# --> OK

# Get the number of missing data points per column
# missing_values_count = train.isnull().sum()
# print(missing_values_count)
# --> OK (no missing values)

# -----------------------------------------------------------------------------

## store file

# Get datatypes of all cols
# print(store.dtypes)
# Store                          int64
# StoreType                     object
# Assortment                    object
# CompetitionDistance          float64
# CompetitionOpenSinceMonth    float64
# CompetitionOpenSinceYear     float64
# Promo2                         int64
# Promo2SinceWeek              float64
# Promo2SinceYear              float64
# PromoInterval                 object

# Get the number of missing data points per column
store_missing_values_count = store.isnull().sum()
# print(store_missing_values_count)
# Store                          0
# StoreType                      0
# Assortment                     0
# CompetitionDistance            3
# CompetitionOpenSinceMonth    354
# CompetitionOpenSinceYear     354
# Promo2                         0
# Promo2SinceWeek              544
# Promo2SinceYear              544
# PromoInterval                544


# ---------- Store (int64) ----------
# a unique Id for each store

# Check negative store IDs or ID = 0
storeIds = store['Store'].loc[(train.Store <= 0)]
# print --> no negative values or ID = 0

# ---------- StoreType (object) ----------
# differentiates between 4 different store models: a, b, c, d

# Check values
store['StoreType'].unique()
# print --> ['c' 'a' 'd' 'b']

# ---------- Assortment (object) ----------
# describes an assortment level: a = basic, b = extra, c = extended

# Check values
store['Assortment'].unique()
# print --> ['a' 'c' 'b']

# ---------- CompetitionDistance (float64) ----------
#  distance in meters to the nearest competitor store

store['CompetitionDistance'].describe().apply(lambda x: format(x, 'f'))
# print
# count     1112.000000
# mean      5404.901079
# std       7663.174720
# min         20.000000
# 25%        717.500000
# 50%       2325.000000
# 75%       6882.500000
# max      75860.000000

# Check max distance 75860 (outlier?)
longDistances = store['CompetitionDistance'].loc[(store.CompetitionDistance > 40000)]
# print(longDistances)
# 109    46590.0
# 121    58260.0
# 452    75860.0
# 461    44320.0
# 523    40860.0
# 725    40540.0
# 746    45740.0
# 800    48330.0
# --> OK

# Null values: CompetitionDistance 3
# Replace NaN values with 0
store['CompetitionDistance'] = store['CompetitionDistance'].fillna(0)

# ---------- CompetitionOpenSinceMonth (float64) ----------
# gives the approximate year and month of the time the nearest competitor was opened

store['CompetitionOpenSinceMonth'].describe().apply(lambda x: format(x, 'f'))
# print
# count    761.000000
# mean       7.224704
# std        3.212348
# min        1.000000
# 25%        4.000000
# 50%        8.000000
# 75%       10.000000
# max       12.000000

# Check values
store['CompetitionOpenSinceMonth'].unique()
# print --> [ 9. 11. 12.  4. 10.  8. nan  3.  6.  5.  1.  2.  7.]

# Null values: CompetitionOpenSinceMonth 354
# Replace NaN values with 0
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)

# transform from float to int
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].astype('int64')

# check dtype again
datatype = store['CompetitionOpenSinceMonth'].dtype
# print --> int64

# Check values again
store['CompetitionOpenSinceMonth'].unique()
# print --> [ 9 11 12  4 10  8  0  3  6  5  1  2  7]

# ---------- CompetitionOpenSinceYear (float64) ----------
# gives the approximate year and month of the time the nearest competitor was opened

store['CompetitionOpenSinceYear'].describe().apply(lambda x: format(x, 'f'))
# print
# count     761.000000
# mean     2008.668857
# std         6.195983
# min      1900.000000
# 25%      2006.000000
# 50%      2010.000000
# 75%      2013.000000
# max      2015.000000

# Check values
store['CompetitionOpenSinceYear'].unique()
# print --> [2008. 2007. 2006. 2009. 2015. 2013. 2014. 2000. 2011.   nan 2010. 2005.
#            1999. 2003. 2012. 2004. 2002. 1961. 1995. 2001. 1990. 1994. 1900. 1998.]

# Null values: CompetitionOpenSinceYear 354
# Replace NaN values with 0
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0)

# transform from float to int
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].astype('int64')

# check dtype again
datatype = store['CompetitionOpenSinceYear'].dtype
# print --> int64

# Check values again
store['CompetitionOpenSinceYear'].unique()
# print --> [2008 2007 2006 2009 2015 2013 2014 2000 2011    0 2010 2005 1999 2003
#            2012 2004 2002 1961 1995 2001 1990 1994 1900 1998]

# ---------- Promo2 (int64) ----------
# Promo2 is a continuing and consecutive promotion for some stores:
# 0 = store is not participating, 1 = store is participating

# Check values
store['Promo2'].unique()
# print --> [0 1]

# ---------- Promo2SinceWeek (float64) ----------
# describes the year and calendar week when the store started participating in Promo2

store['Promo2SinceWeek'].describe().apply(lambda x: format(x, 'f'))
# print
# count    571.000000
# mean      23.595447
# std       14.141984
# min        1.000000
# 25%       13.000000
# 50%       22.000000
# 75%       37.000000
# max       50.000000

# Check values
store['Promo2SinceWeek'].unique()
# print --> [nan 13. 14.  1. 45. 40. 26. 22.  5.  6. 10. 31. 37.  9. 39. 27. 18. 35. 23. 48. 36. 50. 44. 49. 28.]

# Null values: Promo2SinceWeek 544
# Replace NaN values with 0
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)

# transform from float to int
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].astype('int64')

# check dtype again
datatype = store['Promo2SinceWeek'].dtype
# print(datatype)
# print --> int64

# Check values again
store['Promo2SinceWeek'].unique()
# print --> [ 0 13 14  1 45 40 26 22  5  6 10 31 37  9 39 27 18 35 23 48 36 50 44 49 28]

# ---------- Promo2SinceYear (float64) ----------
# describes the year and calendar week when the store started participating in Promo2

store['Promo2SinceYear'].describe().apply(lambda x: format(x, 'f'))
# print
# count     571.000000
# mean     2011.763573
# std         1.674935
# min      2009.000000
# 25%      2011.000000
# 50%      2012.000000
# 75%      2013.000000
# max      2015.000000

# Check values
store['Promo2SinceYear'].unique()
# print --> [  nan 2010. 2011. 2012. 2009. 2014. 2015. 2013.]

# Null values: Promo2SinceYear 544
# Replace NaN values with 0
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0)

# transform from float to int
store['Promo2SinceYear'] = store['Promo2SinceYear'].astype('int64')

# check dtype again
datatype = store['Promo2SinceYear'].dtype
# print(datatype)
# print --> int64

# Check values again
store['Promo2SinceYear'].unique()
# print --> [   0 2010 2011 2012 2009 2014 2015 2013]

# ---------- PromoInterval (object) ----------
# describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.
# E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

store['PromoInterval'].describe()
# print
# count                 571
# unique                  3
# top       Jan,Apr,Jul,Oct
# freq                  335

# Check values
store['PromoInterval'].unique()
# print --> [nan 'Jan,Apr,Jul,Oct' 'Feb,May,Aug,Nov' 'Mar,Jun,Sept,Dec']

# Null values: PromoInterval 544
# Replace NaN values with empty string
store['PromoInterval'] = store['PromoInterval'].fillna('')

# Check values again
store['PromoInterval'].unique()
# print --> ['' 'Jan,Apr,Jul,Oct' 'Feb,May,Aug,Nov' 'Mar,Jun,Sept,Dec']

# -----------------------------------------------------------------------------
# --- Final proof of dataframe ---

# Check dtypes again
# print(store.dtypes)
# --> OK

# show data metrics
# print(store.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
# --> OK

# Get the number of missing data points per column
store_missing_values_count = store.isnull().sum()
# print(store_missing_values_count)
# --> OK (no missing values)

# -----------------------------------------------------------------------------
## Join dataframes

# left join()
sales = pd.merge(left=train, right=store, how='left', left_on='Store', right_on='Store')

# Check null values after left join
sales_missing_values_count = sales.isnull().sum()
# print(sales_missing_values_count)
# --> OK (no missing values)

# Null values: CompetitionDistance 3
# TODO: imputation of 0 values
# print(sales['CompetitionDistance'].loc[(sales.CompetitionDistance == 0)].count())
# print --> 2642

# Null values: CompetitionOpenSinceMonth 354
# TODO: imputation of 0 values
# print(sales['CompetitionOpenSinceMonth'].loc[(sales.CompetitionOpenSinceMonth == 0)].count())
# print --> 323348

# Null values: CompetitionOpenSinceYear 354
# TODO: imputation of 0 values
# print(sales['CompetitionOpenSinceYear'].loc[(sales.CompetitionOpenSinceYear == 0)].count())
# print --> 323348

# Null values: Promo2SinceWeek 544
# TODO: imputation of 0 values
# print(sales['Promo2SinceWeek'].loc[(sales.Promo2SinceWeek == 0)].count())
# print --> 508031

# Null values: Promo2SinceYear 544
# TODO: imputation of 0 values
# print(sales['Promo2SinceYear'].loc[(sales.Promo2SinceYear == 0)].count())
# print --> 508031

# Null values: PromoInterval 544
# TODO: transform information (split into separate cols)
# print(sales['PromoInterval'].loc[(sales.PromoInterval == '')].count())
# print --> 508031
# print --> ['' 'Jan,Apr,Jul,Oct' 'Feb,May,Aug,Nov' 'Mar,Jun,Sept,Dec']

# Add new column with month of date column
sales['Month'] = sales['Date'].dt.month

# Defining a function to check if the PromotionInterval corresponds to the Month
def label_isPromoMonth (row):
   if ((row['PromoInterval'] == 'Jan,Apr,Jul,Oct') and (row['Month'] in [1, 4, 7, 10])):
      return 1
   if ((row['PromoInterval'] == 'Feb,May,Aug,Nov') and (row['Month'] in [2, 5, 8, 11])):
      return 1
   if ((row['PromoInterval'] == 'Mar,Jun,Sept,Dec') and (row['Month'] in [3, 6, 9, 12])):
      return 1
   return 0

sales['isPromoMonth'] = sales.apply (lambda row: label_isPromoMonth(row), axis=1)

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

# Store data for modeling tasks
sales.to_pickle('../../../data/rossmann/intermediate/sales.pkl')

# this exceeds GitHub's file size limit of 100.00 MB
# Commands for large files
# git lfs install
# git lfs track "*.pkl"
# see --> .gitattributes

# output = pd.DataFrame(sales)
# output.to_csv('../../../data/rossmann/output/sales.csv', index=False)

# --------------------------------------------------------------------------
## Plot data

# Sum of all stores
grouped_sales = sales.groupby('Date').Sales.sum()
# print(grouped_sales)

# Mean of all stores
# grouped_sales = sales.groupby('Date').Sales.mean()
# print(grouped_sales)

# Determine rolling statistics
rolmean = grouped_sales.rolling(window=30).mean() #window size 30 denotes 30 days
rolstd = grouped_sales.rolling(window=30).std()

# Plot rolling statistics for all stores
orig = plt.plot(grouped_sales, color='turquoise', label='Original')
mean = plt.plot(rolmean, color='darkgoldenrod', label='Rolling Mean')
std = plt.plot(rolstd, color='indigo', label='Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Summed Sales Figures of all Stores')
plt.show()

# -----------------------------------------------------------------------------------
### STORE 1

# Select data for StoreId == 1
sales_Store1 = sales.loc[sales.Store == 1]

# get defined columns
sales_Store1 = sales_Store1.loc[:, ['Sales', 'Date']]

# sort by date
sales_Store1 = sales_Store1.sort_values(by=['Date'])

# groupby Date if multiple entries per date exist
sales_Store1 = sales_Store1.groupby('Date').Sales.sum()

# Determine rolling statistics
rolmean_S1 = sales_Store1.rolling(window=30).mean() #window size 30 denotes 30 days
rolstd_S1 = sales_Store1.rolling(window=30).std()

# Plot rolling statistics for Store 1
orig_S1 = plt.plot(sales_Store1, color='turquoise', label='Original')
mean_S1 = plt.plot(rolmean_S1, color='darkgoldenrod', label='Rolling Mean')
std_S1 = plt.plot(rolstd_S1, color='indigo', label='Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures Store 1')
plt.show()

# -----------------------------------------------------------------------------------
### Several Stores

# Select data for different stores
sales_Store6 = sales.loc[sales.Store == 6]
sales_Store20 = sales.loc[sales.Store == 20]
sales_Store25 = sales.loc[sales.Store == 25]

# Store data for modeling tasks
sales_Store6 .to_pickle('../../../data/rossmann/intermediate/store6.pkl')
sales_Store20.to_pickle('../../../data/rossmann/intermediate/store20.pkl')
sales_Store25.to_pickle('../../../data/rossmann/intermediate/store25.pkl')

# get defined columns
sales_Store6 = sales_Store6.loc[:, ['Sales', 'Date']]
sales_Store20 = sales_Store20.loc[:, ['Sales', 'Date']]
sales_Store25 = sales_Store25.loc[:, ['Sales', 'Date']]

# sort by date
sales_Store6 = sales_Store6.sort_values(by=['Date'])
sales_Store20 = sales_Store20.sort_values(by=['Date'])
sales_Store25 = sales_Store25.sort_values(by=['Date'])

# groupby Date if multiple entries per date exist
sales_Store6 = sales_Store6.groupby('Date').Sales.sum()
sales_Store20 = sales_Store20.groupby('Date').Sales.sum()
sales_Store25 = sales_Store25.groupby('Date').Sales.sum()

# Determine rolling statistics
rolmean_S6 = sales_Store6.rolling(window=14).mean()
rolmean_S20 = sales_Store20.rolling(window=14).mean()
rolmean_S25 = sales_Store25.rolling(window=14).mean()

# Plot rolling statistics for Store 1
store6 = plt.plot(rolmean_S6, color='turquoise', label='Store 6')
store20 = plt.plot(rolmean_S20, color='darkgoldenrod', label='Store 20')
store25 = plt.plot(rolmean_S25, color='indigo', label='Store 25')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures of several Stores')
plt.show()
