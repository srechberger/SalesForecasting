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


# import libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# -----------------------------------------------------------------------------

### Step 1: Gather the data

# Filepaths
train_file_path = "../../../data/rossmann/input/train.csv"
store_file_path = "../../../data/rossmann/input/store.csv"

# Load data
train = pd.read_csv(train_file_path)
store = pd.read_csv(store_file_path)

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
# Check negative sales
sales = train['Sales'].loc[(train.Sales < 0)]
# 0 --> no negative values

train['Sales'].describe().apply(lambda x: format(x, 'f'))
# count    1017209.000000
# mean        5773.818972
# std         3849.926175
# min            0.000000
# 25%         3727.000000
# 50%         5744.000000
# 75%         7856.000000
# max        41551.000000

# ---------- Promo (int64) ----------
# Check values
train['Promo'].unique()
# Result unique values --> [1 0]

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
# Possible Values:
# a = Public holiday, b = Easter holiday, c = Christmas, 0 = None
train['StateHoliday'].unique()
# Values --> ['0' 'a' 'b' 'c' 0]
# Convert object to string
train['StateHoliday'] = train['StateHoliday'].astype('str')
# Replace '0' to 'n'
train['StateHoliday'] = train['StateHoliday'].str.replace('0', 'n')
# Result Values --> ['n' 'a' 'b' 'c']


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

# Decision: drop all columns with at least 30 % missing values
# 'CompetitionOpenSinceMonth' and 'CompetitionOpenSinceYear' --> 31,75 %
# 'Promo2SinceWeek' and 'Promo2SinceYear' --> 48,79 %
drop_features = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
store = store.drop(drop_features, axis=1)

# Null values: PromoInterval 544
# Replace null values with empty string
store['PromoInterval'] = store['PromoInterval'].fillna('')

# Null values: CompetitionDistance 3
# Replace null values with median distance of stores
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# Exception: PromoInterval
# A PromoInterval can only exist, if Promo2 == 1
store_promo2 = store.loc[(store.Promo2 == 1)]
store_promo2_missing_values_count = store_promo2['PromoInterval'].isnull().sum()
# print(store_promo2_missing_values_count)
# result --> 0 (no missing values)
# PromoInterval will not be dropped


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
# Replace null values with median distance of stores
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

# Check values
distance_is_null = store['CompetitionDistance'].isnull().sum()
# print(distance_is_null)
# result --> 0 (no missing values)

# ---------- Promo2 (int64) ----------
# Promo2 is a continuing and consecutive promotion for some stores:
# 0 = store is not participating, 1 = store is participating

# Check values
store['Promo2'].unique()
# print --> [0 1]


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

## Join dataframes and final checks

# left join()
sales = pd.merge(left=train, right=store, how='left', left_on='Store', right_on='Store')

# Generate duplicate date column ('DateCol')
sales['DateCol'] = sales['Date']

# Sort data by Date
sales.sort_values(by='Date')

# Set Date as index
sales = sales.set_index('Date')

# Check null values after left join
sales_missing_values_count = sales.isnull().sum()
# print(sales_missing_values_count)
# --> OK (no missing values)


# -----------------------------------------------------------------------------

## Storage of joined dataframe

# Store data for feature engineering
sales.to_pickle('../../../data/rossmann/intermediate/01_SalesDataCleaned/sales.pkl')

# this exceeds GitHub's file size limit of 100.00 MB
# Commands for large files
# git lfs install
# git lfs track "*.pkl"
# see --> .gitattributes

