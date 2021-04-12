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

# set display options
pd.set_option('display.max_columns', 15)

# -----------------------------------------------------------------------------

### Step 1: Gather the data

# Filepaths
train_file_path = "../../data/rossmann/input/train.csv"
store_file_path = "../../data/rossmann/input/store.csv"
# test and submission not relevant for modeling
# test_file_path = "../../data/rossmann/input/test.csv"
# submission_file_path = "../../data/rossmann/input/sample_submission.csv"

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
train['Store'].loc[(train.Store <= 0)]
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
train['Sales'].loc[(train.Sales < 0)]
# print --> no negative values

# ---------- Customers (int64) ----------
# Customers - the number of customers on a given day

# Check negative values
train['Customers'].loc[(train.Customers < 0)]
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
store['Store'].loc[(train.Store <= 0)]
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

# Null values:
# CompetitionDistance            3


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

# Replace NaN values with 0
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)

# transform from float to int
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].astype('int64')

# check dtype again
store['CompetitionOpenSinceMonth'].dtype
# print --> int64

# Check values again
store['CompetitionOpenSinceMonth'].unique()
# print --> [ 9 11 12  4 10  8  0  3  6  5  1  2  7]

# Null values:
# CompetitionOpenSinceMonth    354


# ---------- CompetitionOpenSinceYear (float64) ----------
# gives the approximate year and month of the time the nearest competitor was opened

print(store['CompetitionOpenSinceMonth'].describe().apply(lambda x: format(x, 'f')))

# Null values:
# CompetitionOpenSinceYear     354


# ---------- Promo2 (int64) ----------
# Promo2 is a continuing and consecutive promotion for some stores:
# 0 = store is not participating, 1 = store is participating


# ---------- Promo2SinceWeek (float64) ----------
# describes the year and calendar week when the store started participating in Promo2

# Null values:
# Promo2SinceWeek              544


# ---------- Promo2SinceYear (float64) ----------
# describes the year and calendar week when the store started participating in Promo2

# Null values:
# Promo2SinceYear              544


# ---------- PromoInterval (object) ----------
# describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew.
# E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

# Null values:
# PromoInterval                544

# -----------------------------------------------------------------------------

# sort by multiple values
# train.sort_values(by=['Date', 'Store'])

# -----------------------------------------------------------------------------

### Step 3: Select a model

### Step 4: Train the model

### Step 5: Evaluate the model

### Step 6: Tune parameters

### Step 7: Get predictions
