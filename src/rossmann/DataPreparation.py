import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 15)

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

### Step 2: Prepare the data

## train file

# Get datatypes of all cols
# print(train.dtypes)

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
# train['SchoolHoliday'].unique()
# print --> [1 0]

# -----------------------------------------------------------------------------

# Check dtypes again
# print(train.dtypes)
# --> OK

# show data metrics
# print(train.describe().apply(lambda s: s.apply(lambda x: format(x, 'g'))))
# --> OK

# -----------------------------------------------------------------------------




# -----------------------------------------------------------------------------

# sort by multiple values
# train.sort_values(by=['Date', 'Store'])

# -----------------------------------------------------------------------------

### Step 3: Select a model

### Step 4: Train the model

### Step 5: Evaluate the model

### Step 6: Tune parameters

### Step 7: Get predictions
