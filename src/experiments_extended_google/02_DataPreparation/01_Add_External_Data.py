import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Define Time Intervals
date_train_from = datetime.datetime(2013, 1, 1)
date_test_3M = datetime.datetime(2015, 3, 31)

# Get sales data
sales = pd.read_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales.pkl')

# ----------------------- Google Trend Data --------------------------------------------------------

# Get external google trend data
google_path = "../../../data/rossmann/input/google-trends/Rossmann_DE.csv"
google = pd.read_csv(google_path)

# Transform Week raw data
google['StartDate'] = google['Week'].str[:10]
google = google.drop('Week', axis=1)

# Parse date columns from object to datetime
google['StartDate'] = pd.to_datetime(google['StartDate'], format="%Y-%m-%d")

# Transform data from weekly to daily
google = google.set_index('StartDate').resample('D').ffill()

# Reduce google trend and sales data to relevant time slot
sales = sales.loc[(sales.index >= date_train_from) & (sales.index <= date_test_3M)]
google = google.loc[(google.index >= date_train_from) & (google.index <= date_test_3M)]

# Rename rossmann col
google.columns = ['GoogleTrend']

# Join Sales with GoogleTrend
sales = pd.merge(left=sales, right=google, how='left', left_on=sales.index, right_on=google.index)
sales = sales.rename(mapper={'key_0': 'Date'}, axis='columns')
sales = sales.set_index('Date')


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Store data for analysis
sales.to_pickle('../../../data/rossmann/intermediate/05_SalesModelingExtendedGoogle/sales_extended.pkl')
