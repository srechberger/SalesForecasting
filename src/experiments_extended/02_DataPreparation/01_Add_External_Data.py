import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
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


# ----------------------- Weather Data ---------------------------------------------------------------

# Get external weather data
weather_path = "../../../data/rossmann/input/weather/thueringen_weather.csv"
weather = pd.read_csv(weather_path, sep=";")

# Parse date columns from object to datetime
weather['Date'] = pd.to_datetime(weather['Date'], format="%Y-%m-%d")

# Set date index
weather = weather.set_index('Date')

# Select Mean Temperature + Events
weather = weather.loc[:, ['Mean_TemperatureC', 'Events']]

# Reduce weather data to relevant time slot
weather = weather.loc[(weather.index >= date_train_from) & (weather.index <= date_test_3M)]

# Check categorical variables
s = (weather.dtypes == 'object')
object_cols = list(s[s].index)

# print("Categorical variables:")
# print(object_cols)
# ['Events']
weather['Events'] = weather['Events'].astype('str')

# Transform 'Events' to numeric values
label_encoder = LabelEncoder()
for col in object_cols:
    weather[col] = label_encoder.fit_transform(weather[col])

# Join Sales with Weather
sales = pd.merge(left=sales, right=weather, how='left', left_on=sales.index, right_on=weather.index)
sales = sales.rename(mapper={'key_0': 'Date', 'Mean_TemperatureC': 'Temperature', 'Events': 'WeatherEvents'},
                     axis='columns')
sales = sales.set_index('Date')


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Store data for analysis
sales.to_pickle('../../../data/rossmann/intermediate/05_SalesModelingExtended/sales_extended.pkl')
