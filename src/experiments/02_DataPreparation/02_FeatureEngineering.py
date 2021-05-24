import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Get data
sales = pd.read_pickle('../../../data/rossmann/intermediate/01_SalesDataCleaned/sales.pkl')

# Transform sales column to float
sales['Sales'] = sales['Sales'].astype(float)

# ------------------------------------ FEATURE ENGINEERING -----------------------------------------

# Create new columns
# Add new column with day of month from date column
sales['DayOfMonth'] = sales['DateCol'].dt.day
# Add new column with month from date column
sales['Month'] = sales['DateCol'].dt.month
# Add new column with year from date column
sales['Year'] = sales['DateCol'].dt.year
# Add new column with week of year from date column
sales['WeekOfYear'] = sales.index.weekofyear


# Function to check if the PromotionInterval corresponds to the month col
def label_is_promo_month(row):
    if (row['PromoInterval'] == 'Jan,Apr,Jul,Oct') and (row['Month'] in [1, 4, 7, 10]):
        return 1
    if (row['PromoInterval'] == 'Feb,May,Aug,Nov') and (row['Month'] in [2, 5, 8, 11]):
        return 1
    if (row['PromoInterval'] == 'Mar,Jun,Sept,Dec') and (row['Month'] in [3, 6, 9, 12]):
        return 1
    return 0


# Add new column IsPromoMonth
sales['IsPromoMonth'] = sales.apply(lambda row: label_is_promo_month(row), axis=1)
# Drop PromoInterval after adding IsPromoMonth
sales = sales.drop('PromoInterval', axis=1)

# Get all variables of dataset
# print(sales.info())
#
# DatetimeIndex: 1017209 entries, 2015-07-31 to 2013-01-01
# Data columns (total 18 columns):
#  #   Column               Non-Null Count    Dtype
# ---  ------               --------------    -----
#  0   Store                1017209 non-null  int64
#  1   DayOfWeek            1017209 non-null  int64
#  2   Sales                1017209 non-null  float64
#  3   Customers            1017209 non-null  int64
#  4   Open                 1017209 non-null  int64
#  5   Promo                1017209 non-null  int64
#  6   StateHoliday         1017209 non-null  object
#  7   SchoolHoliday        1017209 non-null  int64
#  8   StoreType            1017209 non-null  object
#  9   Assortment           1017209 non-null  object
#  10  CompetitionDistance  1017209 non-null  float64
#  11  Promo2               1017209 non-null  int64
#  12  DateCol              1017209 non-null  datetime64[ns]
#  13  DayOfMonth           1017209 non-null  int64
#  14  Month                1017209 non-null  int64
#  15  Year                 1017209 non-null  int64
#  16  WeekOfYear           1017209 non-null  int64
#  17  IsPromoMonth         1017209 non-null  int64
# dtypes: datetime64[ns](1), float64(2), int64(12), object(3)


## Transform all objects to numeric values

# Transform 'StateHoliday'
# a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# a, b, c --> is a state holiday
# n --> is not a state holiday
# sales['StateHoliday'] = sales.StateHoliday.map({'n': 0, 'a': 1, 'b': 1, 'c': 1})

# print("Categorical variables:")
# print(object_cols)
# ['StoreType', 'Assortment', 'StateHoliday']

# Remaining categorical variables
s = (sales.dtypes == 'object')
object_cols = list(s[s].index)

# Transform 'StoreType', 'Assortment' and 'StateHoliday'
# 'StoreType'  --> 4 different store models: a, b, c, d
# 'Assortment' --> 3 assortment levels: a = basic, b = extra, c = extended
# 'StateHoliday' --> 4 different state holidays: a = state holiday, b = easter, c = christmas, n = no holiday
label_encoder = LabelEncoder()
for col in object_cols:
    sales[col] = label_encoder.fit_transform(sales[col])


# Drop redundant columns (DateCol)
sales = sales.drop(['DateCol'], axis=1)

## Final Features for Exploratory Data Analysis

# print(sales.info())
#
# DatetimeIndex: 1017209 entries, 2015-07-31 to 2013-01-01
# Data columns (total 17 columns):
#  #   Column               Non-Null Count    Dtype
# ---  ------               --------------    -----
#  0   Store                1017209 non-null  int64
#  1   DayOfWeek            1017209 non-null  int64
#  2   Sales                1017209 non-null  float64
#  3   Customers            1017209 non-null  int64
#  4   Open                 1017209 non-null  int64
#  5   Promo                1017209 non-null  int64
#  6   StateHoliday         1017209 non-null  int32
#  7   SchoolHoliday        1017209 non-null  int64
#  8   StoreType            1017209 non-null  int32
#  9   Assortment           1017209 non-null  int32
#  10  CompetitionDistance  1017209 non-null  float64
#  11  Promo2               1017209 non-null  int64
#  12  DayOfMonth           1017209 non-null  int64
#  13  Month                1017209 non-null  int64
#  14  Year                 1017209 non-null  int64
#  15  WeekOfYear           1017209 non-null  int64
#  16  IsPromoMonth         1017209 non-null  int64
# dtypes: float64(2), int32(3), int64(12)


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Store data for modeling tasks
sales.to_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales.pkl')
