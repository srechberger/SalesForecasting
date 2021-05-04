import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime
import warnings

warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Get data
sales = pd.read_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales.pkl')


# ----------------------------------- DISTRIBUTION BY COLUMNS ------------------------------------------

def plot_distribution(sales_data, no_subplots_per_row):
    # Select only numeric cols (there should not be any objects left in the dataset)
    num_columns = sales_data.select_dtypes(exclude=[object]).columns

    plt.figure(figsize=(20, 25))
    # Set number of rows for subplots
    no_of_rows = math.ceil(len(num_columns) / no_subplots_per_row)
    # Set has_statsmodel = False (to avoid error if bandwidth is 0)
    sns.distributions._has_statsmodels = False

    for i in range(len(num_columns)):
        plt.subplot(no_of_rows, no_subplots_per_row, i + 1)
        out = sns.distplot(sales_data[num_columns[i]])

    plt.tight_layout()
    plt.show()


# Plot data
plot_distribution(sales, 3)


# ------------------------------------- CORRELATION HEATMAP ---------------------------------------------

# Get correlation matrix
correlation = sales.corr()

# Plot heatmap of correlation matrix
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(correlation, mask=mask,
            square=True, linewidths=.5, ax=ax, cmap="BuPu")
plt.title("Correlation Heatmap", fontsize=20)
plt.tight_layout()
plt.show()

# -------------------------------- SALES BY DATE ---------------------------------------

# Sales by Year
plt.figure(figsize=(10, 6))
plt.title("Sales by Year")
sns.barplot(x='Year', y='Sales', data=sales)
plt.show()

# Sales by Month
plt.figure(figsize=(10, 6))
plt.title("Sales by Month")
sns.barplot(x='Month', y='Sales', data=sales)
plt.show()

# DayOfWeek
plt.figure(figsize=(10, 6))
plt.title("Sales by DayOfWeek")
sns.barplot(x='DayOfWeek', y='Sales', data=sales, order=[1, 2, 3, 4, 5, 6, 7])
plt.show()

# -------------------------------- SALES BY PROMO ---------------------------------------

# Promo
plt.figure(figsize=(10, 6))
plt.title("Sales by Promo")
sns.barplot(x='Promo', y='Sales', data=sales)
plt.show()

# Promo2
plt.figure(figsize=(10, 6))
plt.title("Sales by Promo2")
sns.barplot(x='Promo2', y='Sales', data=sales)
plt.show()

# Distribution Promo
sales_promo0 = sales[sales.Promo == 0]
sales_promo1 = sales[sales.Promo == 1]
# KDE plots
sns.kdeplot(data=sales_promo0['Sales'], label="Promo 0", shade=True)
sns.kdeplot(data=sales_promo1['Sales'], label="Promo 1", shade=True)
# Add title
plt.title("Distribution of Sales by Promo")
plt.show()

## Trend Analysis
# Sales trend over the months and year
sns.factorplot(data=sales, x="Month", y="Sales",
               col='Promo',
               hue='Promo2',
               row="Year")
plt.title("Trend Analysis over Month and Year")
plt.show()
# Conclusion: seasonality exists

# Sales trend over days
sns.factorplot(data=sales, x="DayOfWeek", y="Sales", hue="Promo")
plt.title("Trend Analysis over DayOfWeek")
plt.show()


# -------------------------------- SALES BY STATE HOLIDAYS ---------------------------------------

# StateHoliday
plt.figure(figsize=(10, 6))
plt.title("Sales by State Holiday")
sns.barplot(x='StateHoliday', y='Sales', data=sales)
plt.show()

# SchoolHoliday
plt.figure(figsize=(10, 6))
plt.title("Sales by School Holiday")
sns.barplot(x='SchoolHoliday', y='Sales', data=sales)
plt.show()


# ----------------------------- SALES BY STORE TYPE AND ASSORTMENT ---------------------------------------

# StoreType
plt.figure(figsize=(10, 6))
plt.title("Sales by Store Type")
sns.barplot(x='StoreType', y='Sales', data=sales)
plt.show()

# Assortment
plt.figure(figsize=(10, 6))
plt.title("Sales by Assortment")
sns.barplot(x='Assortment', y='Sales', data=sales)
plt.show()


# -------------------------------- SALES BY COMPETITION DISTANCE ---------------------------------------

# CompetitionDistance
sales.plot(kind='kde', x='CompetitionDistance', y='Sales', figsize=(15, 4))
plt.show()


# ----------------------------------- DATA SPLITTING -------------------------------------------------

# Define Prediction Intervals
date_train_until = datetime.datetime(2014, 12, 31)
date_test_2W = datetime.datetime(2015, 1, 14)
date_test_1M = datetime.datetime(2015, 1, 31)
date_test_3M = datetime.datetime(2015, 3, 31)


# Prepare Training Data for Modeling
train = sales.loc[(sales.index <= date_train_until)]
train_store708 = train.loc[(train.Store == 708)]
train_store198 = train.loc[(train.Store == 198)]
train_store897 = train.loc[(train.Store == 897)]

train_X = train.drop(['Sales'], axis=1)
train_store708_X = train_store708.drop(['Sales'], axis=1)
train_store198_X = train_store198.drop(['Sales'], axis=1)
train_store897_X = train_store897.drop(['Sales'], axis=1)

train_y = train['Sales']
train_store708_y = train_store708['Sales']
train_store198_y = train_store198['Sales']
train_store897_y = train_store897['Sales']


# Prepare Test Data for Modeling
test = sales.loc[(sales.index > date_train_until)]
test_store708 = test.loc[(test.Store == 708)]
test_store198 = test.loc[(test.Store == 198)]
test_store897 = test.loc[(test.Store == 897)]

test_X = test.drop(['Sales'], axis=1)
test_store708_X_2W = test_store708.loc[(test_store708.index <= date_test_2W)].drop(['Sales'], axis=1)
test_store708_X_1M = test_store708.loc[(test_store708.index <= date_test_1M)].drop(['Sales'], axis=1)
test_store708_X_3M = test_store708.loc[(test_store708.index <= date_test_3M)].drop(['Sales'], axis=1)
test_store198_X_2W = test_store198.loc[(test_store198.index <= date_test_2W)].drop(['Sales'], axis=1)
test_store198_X_1M = test_store198.loc[(test_store198.index <= date_test_1M)].drop(['Sales'], axis=1)
test_store198_X_3M = test_store198.loc[(test_store198.index <= date_test_3M)].drop(['Sales'], axis=1)
test_store897_X_2W = test_store897.loc[(test_store897.index <= date_test_2W)].drop(['Sales'], axis=1)
test_store897_X_1M = test_store897.loc[(test_store897.index <= date_test_1M)].drop(['Sales'], axis=1)
test_store897_X_3M = test_store897.loc[(test_store897.index <= date_test_3M)].drop(['Sales'], axis=1)

test_y = test['Sales']
test_store708_y_2W = test_store708.loc[(test_store708.index <= date_test_2W)]['Sales']
test_store708_y_1M = test_store708.loc[(test_store708.index <= date_test_1M)]['Sales']
test_store708_y_3M = test_store708.loc[(test_store708.index <= date_test_3M)]['Sales']
test_store198_y_2W = test_store198.loc[(test_store198.index <= date_test_2W)]['Sales']
test_store198_y_1M = test_store198.loc[(test_store198.index <= date_test_1M)]['Sales']
test_store198_y_3M = test_store198.loc[(test_store198.index <= date_test_3M)]['Sales']
test_store897_y_2W = test_store897.loc[(test_store897.index <= date_test_2W)]['Sales']
test_store897_y_1M = test_store897.loc[(test_store897.index <= date_test_1M)]['Sales']
test_store897_y_3M = test_store897.loc[(test_store897.index <= date_test_3M)]['Sales']


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Store data for modeling tasks
sales.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/sales.pkl')

# All Stores
train.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train.pkl')
train_X.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_X.pkl')
train_y.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/train_y.pkl')
test.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test.pkl')
test_X.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_X.pkl')
test_y.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/01_AllStores/test_y.pkl')

# Store 708
train_store708.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708.pkl')
train_store708_X.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_X.pkl')
train_store708_y.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708_y.pkl')
test_store708.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708.pkl')
test_store708_X_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_2W.pkl')
test_store708_X_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_1M.pkl')
test_store708_X_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_X_3M.pkl')
test_store708_y_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_2W.pkl')
test_store708_y_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_1M.pkl')
test_store708_y_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708_y_3M.pkl')

# Store 198
train_store198.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198.pkl')
train_store198_X.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_X.pkl')
train_store198_y.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198_y.pkl')
test_store198.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198.pkl')
test_store198_X_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_2W.pkl')
test_store198_X_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_1M.pkl')
test_store198_X_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_X_3M.pkl')
test_store198_y_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_2W.pkl')
test_store198_y_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_1M.pkl')
test_store198_y_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198_y_3M.pkl')

# Store 897
train_store897.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897.pkl')
train_store897_X.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_X.pkl')
train_store897_y.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897_y.pkl')
test_store897.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897.pkl')
test_store897_X_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_2W.pkl')
test_store897_X_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_1M.pkl')
test_store897_X_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_X_3M.pkl')
test_store897_y_2W.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_2W.pkl')
test_store897_y_1M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_1M.pkl')
test_store897_y_3M.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897_y_3M.pkl')














