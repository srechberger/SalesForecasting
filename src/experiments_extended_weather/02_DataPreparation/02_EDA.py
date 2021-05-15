import pandas as pd
import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Define Time Intervals
date_train_from = datetime.datetime(2013, 1, 1)
date_train_until = datetime.datetime(2014, 12, 31)
date_test_2W = datetime.datetime(2015, 1, 14)
date_test_1M = datetime.datetime(2015, 1, 31)
date_test_3M = datetime.datetime(2015, 3, 31)

# Get sales data
sales = pd.read_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/sales_extended.pkl')


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


# -------------------------------- SALES BY TEMPERATURE ----------------------------------------

# Sales by temperature
sns.factorplot(data=sales, x="Temperature", y="Sales")
plt.title("Sales by Temperature")
plt.show()

# -------------------------------- SALES BY WEATHER EVENT --------------------------------------

# WeatherEvents
plt.figure(figsize=(10, 6))
plt.title("Sales by Weather Event")
sns.barplot(x='WeatherEvents', y='Sales', data=sales)
plt.show()


# ----------------------------------- DATA SPLITTING -------------------------------------------------

# Sales
sales708 = sales.loc[(sales.Store == 708)]
sales198 = sales.loc[(sales.Store == 198)]
sales897 = sales.loc[(sales.Store == 897)]

# Prepare Training Data for Modeling
train = sales.loc[(sales.index >= date_train_from) & (sales.index <= date_train_until)]
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
sales.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/sales.pkl')
sales708.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/sales708.pkl')
sales198.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/sales198.pkl')
sales897.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/sales897.pkl')

# All Stores
train.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/train.pkl')
train_X.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/train_X.pkl')
train_y.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/train_y.pkl')
test.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/test.pkl')
test_X.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/test_X.pkl')
test_y.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/01_AllStores/test_y.pkl')

# Store 708
train_store708.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/train_store708.pkl')
train_store708_X.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/train_store708_X.pkl')
train_store708_y.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/train_store708_y.pkl')
test_store708.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708.pkl')
test_store708_X_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_2W.pkl')
test_store708_X_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_1M.pkl')
test_store708_X_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_X_3M.pkl')
test_store708_y_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_2W.pkl')
test_store708_y_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_1M.pkl')
test_store708_y_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/02_Store708/test_store708_y_3M.pkl')

# Store 198
train_store198.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198.pkl')
train_store198_X.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_X.pkl')
train_store198_y.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/train_store198_y.pkl')
test_store198.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198.pkl')
test_store198_X_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_2W.pkl')
test_store198_X_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_1M.pkl')
test_store198_X_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_X_3M.pkl')
test_store198_y_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_2W.pkl')
test_store198_y_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_1M.pkl')
test_store198_y_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/03_Store198/test_store198_y_3M.pkl')

# Store 897
train_store897.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/train_store897.pkl')
train_store897_X.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/train_store897_X.pkl')
train_store897_y.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/train_store897_y.pkl')
test_store897.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897.pkl')
test_store897_X_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_2W.pkl')
test_store897_X_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_1M.pkl')
test_store897_X_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_X_3M.pkl')
test_store897_y_2W.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_2W.pkl')
test_store897_y_1M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_1M.pkl')
test_store897_y_3M.to_pickle('../../../data/rossmann/intermediate/06_SalesModelingExtendedWeather/04_Store897/test_store897_y_3M.pkl')
