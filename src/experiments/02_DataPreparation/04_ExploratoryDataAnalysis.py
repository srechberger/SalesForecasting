import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# set display options
pd.set_option('display.max_columns', 20)

# Get data
sales = pd.read_pickle('../../../data/rossmann/intermediate/02_SalesDataPrepared/sales.pkl')

# Get correlation matrix
correlation = sales.corr()

# Plot heatmap of correlation matrix
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(correlation, mask=mask,
            square=True, linewidths=.5, ax=ax, cmap="BuPu")
plt.title("Correlation Heatmap", fontsize=20)
plt.show()


## Trend Analysis
# Sales trend over the months and year
sns.factorplot(data=sales, x="Month", y="Sales",
               col='Promo',
               hue='Promo2',
               row="Year")
plt.show()

# Conclusion: seasonality exists

# Sales trend over days
sns.factorplot(data=sales, x="DayOfWeek", y="Sales", hue="Promo")
plt.show()

## Stationary Analysis


# Assigning store 6
store6 = sales[sales.Store == 25]['Sales']

# Rolling mean analysis (Stationary)

# Function to test the stationarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    roll_mean = timeseries.rolling(window=7).mean()
    roll_std = timeseries.rolling(window=7).std()

    print(roll_std)

    # Plotting rolling statistics:
    orig = plt.plot(timeseries.resample('W').mean(), color='blue', label='Original')
    mean = plt.plot(roll_mean.resample('W').mean(), color='red', label='Rolling Mean')
    std = plt.plot(roll_std.resample('W').mean(), color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.show(block=False)

# Testing stationarity of store type a
test_stationarity(store6)


# ----------------------------------- DATA STORAGE ---------------------------------------------------

# Get data of single stores
sales_store708 = sales.loc[sales.Store == 708]
sales_store198 = sales.loc[sales.Store == 198]
sales_store897 = sales.loc[sales.Store == 897]

# Store data for modeling tasks
sales.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/sales.pkl')
sales_store708.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/sales_store708.pkl')
sales_store198.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/sales_store198.pkl')
sales_store897.to_pickle('../../../data/rossmann/intermediate/03_SalesModelingBase/sales_store897.pkl')
