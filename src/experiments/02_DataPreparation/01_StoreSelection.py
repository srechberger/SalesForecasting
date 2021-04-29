# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

train_file_path = "../../../data/rossmann/input/train.csv"
sales = pd.read_csv(train_file_path)

# Parse date column from object to datetime
sales['Date'] = pd.to_datetime(sales['Date'], format="%Y-%m-%d")


# function for Coefficient of Variation (CV = Variationskoeffizient)
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

# CV calculation per store
rows = []
for i in range(1, 1116):
    store_data = sales.loc[sales.Store == i]['Sales']
    koe = cv(store_data)
    rows.append([i, koe])

koe_store = pd.DataFrame(rows, columns=["StoreId", "CV"])
koe_store = koe_store.sort_values(by='CV', ascending=False)
# print(koe_store)
#
#       StoreId         CV
# 707       708  80.636297
# 102       103  76.964025
# 971       972  76.759814
# 197       198  75.514548
# 896       897  73.228068
# ...       ...        ...
# 422       423  20.018492
# 1096     1097  19.507132
# 768       769  17.100971
# 561       562  16.316082
# 732       733  12.308368
#
# --> Selection of top 5 stores
# --> StoreIds: 708, 103, 972, 198, 897


# Get data of single stores
sales_store708 = sales.loc[sales.Store == 708]
sales_store103 = sales.loc[sales.Store == 103]
sales_store972 = sales.loc[sales.Store == 972]
sales_store198 = sales.loc[sales.Store == 198]
sales_store897 = sales.loc[sales.Store == 897]


def plot_stores(store_data, store_id):
    # get defined columns
    store_data = store_data.loc[:, ['Sales', 'Date']]
    # sort by date
    store_data = store_data.sort_values(by=['Date'])
    # groupby Date if multiple entries per date exist
    store_data = store_data.groupby('Date').Sales.sum()

    # get rolling (weekly) data
    weekly_mean = store_data.rolling(window=7).mean() # window size 7 steht für 7 Tage
    weekly_std = store_data.rolling(window=7).std()

    # set title
    title = 'Sales Figures ' + store_id

    # plot rolling statistics
    orig = plt.plot(store_data, color='turquoise', label='Original')
    mean = plt.plot(weekly_mean, color='darkgoldenrod', label='Weekly Rolling Mean')
    std = plt.plot(weekly_std, color='indigo', label='Weekly Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


# plot selected stores
plot_stores(sales, 'All Stores')
plot_stores(sales_store708, 'Store 708')
plot_stores(sales_store103, 'Store 103')
plot_stores(sales_store972, 'Store 972')
plot_stores(sales_store198, 'Store 198')
plot_stores(sales_store897, 'Store 897')

# --> Final Selection: 708, 198, 897
# 708, 103, 972: all 3 stores are affected by refurbishment
# 198, 897: no refurbishment
