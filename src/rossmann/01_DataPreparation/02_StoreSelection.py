# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sales = pd.read_pickle('../../../data/rossmann/intermediate/sales.pkl')

# Funktion für Variationskoeffizient (cv = koe)
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

# Berechnung Variationskoeffizient pro Store
rows = []
for i in range(1, 1116):
    store_data = sales.loc[sales.Store == i]['Sales']
    koe = cv(store_data)
    rows.append([i, koe])

koe_store = pd.DataFrame(rows, columns=["StoreId", "CV"])
koe_store = koe_store.sort_values(by='CV', ascending=False)
# print(koe_store)
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
# --> Auswahl von 3 Stores mit Schwankungsbreite > 75 %
# --> StoreIds: 708, 103, 972

## Plot data

# Sum of all stores
grouped_sales = sales.groupby('DateCol').Sales.sum()
# print(grouped_sales)

# Mean of all stores
# grouped_sales = sales.groupby('DateCol').Sales.mean()
# print(grouped_sales)

# Determine rolling statistics
rolmean = grouped_sales.rolling(window=7).mean() #window size 7 steht für 7 Tage
rolstd = grouped_sales.rolling(window=7).std()

# Plot rolling statistics for all stores
orig = plt.plot(grouped_sales, color='turquoise', label='Original')
mean = plt.plot(rolmean, color='darkgoldenrod', label='Rolling Mean')
std = plt.plot(rolstd, color='indigo', label='Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Summed Sales Figures of all Stores')
plt.show()

# -----------------------------------------------------------------------------------
### STORE 1

# Select data for StoreId == 1
sales_Store1 = sales.loc[sales.Store == 1]

# get defined columns
sales_Store1 = sales_Store1.loc[:, ['Sales', 'DateCol']]

# sort by date
sales_Store1 = sales_Store1.sort_values(by=['DateCol'])

# groupby Date if multiple entries per date exist
sales_Store1 = sales_Store1.groupby('DateCol').Sales.sum()

# Determine rolling statistics
rolmean_S1 = sales_Store1.rolling(window=7).mean() #window size 30 denotes 30 days
rolstd_S1 = sales_Store1.rolling(window=7).std()

# Plot rolling statistics for Store 1
orig_S1 = plt.plot(sales_Store1, color='turquoise', label='Original')
mean_S1 = plt.plot(rolmean_S1, color='darkgoldenrod', label='Rolling Mean')
std_S1 = plt.plot(rolstd_S1, color='indigo', label='Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures Store 1')
plt.show()

# -----------------------------------------------------------------------------------
### Several Stores

# Select data for different stores
sales_Store6 = sales.loc[sales.Store == 6]
sales_Store20 = sales.loc[sales.Store == 20]
sales_Store25 = sales.loc[sales.Store == 25]

# Store data for modeling tasks
sales_Store6 .to_pickle('../../../data/rossmann/intermediate/store6.pkl')
sales_Store20.to_pickle('../../../data/rossmann/intermediate/store20.pkl')
sales_Store25.to_pickle('../../../data/rossmann/intermediate/store25.pkl')

# get defined columns
sales_Store6 = sales_Store6.loc[:, ['Sales', 'DateCol']]
sales_Store20 = sales_Store20.loc[:, ['Sales', 'DateCol']]
sales_Store25 = sales_Store25.loc[:, ['Sales', 'DateCol']]

# sort by date
sales_Store6 = sales_Store6.sort_values(by=['DateCol'])
sales_Store20 = sales_Store20.sort_values(by=['DateCol'])
sales_Store25 = sales_Store25.sort_values(by=['DateCol'])

# groupby Date if multiple entries per date exist
sales_Store6 = sales_Store6.groupby('DateCol').Sales.sum()
sales_Store20 = sales_Store20.groupby('DateCol').Sales.sum()
sales_Store25 = sales_Store25.groupby('DateCol').Sales.sum()

# Determine rolling statistics
rolmean_S6 = sales_Store6.rolling(window=7).mean()
rolmean_S20 = sales_Store20.rolling(window=7).mean()
rolmean_S25 = sales_Store25.rolling(window=7).mean()

# Plot rolling statistics for Store 1
store6 = plt.plot(rolmean_S6, color='turquoise', label='Store 6')
store20 = plt.plot(rolmean_S20, color='darkgoldenrod', label='Store 20')
store25 = plt.plot(rolmean_S25, color='indigo', label='Store 25')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures of several Stores')
plt.show()

# ----------------------------------

# Determine rolling statistics
weekly_mean_store6 = sales_Store6.rolling(window=7).mean()
weekly_std_store6 = sales_Store6.rolling(window=7).std()

# Plot rolling statistics for Store 6
orig_S6 = plt.plot(sales_Store6, color='turquoise', label='Original')
mean_S6 = plt.plot(weekly_mean_store6, color='darkgoldenrod', label='Weekly Rolling Mean')
std_S6 = plt.plot(weekly_std_store6, color='indigo', label='Weekly Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures Store 6')
plt.show()

# Determine rolling statistics
weekly_mean_store20 = sales_Store20.rolling(window=7).mean()
weekly_std_store20 = sales_Store20.rolling(window=7).std()

# Plot rolling statistics for Store 20
orig_S20 = plt.plot(sales_Store20, color='turquoise', label='Original')
mean_S20 = plt.plot(weekly_mean_store20, color='darkgoldenrod', label='Weekly Rolling Mean')
std_S20 = plt.plot(weekly_std_store20, color='indigo', label='Weekly Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures Store 20')
plt.show()

# Determine rolling statistics
weekly_mean_store25 = sales_Store25.rolling(window=7).mean()
weekly_std_store25 = sales_Store25.rolling(window=7).std()

# Plot rolling statistics for Store 25
orig_S25 = plt.plot(sales_Store25, color='turquoise', label='Original')
mean_S25 = plt.plot(weekly_mean_store25, color='darkgoldenrod', label='Weekly Rolling Mean')
std_S25 = plt.plot(weekly_std_store25, color='indigo', label='Weekly Rolling Std')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc='best')
plt.title('Sales Figures Store 25')
plt.show()



