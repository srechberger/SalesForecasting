import pandas as pd
import datetime

# Train data
train_708 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/train_store708.pkl')
train_198 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/train_store198.pkl')
train_897 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/train_store897.pkl')

# Test data
test_708 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/test_store708.pkl')
test_198 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/test_store198.pkl')
test_897 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/test_store897.pkl')

# Sales data
sales_708 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/02_Store708/sales708.pkl')
sales_198 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/03_Store198/sales198.pkl')
sales_897 = pd.read_pickle(
    '../../../../../data/rossmann/intermediate/03_SalesModelingBase/04_Store897/sales897.pkl')


# Drop not important features
drop_features = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance', 'Promo2', 'DayOfMonth', 'IsPromoMonth']

# Transform train data
train_708 = train_708.drop(drop_features, axis=1)
train_708['Date'] = train_708.index
train_708.to_csv('train_708.csv', index=False, header=True)
train_198 = train_198.drop(drop_features, axis=1)
train_198['Date'] = train_198.index
train_198.to_csv('train_198.csv', index=False, header=True)
train_897 = train_897.drop(drop_features, axis=1)
train_897['Date'] = train_897.index
train_897.to_csv('train_897.csv', index=False, header=True)

# Transform test data
date_test_2W = datetime.datetime(2015, 1, 14)
date_test_1M = datetime.datetime(2015, 1, 31)
date_test_3M = datetime.datetime(2015, 3, 31)
test_708 = test_708.drop(drop_features, axis=1)
test_708['Date'] = test_708.index
test_708_2W = test_708.loc[(test_708.index <= date_test_2W)]
test_708_2W.to_csv('test_708_2W.csv', index=False, header=True)
test_708_1M = test_708.loc[(test_708.index <= date_test_1M)]
test_708_1M.to_csv('test_708_1M.csv', index=False, header=True)
test_708_3M = test_708.loc[(test_708.index <= date_test_3M)]
test_708_3M.to_csv('test_708_3M.csv', index=False, header=True)
test_198 = test_198.drop(drop_features, axis=1)
test_198['Date'] = test_198.index
test_198_2W = test_198.loc[(test_198.index <= date_test_2W)]
test_198_2W.to_csv('test_198_2W.csv', index=False, header=True)
test_198_1M = test_198.loc[(test_198.index <= date_test_1M)]
test_198_1M.to_csv('test_198_1M.csv', index=False, header=True)
test_198_3M = test_198.loc[(test_198.index <= date_test_3M)]
test_198_3M.to_csv('test_198_3M.csv', index=False, header=True)
test_897 = test_897.drop(drop_features, axis=1)
test_897['Date'] = test_897.index
test_897_2W = test_897.loc[(test_897.index <= date_test_2W)]
test_897_2W.to_csv('test_897_2W.csv', index=False, header=True)
test_897_1M = test_897.loc[(test_897.index <= date_test_1M)]
test_897_1M.to_csv('test_897_1M.csv', index=False, header=True)
test_897_3M = test_897.loc[(test_897.index <= date_test_3M)]
test_897_3M.to_csv('test_897_3M.csv', index=False, header=True)

# Sales data
sales_708['Date'] = sales_708.index
sales_708_2W = sales_708.loc[(sales_708.index <= date_test_2W)]
sales_708_2W.to_csv('sales_708_2W.csv', index=False, header=True)
sales_708_1M = sales_708.loc[(sales_708.index <= date_test_1M)]
sales_708_1M.to_csv('sales_708_1M.csv', index=False, header=True)
sales_708_3M = sales_708.loc[(sales_708.index <= date_test_3M)]
sales_708_3M.to_csv('sales_708_3M.csv', index=False, header=True)
sales_198 = sales_198.drop(drop_features, axis=1)
sales_198['Date'] = sales_198.index
sales_198_2W = sales_198.loc[(sales_198.index <= date_test_2W)]
sales_198_2W.to_csv('sales_198_2W.csv', index=False, header=True)
sales_198_1M = sales_198.loc[(sales_198.index <= date_test_1M)]
sales_198_1M.to_csv('sales_198_1M.csv', index=False, header=True)
sales_198_3M = sales_198.loc[(sales_198.index <= date_test_3M)]
sales_198_3M.to_csv('sales_198_3M.csv', index=False, header=True)
sales_897 = sales_897.drop(drop_features, axis=1)
sales_897['Date'] = sales_897.index
sales_897_2W = sales_897.loc[(sales_897.index <= date_test_2W)]
sales_897_2W.to_csv('sales_897_2W.csv', index=False, header=True)
sales_897_1M = sales_897.loc[(sales_897.index <= date_test_1M)]
sales_897_1M.to_csv('sales_897_1M.csv', index=False, header=True)
sales_897_3M = sales_897.loc[(sales_897.index <= date_test_3M)]
sales_897_3M.to_csv('sales_897_3M.csv', index=False, header=True)
