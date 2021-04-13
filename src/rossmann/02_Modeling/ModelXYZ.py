import pandas as pd

# set display options
pd.set_option('display.max_columns', 15)

sales = pd.read_pickle('../../../data/rossmann/intermediate/sales.pkl')

print(sales.dtypes)
