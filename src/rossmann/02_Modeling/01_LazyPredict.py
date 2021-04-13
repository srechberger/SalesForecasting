# Terminal Statement to install lazypredict
# pip install lazypredict

# Most likely, you will encounter some errors about missing libraries,
# so just install them separately using pip or conda.
# I mention this later on as a possible improvement.
# Then, we load the required libraries:
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

# set display options
pd.set_option('display.max_columns', 15)

# Load data
sales = pd.read_pickle('../../../data/rossmann/intermediate/store6.pkl')
# Create target object
y = sales.Sales
# Create features
features = ['DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
X = sales[features]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=None)

# fit all models
reg = LazyRegressor(predictions=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

