import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Read the data
train_data = pd.read_csv('../../../data/kaggleTutorials/input/train.csv', index_col='Id')
test_data = pd.read_csv('../../../data/kaggleTutorials/input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())

# Begin by writing a function get_score() that reports the average
# (over three cross-validation folds) MAE of a machine learning pipeline that uses:
#
#     the data in X and y to create folds,
#     SimpleImputer() (with all parameters left as default) to replace missing values, and
#     RandomForestRegressor() (with random_state=0) to fit a random forest model.
#
# The n_estimators parameter supplied to get_score() is used
# when setting the number of trees in the random forest model.

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
    return scores.mean()

# Now, you will use the function that you defined in Step 1 to evaluate the model performance
# corresponding to eight different values for the number of trees in the random forest:
# 50, 100, 150, ..., 300, 350, 400.
#
# Store your results in a Python dictionary results, where results[i] is the average MAE returned by get_score(i).
results = {}
for i in range(1,9):
    results[50*i] = get_score(50*i)

# Use matplot to visualize the results
plt.plot(list(results.keys()), list(results.values()))
plt.show()

###### Find the best parameter value ######
n_estimators_best = min(results, key=results.get)

# If you'd like to learn more about hyperparameter optimization,
# you're encouraged to start with grid search,
# which is a straightforward method for determining the best combination of parameters for a machine learning model.
# Thankfully, scikit-learn also contains a built-in function GridSearchCV()
# that can make your grid search code very efficient!
# https://en.wikipedia.org/wiki/Hyperparameter_optimization
