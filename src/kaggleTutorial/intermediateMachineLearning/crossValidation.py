# Cross-Validation

# For small datasets,
# where extra computational burden isn't a big deal, you should run cross-validation.

# For larger datasets,
# a single validation set is sufficient.
# Your code will run faster, and you may have enough data that there's little need to re-use some of it for holdout.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Read the data
data = pd.read_csv('../../../data/housingPrices/input/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price


# Define a pipeline that uses an imputer to fill in missing values and a random forest model to make predictions.

# While it's possible to do cross-validation without pipelines, it is quite difficult!
# Using a pipeline will make the code remarkably straightforward.

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

# cross_val_score()
# We obtain the cross-validation scores with the cross_val_score() function from scikit-learn.
# We set the number of folds with the cv parameter.

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# common error rates: https://scikit-learn.org/stable/modules/model_evaluation.html

# calc average error rate
print("Average MAE score (across experiments):")
print(scores.mean())
