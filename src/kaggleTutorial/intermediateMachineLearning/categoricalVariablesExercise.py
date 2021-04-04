import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Read the data
X_full = pd.read_csv('../../../data/housingPrices/input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../../../data/housingPrices/input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Show first 5 rows of train data
print(X_train.head())

# function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# 1) Drop Categorical Variables
# Fill in the lines below: drop columns in training and validation data
#drop_X_train = X_train.select_dtypes(exclude=['object'])
#drop_X_valid = X_valid.select_dtypes(exclude=['object'])

#print("MAE from Approach 1 (Drop categorical variables):")
#print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

# Before jumping into label encoding, we'll investigate the dataset.
# Specifically, we'll look at the 'Condition2' column.
# prints the unique entries in both the training and validation sets.
#print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
#print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

# 2) Label encoding
# Fitting a label encoder to a column in the training data creates a corresponding integer-valued label
# for each unique value that appears in the training data.
# In the case that the validation data contains values that don't also appear in the training data,
# the encoder will throw an error, because these values won't have an integer assigned to them.
# Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn',
# but these don't appear in the training data -- thus, if we try to use a label encoder with scikit-learn,
# the code will throw an error.

# This is a common problem that you'll encounter with real-world data,
# and there are many approaches to fixing this issue.
# For instance, you can write a custom label encoder to deal with new categories.
# The simplest approach, however, is to drop the problematic categorical columns.

# Run the code cell below to save the problematic columns to a Python list bad_label_cols.
# Likewise, columns that can be safely label encoded are stored in good_label_cols.

# All categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Columns that can be safely label encoded
good_label_cols = [col for col in object_cols if
                   set(X_train[col]) == set(X_valid[col])]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('Categorical columns that will be label encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)


# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder
label_encoder = LabelEncoder()
for col in set(good_label_cols):
    label_X_train[col] = label_encoder.fit_transform(X_train[col])
    label_X_valid[col] = label_encoder.transform(X_valid[col])

# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)
