import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
warnings.filterwarnings('ignore')

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

df = pd.read_csv("../../../data/kaggleTutorials/input/ames.csv.zip")

# First you'll need to choose which features you want to apply a target encoding to.
# Categorical features with a large number of categories are often good candidates.
# Run this cell to see how many categories each categorical feature in the Ames dataset has.

print(df.select_dtypes(["object"]).nunique())

# We talked about how the M-estimate encoding uses smoothing to improve estimates for rare categories.
# To see how many times a category occurs in the dataset, you can use the value_counts method.
# This cell shows the counts for SaleType, but you might want to consider others as well.

print(df["SaleType"].value_counts())

# Which features did you identify for target encoding?
# After you've thought about your answer, run the next cell for some discussion.
# --> The Neighborhood feature looks promising.
# --> Others that could be worth considering are SaleType, MSSubClass, Exterior1st, Exterior2nd

# Now you'll apply a target encoding to your choice of feature.
# As we discussed in the tutorial, to avoid overfitting,
# we need to fit the encoder on data heldout from the training set.
# Run this cell to create the encoding and training splits:

# Encoding split
X_encode = df.sample(frac=0.20, random_state=0)
y_encode = X_encode.pop("SalePrice")

# Training split
X_pretrain = df.drop(X_encode.index)
y_train = X_pretrain.pop("SalePrice")

### Apply M-Estimate Encoding
# Apply a target encoding to your choice of categorical features.
# Also choose a value for the smoothing parameter m (any value is okay for a correct answer).

# Create the MEstimateEncoder
# Choose a set of features to encode and a value for m
encoder = MEstimateEncoder(
    cols=["Neighborhood"],
    m=1.0,
)

# Fit the encoder on the encoding split
encoder.fit(X_encode, y_encode)

# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)

# If you'd like to see how the encoded feature compares to the target, you can run this cell:
feature = encoder.cols

plt.figure(dpi=90)
ax = sns.distplot(y_train, kde=True, hist=False)
ax = sns.distplot(X_train[feature], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

# From the distribution plots, does it seem like the encoding is informative?
#
# And this cell will show you the score of the encoded set compared to the original set:
X = df.copy()
y = X.pop("SalePrice")
score_base = score_dataset(X, y)
score_new = score_dataset(X_train, y_train)

print(f"Baseline Score: {score_base:.4f} RMSLE")
print(f"Score with Encoding: {score_new:.4f} RMSLE")

######
# In this question, you'll explore the problem of overfitting with target encodings.
# This will illustrate this importance of training fitting target encoders on data held-out from the training set.
#
# So let's see what happens when we fit the encoder and the model on the same dataset.
# To emphasize how dramatic the overfitting can be,
# we'll mean-encode a feature that should have no relationship with SalePrice, a count: 0, 1, 2, 3, 4, 5, ....

# Try experimenting with the smoothing parameter m
# Try 0, 1, 5, 50
m = 0

X = df.copy()
y = X.pop('SalePrice')

# Create an uninformative feature
X["Count"] = range(len(X))
X["Count"][1] = 0  # actually need one duplicate value to circumvent error-checking in MEstimateEncoder

# fit and transform on the same dataset
encoder = MEstimateEncoder(cols="Count", m=m)
X = encoder.fit_transform(X, y)

# Results
score =  score_dataset(X, y)
print(f"Score: {score:.4f} RMSLE")

# And the distributions are almost exactly the same, too.
plt.figure(dpi=90)
ax = sns.distplot(y, kde=True, hist=False)
ax = sns.distplot(X["Count"], color='r', ax=ax, hist=True, kde=False, norm_hist=True)
ax.set_xlabel("SalePrice")
plt.show()

##### Tutorial fuer gesamten Prozess siehe:
# https://www.kaggle.com/ryanholbrook/feature-engineering-for-house-prices
