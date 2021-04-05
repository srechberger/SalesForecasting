import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

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

# Prepare data
df = pd.read_csv("../../../data/kaggleTutorials/input/ames.csv.zip")
X = df.copy()
y = X.pop("SalePrice")

# Create the following features:
#
#     LivLotRatio: the ratio of GrLivArea to LotArea
#     Spaciousness: the sum of FirstFlrSF and SecondFlrSF divided by TotRmsAbvGrd
#     TotalOutsideSF: the sum of WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, and ScreenPorch

X_1 = pd.DataFrame()  # dataframe to hold new features

X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch

# If you've discovered an interaction effect between a numeric feature and a categorical feature,
# you might want to model it explicitly using a one-hot encoding, like so:

# One-hot encode Categorical feature, adding a column prefix "Cat"
# X_new = pd.get_dummies(df.Categorical, prefix="Cat")

# Multiply row-by-row
# X_new = X_new.mul(df.Continuous, axis=0)

# Join the new features to the feature set
# X = X.join(X_new)

### Interaction with a Categorical
# We discovered an interaction between BldgType and GrLivArea in Exercise 2. Now create their interaction features.

# One-hot encode BldgType. Use `prefix="Bldg"` in `get_dummies`
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
# Multiply
X_2 = X_2.mul(df.GrLivArea, axis=0)

### Count Feature

# Let's try creating a feature that describes how many kinds of outdoor areas a dwelling has.
# Create a feature PorchTypes that counts how many of the following are greater than 0.0:

# WoodDeckSF, OpenPorchSF, EnclosedPorch, Threeseasonporch, ScreenPorch

X_3 = pd.DataFrame()

X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)

### Break Down a Categorical Feature

# MSSubClass describes the type of a dwelling:
df.MSSubClass.unique()

# You can see that there is a more general categorization described (roughly) by the first word of each category.
# Create a feature containing only these first words by splitting MSSubClass at the first underscore _.
# (Hint: In the split method use an argument n=1.)

X_4 = pd.DataFrame()

X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

### Use a Grouped Transform

# The value of a home often depends on how it compares to typical homes in its neighborhood.
# Create a feature MedNhbdArea that describes the median of GrLivArea grouped on Neighborhood.

X_5 = pd.DataFrame()

X_5["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

# Now you've made your first new feature set!
# If you like, you can run the cell below to score the model with all of your new features added:
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)

print('Original score:' + str(score_dataset(X, y)))
print('Score with new features:' + str(score_dataset(X_new, y)))
