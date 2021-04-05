import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("../../../data/kaggleTutorials/input/concrete.csv")
print(df.head())

# We'll first establish a baseline by training the model on the un-augmented dataset.
# This will help us determine whether our new features are actually useful.

# Establishing baselines like this is good practice at the start of the feature engineering process.
# A baseline score can help you decide whether your new features are worth keeping,
# or whether you should discard them and possibly try something else.

X = df.copy()
y = X.pop("CompressiveStrength")

# Train and score baseline model
baseline = RandomForestRegressor(criterion="mae", random_state=0)
baseline_score = cross_val_score(
    baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")

# If you ever cook at home, you might know that the ratio of ingredients in a recipe is usually a better predictor
# of how the recipe turns out than their absolute amounts.
# We might reason then that ratios of the features above would be a good predictor of CompressiveStrength.

X = df.copy()
y = X.pop("CompressiveStrength")

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score model on dataset with additional ratio features
model = RandomForestRegressor(criterion="mae", random_state=0)
score = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Features: {score:.4}")
