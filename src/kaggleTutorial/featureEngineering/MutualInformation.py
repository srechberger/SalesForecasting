# A great first step is to construct a ranking with a feature utility metric,
# a function measuring associations between a feature and the target.
# Then you can choose a smaller set of the most useful features to develop initially and
# have more confidence that your time will be well spent.

# The metric we'll use is called "mutual information".
# Mutual information is a lot like correlation in that it measures a relationship between two quantities.
# The advantage of mutual information is that it can detect any kind of relationship,
# while correlation only detects linear relationships

# Mutual Information
# Mutual information describes relationships in terms of uncertainty.
# The mutual information (MI) between two quantities is a measure of the extent
# to which knowledge of one quantity reduces uncertainty about the other.
# If you knew the value of a feature, how much more confident would you be about the target?

# It's possible for a feature to be very informative when interacting with other features,
# but not so informative all alone.
# MI can't detect interactions between features. It is a univariate metric.

# The actual usefulness of a feature depends on the model you use it with.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plt.style.use("seaborn-whitegrid")

df = pd.read_csv("../../../data/else/autos.csv")
print(df.head())

# The scikit-learn algorithm for MI treats discrete features differently from continuous features.
# Consequently, you need to tell it which are which.
# As a rule of thumb, anything that must have a float dtype is not discrete.
# Categoricals (object or categorial dtype) can be treated as discrete by giving them a label encoding.
# (You can review label encodings in our Categorical Variables lesson.)

X = df.copy()
y = X.pop("price")

# Label encoding for categoricals
for colname in X.select_dtypes("object"):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int

# Scikit-learn has two mutual information metrics in its feature_selection module:
#
# one for real-valued targets (mutual_info_regression) and
# one for categorical targets (mutual_info_classif).
#
# Our target, price, is real-valued.
# The next cell computes the MI scores for our features and wraps them up in a nice dataframe.

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores[::3])  # show a few features with their MI scores


# And now a bar plot to make comparisions easier:

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()

# Data visualization is a great follow-up to a utility ranking. Let's take a closer look at a couple of these.
#
# As we might expect, the high-scoring curb_weight feature exhibits a strong relationship with price, the target.

path = "../../../figures/mi1.png"
plot = sns.relplot(x="curb_weight", y="price", data=df)
plot.savefig(path)

path = "../../../figures/mi2.png"
plot = sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df)
plot.savefig(path)


