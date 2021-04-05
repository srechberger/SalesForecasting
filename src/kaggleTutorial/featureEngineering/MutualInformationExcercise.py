import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

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


# Load data
df = pd.read_csv("../../../data/else/ames.csv.zip")

# Utility functions from Tutorial
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

# To start, let's review the meaning of mutual information by looking at a few features from the Ames dataset.
features = ["YearBuilt", "MoSold", "ScreenPorch"]
path = "../../../figures/miE1.png"
plot = sns.relplot(
    x="value",
    y="SalePrice",
    col="variable",
    data=df.melt(id_vars="SalePrice", value_vars=features),
    facet_kws=dict(sharex=False),
)
plot.savefig(path)

# The Ames dataset has seventy-eight features -- a lot to work with all at once!
# Fortunately, you can identify the features with the most potential.
#
# Use the make_mi_scores function (introduced in the tutorial)
# to compute mutual information scores for the Ames features:

X = df.copy()
y = X.pop('SalePrice')

mi_scores = make_mi_scores(X, y)

# Now examine the scores using the functions in this cell. Look especially at top and bottom ranks.
print(mi_scores.head(20))
print(mi_scores.tail(20))

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
plot_mi_scores(mi_scores.tail(20))
plt.show()

# The BldgType feature didn't get a very high MI score.
# A plot confirms that the categories in BldgType don't do a good job of distinguishing values in SalePrice
# (the distributions look fairly similar, in other words):

path = "../../../figures/miE2.png"
plot = sns.catplot(x="BldgType", y="SalePrice", data=df, kind="boxen")
plot.savefig(path)

# Still, the type of a dwelling seems like it should be important information.
# Investigate whether BldgType produces a significant interaction with either of the following:

# GrLivArea  - Above ground living area
# MoSold     - Month sold

feature = "GrLivArea"
path = "../../../figures/miE3.png"
plot = sns.lmplot(
    x=feature, y="SalePrice", hue="BldgType", col="BldgType",
    data=df, scatter_kws={"edgecolor": 'w'}, col_wrap=3, height=4,
)
plot.savefig(path)

print(mi_scores.head(10))

# Do you recognize the themes here? Location, size, and quality.
# You needn't restrict development to only these top features,
# but you do now have a good place to start.
# Combining these top features with other related features,
# especially those you've identified as creating interactions,
# is a good strategy for coming up with a highly informative set of features to train your model on.
