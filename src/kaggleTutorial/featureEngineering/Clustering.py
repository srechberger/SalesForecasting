# Clustering simply means the assigning of data points to groups based upon how similar the points are to each other.
# A clustering algorithm makes "birds of a feather flock together," so to speak.

# When used for feature engineering, we could attempt discover groups of customers representing a market segment,
# for instance, or geographic areas that share similar weather patterns.
# Adding a feature of cluster labels can help machine learning models
# untangle complicated relationships of space or proximity.

# It's important to remember that this Cluster feature is categorical.
# Here, it's shown with a label encoding (that is, as a sequence of integers)
# as a typical clustering algorithm would produce; depending on your model,
# a one-hot encoding may be more appropriate.

# The motivating idea for adding cluster labels is that the clusters
# will break up complicated relationships across features into simpler chunks.
# Our model can then just learn the simpler chunks one-by-one instead
# having to learn the complicated whole all at once.
# It's a "divide and conquer" strategy.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

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

df = pd.read_csv("../../../data/else/housing.csv.zip")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]
print(X.head())

# Since k-means clustering is sensitive to scale, it can be a good idea rescale or normalize data with extreme values.
# Our features are already roughly on the same scale, so we'll leave them as-is.

# Create cluster feature
kmeans = KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

print(X.head())

# Now let's look at a couple plots to see how effective this was.
# First, a scatter plot that shows the geographic distribution of the clusters.
# It seems like the algorithm has created separate segments for higher-income areas on the coasts.

path = "../../../figures/cluster1.png"
plot = sns.relplot(x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,)
plot.savefig(path)
plt.show()

# The target in this dataset is MedHouseVal (median house value).
# These box-plots show the distribution of the target within each cluster.
# If the clustering is informative, these distributions should, for the most part,
# separate across MedHouseVal, which is indeed what we see.

X["MedHouseVal"] = df["MedHouseVal"]
path = "../../../figures/cluster2.png"
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)
plot.savefig(path)
plt.show()
