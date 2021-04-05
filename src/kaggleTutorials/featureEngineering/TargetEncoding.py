# Most of the techniques we've seen in this course have been for numerical features.
# The technique we'll look at in this lesson, target encoding, is instead meant for categorical features.
# It's a method of encoding categories as numbers, like one-hot or label encoding,
# with the difference that it also uses the target to create the encoding.
# This makes it what we call a supervised feature engineering technique.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from category_encoders.m_estimate import MEstimateEncoder

autos = pd.read_csv("../../../data/kaggleTutorials/input/autos.csv")

# Target Encoding
# A target encoding is any kind of encoding that replaces
# a feature's categories with some number derived from the target.

# A simple and effective version is to apply a group aggregation from Lesson 3, like the mean.
# Using the Automobiles dataset, this computes the average price of each vehicle's make:

autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

print(autos[["make", "price", "make_encoded"]].head(10))

# This kind of target encoding is sometimes called a mean encoding.
# Applied to a binary target, it's also called bin counting.
# (Other names you might come across include: likelihood encoding, impact encoding, and leave-one-out encoding.)

### Smoothing
# 2 Problems
#   First are unknown categories.
#   Second are rare categories.

# A solution to these problems is to add smoothing.
# The idea is to blend the in-category average with the overall average.
# Rare categories get less weight on their category average, while missing categories just get the overall average.

# In pseudocode:
#   encoding = weight * in_category + (1 - weight) * overall

# where weight is a value between 0 and 1 calculated from the category frequency.
# An easy way to determine the value for weight is to compute an m-estimate:
#   weight = n / (n + m)

# where n is the total number of times that category occurs in the data.
# The parameter m determines the "smoothing factor".
# Larger values of m put more weight on the overall estimate.

# Example
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

df = pd.read_csv("../../../data/kaggleTutorials/input/movielens1m.csv.zip")
df = df.astype(np.uint8, errors='ignore') # reduce memory footprint
print("Number of Unique Zipcodes: {}".format(df["Zipcode"].nunique()))

# With over 3000 categories, the Zipcode feature makes a good candidate for target encoding,
# and the size of this dataset (over one-million rows) means we can spare some data to create the encoding.

# We'll start by creating a 25% split to train the target encoder.

X = df.copy()
y = X.pop('Rating')

X_encode = X.sample(frac=0.25)
y_encode = y[X_encode.index]
X_pretrain = X.drop(X_encode.index)
y_train = y[X_pretrain.index]

# The category_encoders package in scikit-learn-contrib implements an m-estimate encoder,
# which we'll use to encode our Zipcode feature.

# Create the encoder instance. Choose m to control noise.
encoder = MEstimateEncoder(cols=["Zipcode"], m=5.0)

# Fit the encoder on the encoding split.
encoder.fit(X_encode, y_encode)

# Encode the Zipcode column to create the final training data
X_train = encoder.transform(X_pretrain)

# Let's compare the encoded values to the target to see how informative our encoding might be.
plt.figure(dpi=90)
ax = sns.distplot(y, kde=False, norm_hist=True)
ax = sns.kdeplot(X_train.Zipcode, color='r', ax=ax)
ax.set_xlabel("Rating")
ax.legend(labels=['Zipcode', 'Rating'])
plt.show()

