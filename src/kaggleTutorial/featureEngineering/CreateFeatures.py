import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

accidents = pd.read_csv("../../../data/else/accidents.csv.zip")
autos = pd.read_csv("../../../data/else/autos.csv")
concrete = pd.read_csv("../../../data/else/concrete.csv")
customer = pd.read_csv("../../../data/else/customer.csv.zip")

### Mathematical Transforms
autos["stroke_ratio"] = autos.stroke / autos.bore

print(autos[["stroke", "bore", "stroke_ratio"]].head())

# The more complicated a combination is, the more difficult it will be for a model to learn,
# like this formula for an engine's "displacement", a measure of its power:
autos["displacement"] = (
    np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
)

# Data visualization can suggest transformations, often a "reshaping" of a feature through powers or logarithms.
# The distribution of WindSpeed in US Accidents is highly skewed, for instance.
# In this case the logarithm is effective at normalizing it:

# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1])
plt.show()

# Counts
# Features describing the presence or absence of something often come in sets,
# the set of risk factors for a disease, say.
# You can aggregate such features by creating a count.

roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

print(accidents[roadway_features + ["RoadwayFeatures"]].head(10))

# You could also use a dataframe's built-in methods to create boolean values.
# In the Concrete dataset are the amounts of components in a concrete formulation.
# Many formulations lack one or more components (that is, the component has a value of 0).
# This will count how many components are in a formulation with the dataframe's built-in greater-than gt method:

components = [ "Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
               "Superplasticizer", "CoarseAggregate", "FineAggregate"]
concrete["Components"] = concrete[components].gt(0).sum(axis=1)

print(concrete[components + ["Components"]].head(10))

# Building-Up and Breaking-Down Features
# ID numbers: '123-45-6789'
# Phone numbers: '(999) 555-0123'
# Street addresses: '8241 Kaggle Ln., Goose City, NV'
# Internet addresses: 'http://www.kaggle.com
# Product codes: '0 36000 29145 2'
# Dates and times: 'Mon Sep 30 07:06:05 2013'

# The str accessor lets you apply string methods like split directly to columns.
# The Customer Lifetime Value dataset contains features describing customers of an insurance company.
# From the Policy feature, we could separate the Type from the Level of coverage:

customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)

print(customer[["Policy", "Type", "Level"]].head(10))

# You could also join simple features into a composed feature
# if you had reason to believe there was some interaction in the combination:

autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]

print(autos[["make", "body_style", "make_and_style"]].head())

###
### GROUP TRANSFORM
###

# Finally we have Group transforms, which aggregate information across multiple rows grouped by some category.
# With a group transform you can create features like: "the average income of a person's state of residence,
# " or "the proportion of movies released on a weekday, by genre."
# If you had discovered a category interaction,
# a group transform over that categry could be something good to investigate.

customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)

print(customer[["State", "Income", "AverageIncome"]].head(10))

# The mean function is a built-in dataframe method, which means we can pass it as a string to transform.
# Other handy methods include max, min, median, var, std, and count.
# Here's how you could calculate the frequency with which each state occurs in the dataset:

customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)

print(customer[["State", "StateFreq"]].head(10))

# You could use a transform like this to create a "frequency encoding" for a categorical feature.

# If you're using training and validation splits, to preserve their independence,
# it's best to create a grouped feature using only the training set and then join it to the validation set.
# We can use the validation set's merge method after creating a unique set of values
# with drop_duplicates on the training set:

# Create splits
df_train = customer.sample(frac=0.5) # die Haelfte der Datensaetze wird ausgewählt
df_valid = customer.drop(df_train.index) # Gesamtmenge minus Trainingsdaten

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

print(df_valid[["Coverage", "AverageClaim"]].head(10))
