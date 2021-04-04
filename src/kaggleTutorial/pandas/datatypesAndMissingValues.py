import pandas as pd

# get data from input file
reviews = pd.read_csv("../../../data/wineReviews/input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)

# datatypes
# The data type for a column in a DataFrame or a Series is known as the dtype.
# You can use the dtype property to grab the type of a specific column.
# For instance, we can get the dtype of the price column in the reviews DataFrame:
print(reviews.price.dtype)
# dtype('float64')

# get datatypes of all cols
print(reviews.dtypes)
# string = object, integer = int64, float = float64

# astype()
# transform the points column from its existing int64 data type into a float64 data type
reviews.points.astype('float64')
# transform the points to string
point_strings = reviews.points.astype('str')

# missing data
# Entries missing values are given the value NaN, short for "Not a Number".
# For technical reasons these NaN values are always of the float64 dtype.
print(reviews[pd.isnull(reviews.country)])
print(reviews[pd.notnull(reviews.country)])

# count null values for price column
# if we sum a boolean series, True is treated as 1 and False as 0
n_missing_prices = pd.isnull(reviews.price).sum()

# fillna()
# Replacing missing values is a common operation.
# Pandas provides a really handy method for this problem: fillna().
# fillna() provides a few different strategies for mitigating such data.
# For example, we can simply replace each NaN with an "Unknown":
reviews.region_2.fillna("Unknown")

# example
#  Create a Series counting the number of times each value occurs in the region_1 field.
#  This field is often missing data, so replace missing values with Unknown.
#  Sort in descending order.
reviews_per_region = reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)


# replace()
# Replace a non-null value
# For example, suppose that since this dataset was published,
# reviewer Kerin O'Keefe has changed her Twitter handle from @kerinokeefe to @kerino.
reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino")
