import pandas as pd

##### COMMON FUNCTIONS #####

# get data from input file
reviews = pd.read_csv("../../../data/kaggleTutorials/input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)

# summary function for numeric values
print(reviews.points.describe())
print('Mean points: ' + str(reviews.points.mean()))
print('Median points: ' + str(reviews.points.median()))

# summary function for string values
print(reviews.taster_name.describe())
# unique values
print(reviews.taster_name.unique())

# count unique values
print(reviews.taster_name.value_counts())

##### COMMON MAPS #####

# Note that map() and apply() return new, transformed Series and DataFrames, respectively.
# They don't modify the original data they're called on.

# map()
# In data science we often have a need for creating new representations from existing data,
# or for transforming data from the format it is in now to the format that we want it to be in later.
# For example, suppose that we wanted to remean the scores the wines received to 0.
review_points_mean = reviews.points.mean()
reviews.points.map(lambda p: p - review_points_mean) # Berechnung: points - durchschnitt

# The function you pass to map() should expect a single value from the Series (a point value, in the above example),
# and return a transformed version of that value.
# map() returns a new Series where all the values have been transformed by your function.

# apply()
# is the equivalent method if we want to transform a whole DataFrame by calling a custom method on each row.
def remean_points(row):
    row.points = row.points - review_points_mean
    return row

reviews.apply(remean_points, axis='columns')

# Standard Python operators >, <, ==, and so on
# Standard operators are faster than map() or apply() because they uses speed ups built into pandas
print(reviews.country + " - " + reviews.region_1)

# Create a variable bargain_wine with the title of the wine with the highest points-to-price ratio in the dataset.
bargain_idx = (reviews.points / reviews.price).idxmax()
bargain_wine = reviews.loc[bargain_idx, 'title']

# Create a Series `descriptor_counts` counting how many times each of these two words appears
# in the `description` column in the dataset.
n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
print(descriptor_counts)

# Star classification
# A score of 95 or higher counts as 3 stars,
# a score of at least 85 but less than 95 is 2 stars.
# Any other score is 1 star.
# Canada should automatically get 3 stars

def stars(row):
    if row.country == 'Canada':
        return 3
    elif row.points >= 95:
        return 3
    elif row.points >= 85:
        return 2
    else:
        return 1

star_ratings = reviews.apply(stars, axis='columns')
print(star_ratings)
