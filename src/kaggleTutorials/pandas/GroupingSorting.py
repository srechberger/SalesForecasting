import pandas as pd

# get data from input file
reviews = pd.read_csv("../../../data/kaggleTutorials/input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)

##### GROUPING METHODS #####

# count rows grouped by points
print(reviews.groupby('points').points.count())
# or
print(reviews.groupby('points').size())

# get cheapest wine grouped by points
print(reviews.groupby('points').price.min())

# get best wine per country and province
print(reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]))

# agg()
# which lets you run a bunch of different functions on your DataFrame simultaneously.
# For example, we can generate a simple statistical summary of the dataset as follows:
print(reviews.groupby(['country']).price.agg([len, min, max]))

# Multi-indexes
countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
print(countries_reviewed)

# Datatype of multi-indexes
mi = countries_reviewed.index
type(mi)

# in general the multi-index method you will use most often is the one
# for converting back to a regular index, the reset_index() method:
print(countries_reviewed.reset_index())

##### SORTING METHODS #####

# sort_values()
# sort by single value
countries_reviewed = countries_reviewed.reset_index()
countries_reviewed.sort_values(by='len')
countries_reviewed.sort_values(by='len', ascending=False)

# sort by multiple values
countries_reviewed.sort_values(by=['country', 'len'])
country_variety_counts = reviews.groupby(['country', 'variety']).size().sort_values(ascending=False)

# sort_index()
# sort by index
countries_reviewed.sort_index()

# Create a Series whose index is wine prices and
# whose values is the maximum number of points a wine costing that much was given in a review
best_rating_per_price = reviews.groupby('price')['points'].max().sort_index()
print(best_rating_per_price)


