import pandas as pd

# get data from input file
reviews = pd.read_csv("../../../data/kaggleTutorials/input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)

# get first row
df1 = reviews.iloc[0]

# get first three rows (0, 1, 2)
df2 = reviews.iloc[:3, 0]

# get defined columns
df3 = reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]

# get data bei defined indices and defined columns
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

cols = ['country', 'variety']
df = reviews.loc[:99, cols]

# set index
reviews.set_index("title")

# conditional selection
df4 = reviews.loc[reviews.country == 'Italy']
df5 = reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
df6 = reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
df7 = reviews.loc[reviews.country.isin(['Italy', 'France'])]
df8 = reviews.loc[reviews.price.notnull()]

top_oceania_wines = reviews.loc[
    (reviews.country.isin(['Australia', 'New Zealand']))
    & (reviews.points >= 95)
]

# assign data
reviews['critic'] = 'everyone'
# reverse index
reviews['index_backwards'] = range(len(reviews), 0, -1)


