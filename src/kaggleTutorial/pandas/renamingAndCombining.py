import pandas as pd

# get data from input file
reviews = pd.read_csv("../../../data/wineReviews/input/winemag-data-130k-v2.csv", index_col=0)
pd.set_option('max_rows', 5)

#### RENAMING ####

# rename()
# which lets you change index names and / or column names.
# For example, to change the points column in our dataset to score, we would do:
reviews.rename(columns={'points': 'score'})
reviews.rename(columns=dict(region_1='region', region_2='locale'))

# rename specific indices
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})

# rename_axis()
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

#### COMBINING ####
# join data tables and files
# methods: concat(), join(), merge()

# concat()
# The simplest combining method is concat().
# Given a list of elements, this function will smush those elements together along an axis.
canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])

# join()
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])
left.join(right, lsuffix='_CAN', rsuffix='_UK')

# example
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")

left = powerlifting_meets.set_index(['MeetID'])
right = powerlifting_competitors.set_index(['MeetID'])
powerlifting_combined = left.join(right, lsuffix='_ME', rsuffix='_CO')
# or
powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))
