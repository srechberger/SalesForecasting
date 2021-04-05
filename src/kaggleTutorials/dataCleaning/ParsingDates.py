# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# read in our data
landslides = pd.read_csv('../../../data/kaggleTutorials/input/catalog.csv')

# set seed for reproducibility
np.random.seed(0)

# Check the data type of our date column
print(landslides.head())

# print the first few rows of the date column
print(landslides['date'].head())

# If you check the pandas dtype documentation here, you'll notice that there's also a specific datetime64 dtypes.
# Because the dtype of our column is object rather than datetime64,
# we can tell that Python doesn't know that this column contains dates.

# check the data type of our date column
print(landslides['date'].dtype)

################################################################################
############### Convert our date columns to datetime ###########################
################################################################################

# There are lots of possible parts of a date, but the most common are
# %d for day,
# %m for month,
# %y for a two-digit year and
# %Y for a four digit year.

# Some examples:
# 1/17/07 has the format "%m/%d/%y"
# 17-1-2007 has the format "%d-%m-%Y"

# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# Now when I check the first few rows of the new column, I can see that the dtype is datetime64.
# I can also see that my dates have been slightly rearranged so that
# they fit the default order datetime objects (year-month-day).

# print the first few rows
print(landslides['date_parsed'].head())

# What if I run into an error with multiple date formats?
# While we're specifying the date format here, sometimes you'll run into an error when there are multiple date formats
# in a single column. If that happens, you have have pandas try to infer what the right date format should be. Y
# ou can do that like so:
# landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

# Why don't you always use infer_datetime_format = True?
# There are two big reasons not to always have pandas guess the time format.
# The first is that pandas won't always been able to figure out the correct date format,
# especially if someone has gotten creative with data entry.
# The second is that it's much slower than specifying the exact format of the dates.

################################################################################
############### Select the day of the month ####################################
################################################################################
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())

################################################################################
############### Plot the day of the month to check the date parsing ############
################################################################################
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
plt.show()
