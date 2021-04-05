# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

# read in our data
earthquakes = pd.read_csv('../../../data/earthquake/input/database.csv.zip')

# set seed for reproducibility
np.random.seed(0)

# dtype of Date
print(earthquakes['Date'].dtype)

# show different Date-Formats
print(earthquakes[3378:3383])

# check how many rows are affected
date_lengths = earthquakes.Date.str.len()
print(date_lengths.value_counts())
# result: only 3 rows

# find the relevant indices
indices = np.where([date_lengths == 24])[1]
print('Indices with corrupted data:', indices)
print(earthquakes.loc[indices])

# Given all of this information,
# it's your turn to create a new column "date_parsed" in the earthquakes dataset that has correctly parsed dates in it.

# Transform corrupted data manually
earthquakes.loc[3378, "Date"] = "02/23/1975"
earthquakes.loc[7512, "Date"] = "04/28/1985"
earthquakes.loc[20650, "Date"] = "03/13/2011"

# Parse dates
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y")

### Select the day of the month
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
print(day_of_month_earthquakes)

### Select month
month_earthquakes = earthquakes['date_parsed'].dt.month
print(month_earthquakes)

### Select year
year_earthquakes = earthquakes['date_parsed'].dt.year
print(year_earthquakes)

### Plot the day of the month to check the date parsing
# Hint: Remove the missing values, and then use sns.distplot() as follows:

# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.histplot(day_of_month_earthquakes, kde=False, bins=31)
plt.show()
