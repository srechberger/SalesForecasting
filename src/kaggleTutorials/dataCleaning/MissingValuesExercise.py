# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
sf_permits = pd.read_csv('../../../data/kaggleTutorials/input/Building_Permits.csv.zip')

# set seed for reproducibility
np.random.seed(0)

# inspect first 5 rows
sf_permits.head()

### How many missing data points do we have?
# get the number of missing data points per column
missing_values_count = sf_permits.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(sf_permits.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100

### Drop missing values: rows
# remove all the rows that contain a missing value
sf_permits.dropna()

### Drop missing values: columns
# remove all columns with at least one missing value
sf_permits_with_na_dropped = sf_permits.dropna(axis=1)

# calculate number of dropped columns
cols_in_original_dataset = sf_permits.shape[1]
cols_in_na_dropped = sf_permits_with_na_dropped.shape[1]
dropped_columns = cols_in_original_dataset - cols_in_na_dropped

### Fill in missing values automatically
# Try replacing all the NaN's in the sf_permits data with the one that comes directly after it
# and then replacing any remaining NaN's with 0.
# Set the result to a new DataFrame sf_permits_with_na_imputed.
sf_permits_with_na_imputed = sf_permits.fillna(method='bfill', axis=0).fillna(0)
