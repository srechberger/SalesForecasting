# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)

# You're working with a dataset composed of bytes. Run the code cell below to print a sample entry.
sample_entry = b'\xa7A\xa6n'
print(sample_entry)
print('data type:', type(sample_entry))

# You notice that it doesn't use the standard UTF-8 encoding.
# Use the next code cell to create a variable new_entry that changes the encoding
# from "big5-tw" to "utf-8".
# new_entry should have the bytes datatype.

before = sample_entry.decode("big5-tw")
print(before)
new_entry = before.encode()
print(new_entry)

# look at the first ten thousand bytes to guess the character encoding
with open('../../../data/police/input/PoliceKillingsUS.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# result: {'encoding': 'ascii', 'confidence': 1.0, 'language': ''}
# Windows-1252???

police_killings = pd.read_csv('../../../data/police/input/PoliceKillingsUS.csv', encoding='Windows-1252')
