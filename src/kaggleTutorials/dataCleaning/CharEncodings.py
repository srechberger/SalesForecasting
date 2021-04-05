import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)

# UTF-8 is the standard text encoding.
# All Python code is in UTF-8 and, ideally, all your data should be as well.
# It's when things aren't in UTF-8 that you run into trouble.

# start with a string
before = "This is the euro symbol: €"
print(before)

# check to see what datatype it is
print(type(before))

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")
print(after)

# check the type
print(type(after))

# convert it back to utf-8
print(after.decode("utf-8"))

##### Check encoding of input data #####

# look at the first ten thousand bytes to guess the character encoding
with open("../../../data/kaggleTutorials/input/ks-projects-201612.csv.zip", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# {'encoding': 'Windows-1252', 'confidence': 0.73, 'language': ''}

# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../../../data/kaggleTutorials/input/ks-projects-201612.csv.zip", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()

# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")