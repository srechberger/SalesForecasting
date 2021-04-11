import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
museum_filepath = "../../../data/kaggleTutorials/input/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)

# Print the last five rows of the data
museum_data.tail()

# Set the width and height of the figure
plt.figure(figsize=(12,6))
# Line chart showing the number of visitors to each museum over time
sns.lineplot(data=museum_data)
# Add title
plt.title("Monthly Visitors to Los Angeles City Museums")

# Set the width and height of the figure
plt.figure(figsize=(12,6))
# Add title
plt.title("Monthly Visitors to Avila Adobe")
# Line chart showing the number of visitors to Avila Adobe over time
sns.lineplot(data=museum_data['Avila Adobe'])
# Add label for horizontal axis
plt.xlabel("Date")

plt.show()
