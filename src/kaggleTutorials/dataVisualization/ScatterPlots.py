import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
insurance_filepath = "../../../data/kaggleTutorials/input/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)

# Print head
print(insurance_data.head())

#### Scatter plots
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

### Regression plot (Scatter plot with regression line)
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

### Scatter plots with colors
# We can use scatter plots to display the relationships between (not two, but...) three variables!
# One way of doing this is by color-coding the points.
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
plt.show()

### lmplot
# This scatter plot shows that while nonsmokers to tend to pay slightly more with increasing BMI, smokers pay MUCH more.
# To further emphasize this fact, we can use the sns.lmplot command to add two regression lines,
# corresponding to smokers and nonsmokers. (You'll notice that the regression line for smokers has a much steeper slope,
# relative to the line for nonsmokers!)
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.show()

### categorical scatter plot
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
plt.show()
