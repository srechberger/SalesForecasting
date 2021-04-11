import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

# Path of the file to read
iris_filepath = "../../../data/kaggleTutorials/input/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
print(iris_data.head())

##### Histograms
# Histogram
# kde=False is something we'll always provide when creating a histogram,
# as leaving it out will create a slightly different plot.
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
plt.show()

##### Density plots
# KDE plot
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
plt.show()

##### 2D KDE plots
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
plt.show()

##### Color-coded plots
# Paths of the files to read
iris_set_filepath = "../../../data/kaggleTutorials/input/iris_setosa.csv"
iris_ver_filepath = "../../../data/kaggleTutorials/input/iris_versicolor.csv"
iris_vir_filepath = "../../../data/kaggleTutorials/input/iris_virginica.csv"

# Read the files into variables
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()

plt.show()


# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")

plt.show()
