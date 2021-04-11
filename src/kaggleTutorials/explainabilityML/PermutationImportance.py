#In this lesson, we'll focus on permutation importance. Compared to most other approaches, permutation importance is:
#    fast to calculate,
#    widely used and understood, and
#    consistent with properties we would want a feature importance measure to have.

# Permutation importance is calculated after a model has been fitted.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

data = pd.read_csv('../../../data/kaggleTutorials/input/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names=val_X.columns.tolist())))

# The first number in each row shows how much model performance decreased with a random shuffling (in this case,
# using "accuracy" as the performance metric).
#
# Like most things in data science, there is some randomness to the exact performance change from a shuffling a column.
# We measure the amount of randomness in our permutation importance calculation
# by repeating the process with multiple shuffles.
# The number after the ± measures how performance varied from one-reshuffling to the next.
#
# You'll occasionally see negative values for permutation importances.
# In those cases, the predictions on the shuffled (or noisy) data happened to be more accurate than the real data.
# This happens when the feature didn't matter (should have had an importance close to 0),
# but random chance caused the predictions on shuffled data to be more accurate.
# This is more common with small datasets, like the one in this example, because there is more room for luck/chance.
#
# In our example, the most important feature was Goals scored.
# That seems sensible. Soccer fans may have some intuition about
# whether the orderings of other variables are surprising or not.
