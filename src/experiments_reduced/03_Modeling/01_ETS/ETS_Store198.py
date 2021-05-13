import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Load data
train_store198_y_ets = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198.pkl')

# Select Date and Sales columns
features = ['Sales']
train_store198_y_ets = train_store198_y_ets[features]
# Ensure that the index is aggregated by days and unique
train_store198_y_ets = train_store198_y_ets.groupby(train_store198_y_ets.index).Sales.sum()
# Set Datetime Index to Frequency = Day (Frequency = 7)
train_store198_y_ets.index.freq = 'D'


# Fit Models

# These 4 hyperparameters will be automatically tuned if optimized=True:
#    smoothing_level (alpha): the smoothing coefficient for the level.
#    smoothing_slope (beta): the smoothing coefficient for the trend.
#    smoothing_seasonal (gamma): the smoothing coefficient for the seasonal component.
#    damping_slope (phi): the coefficient for the damped trend.

# Simple Exponential Smoothing
ses_model_198 = SimpleExpSmoothing(train_store198_y_ets, initialization_method='estimated')
ses_model_198 = ses_model_198.fit(optimized=True)

# Holt (double)
holt_model_198 = Holt(train_store198_y_ets, initialization_method='estimated')
holt_model_198 = holt_model_198.fit(optimized=True)

# Holt-Winters (triple)
holt_winters_model_198 = ExponentialSmoothing(train_store198_y_ets,
                                              trend="add",
                                              seasonal="add",
                                              seasonal_periods=7,
                                              initialization_method='estimated')
holt_winters_model_198 = holt_winters_model_198.fit(optimized=True)

# Save and store training data
train_store198_y_ets.to_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/03_Store198/train_store198_y_ets.pkl')

# Save and store models
ets_ses_model_198_filename = "../../04_Evaluation/00_Models/ets_ses_model_198.pkl"
with open(ets_ses_model_198_filename, 'wb') as file:
    pickle.dump(ses_model_198, file)

ets_holt_model_198_filename = "../../04_Evaluation/00_Models/ets_holt_model_198.pkl"
with open(ets_holt_model_198_filename, 'wb') as file:
    pickle.dump(holt_model_198, file)
    
ets_holt_winters_model_198_filename = "../../04_Evaluation/00_Models/ets_holt_winters_model_198.pkl"
with open(ets_holt_winters_model_198_filename, 'wb') as file:
    pickle.dump(holt_winters_model_198, file)

