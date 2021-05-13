import itertools
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load data
y_897 = pd.read_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897.pkl')
# Select Date and Sales columns
features = ['Sales']
y_897 = y_897[features]

# Set Datetime Index to Frequency = Day
y_897.index = pd.DatetimeIndex(y_897.index).to_period('D')
# Ensure that the index is aggregated by days and unique
y_897 = y_897.groupby(y_897.index).Sales.sum()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

# Hyperparameter tuning for ARIMA
#
# In order to choose the best combination of the above parameters, we’ll use a grid search.
# The best combination of parameters will give the lowest Akaike information criterion (AIC) score.
# AIC tells us the quality of statistical models for a given set of data.

# Determing p,d,q combinations with AIC scores.
rows = []

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y_897,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            aic_text = 'ARIMA{}x{}7 - AIC:{}'.format(param, param_seasonal, results.aic)
            rows.append([results.aic, aic_text])
        except:
            continue

aic_results = pd.DataFrame(rows, columns=["AIC", "Params"])
aic_results = aic_results.sort_values(by='AIC', ascending=True)
print(aic_results)
# Best AIC
# ARIMA(1, 1, 1)x(1, 1, 1, 7)7 - AIC:774.04

# Fitting the data to ARIMA model
model_sarima = sm.tsa.statespace.SARIMAX(y_897,
                                         order=(1, 1, 1),
                                         seasonal_order=(1, 1, 1, 7),
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)

sarima_model_897 = model_sarima.fit()

print(sarima_model_897.summary().tables[1])

# Checking diagnostic plots
sarima_model_897.plot_diagnostics(figsize=(10, 10))
plt.show()

# Save and store training data
y_897.to_pickle('../../../../data/rossmann/intermediate/04_SalesModelingReduced/04_Store897/train_store897_y_sarima.pkl')

# Save and store model
model_897_filename = "../../04_Evaluation/00_Models/sarima_model_897.pkl"
with open(model_897_filename, 'wb') as file:
    pickle.dump(sarima_model_897, file)
