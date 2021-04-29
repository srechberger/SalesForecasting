import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load data
y = pd.read_pickle('../../../../data/rossmann/intermediate/03_SalesModelingBase/sales_store708.pkl')
# Select Date and Sales columns
features = ['Date', 'Sales']
y = y[features]
# Set index 'Dates'
y = y.set_index('Date')

# The 'MS' string groups the data in buckets by start of the month
y = y.groupby('Date').Sales.sum()
y = y.resample('MS').sum()

# Plot the data
y.plot(figsize=(15, 6))
plt.show()

# ---------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Parameter Selection -----------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# ARIMA (AutoRegressive Integrated Moving Average)
# There are three distinct integers (p, d, q) that are used to parametrize ARIMA models.
# Together these three parameters account for seasonality, trend, and noise in datasets:
#    p is the auto-regressive part of the model.
#       It allows us to incorporate the effect of past values into our model.
#    d is the integrated part of the model.
#       This includes terms in the model that incorporate the amount of differencing
#       (the number of past time points to subtract from the current value) to apply to the time series.
#    q is the moving average part of the model.
#       This allows us to set the error of our model as a linear combination of the error values
#       observed at previous time points in the past.

# When dealing with seasonal effects, we make use of the seasonal ARIMA,
# which is denoted as ARIMA(p,d,q)(P,D,Q)s.
#
# Here, (p, d, q) are the non-seasonal parameters described above, while (P, D, Q) follow the same definition
# but are applied to the seasonal component of the time series.
# The term s is the periodicity of the time series (4 for quarterly periods, 12 for yearly periods, etc.).

# grid search - to iteratively explore different combinations of parameters
# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# The code below iterates through combinations of parameters and
# uses the SARIMAX function from statsmodels to fit the corresponding Seasonal ARIMA model.

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# Recommended settings for data
# ARIMA(0, 1, 1)x(0, 1, 1, 12)12 - AIC:89.20250221831131
# ARIMA(0, 1, 1)x(1, 1, 1, 12)12 - AIC:91.2057730614527
# ARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:91.35144124908021
# ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:92.59430722687175
# ...
# --> set all parameters to 1 (very close results)

# ---------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Fitting ARIMA -----------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# SARIMAX model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 10))
plt.show()

# ---------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Validating Forecasts ----------------------------------------
# ---------------------------------------------------------------------------------------------------------------

pred = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['2013':].plot(label='Actual', figsize=(17, 10))
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')
plt.legend()

plt.show()

y_forecasted = pred.predicted_mean
y_truth = y['2015-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# ---------------

pred_dynamic = results.get_prediction(start=pd.to_datetime('2015-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['2013':].plot(label='Actual', figsize=(17, 10))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2015-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('Sales')

plt.legend()
plt.show()

# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['2015-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# ---------------------------------------------------------------------------------------------------------------
# ------------------------------------------------- Producing and Visualizing Forecasts -------------------------
# ---------------------------------------------------------------------------------------------------------------
# Get forecast 12 steps ahead in future
pred_uc = results.get_forecast(steps=12)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='Actual', figsize=(17, 10))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Sales')

plt.legend()
plt.show()
