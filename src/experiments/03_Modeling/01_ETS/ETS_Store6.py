# The golden rule: Stationarity
#
# Before going any further into our analysis, our series has to be made stationary.
# Stationarity is the property of exhibiting constant statistical properties (mean, variance, autocorrelation, etc.).
# If the mean of a time-series increases over time, then it’s not stationary.
# Transforms used for stationarizing data:
#
# De-trending : We remove the underlying trend in the series.
# This can be done in several ways, depending on the nature of data :
#     - Indexed data: data measured in currencies are linked to a price index or related to inflation.
#     Dividing the series by this index (ie deflating) element-wise is therefore the solution to de-trend the data.
#     - Non-indexed data: is it necessary to estimate if the trend is constant, linear or exponential.
#     The first two cases are easy, for the last one it is necessary to estimate a growth rate (inflation or deflation)
#     and apply the same method as for indexed data.
#     Differencing : Seasonal or cyclical patterns can be removed by substracting periodical values.
#     If the data is 12-month seasonal, substracting the series with a 12-lag difference series will give
#     a “flatter” series
#     Logging : in the case where the compound rate in the trend is not due to a price index
#     (ie the series is not measured in a currency), logging can help linearize a series with an exponential trend
#     (recall that log(exp(x)) = x). It does not remove an eventual trend whatsoever, unlike deflation.

# -------------------------------------------------------------------------------------------------------------------

# 01 Checking Stationarity
# Plotting rolling statistics
#
# Plotting rolling means and variances is a first good way to visually inspect our series.
# If the rolling statistics exhibit a clear trend (upwards or downwards) and show varying variance
# (increasing or decreasing amplitude), then you might conclude that the series is very likely not to be stationary.


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('Solarize_Light2')

# Load data
y = pd.read_pickle('../../../../data/rossmann/intermediate/store20.pkl')
# Select Date and Sales columns
features = ['Date', 'Sales']
y = y[features]
# Set index 'Dates'
y = y.set_index('Date')

# Split data in train and test data
train = y.iloc[:-10, :]
test = y.iloc[-10:, :]

# Set initial predictions (actual values)
pred = test.copy()

# Plot data
y.plot(figsize=(12, 3))
plt.title('Store 6 Sales Data')

y['z_Sales'] = (y['Sales'] - y.Sales.rolling(window=150).mean()) / y.Sales.rolling(window=150).std()
y['zp_Sales'] = y['z_Sales'] - y['z_Sales'].shift(150)

plt.show()

def plot_rolling(y):
    fig, ax = plt.subplots(3, figsize=(12, 9))
    ax[0].plot(y.index, y.Sales, label='raw data')
    ax[0].plot(y.Sales.rolling(window=150).mean(), label="rolling mean");
    ax[0].plot(y.Sales.rolling(window=150).std(), label="rolling std (x10)");
    ax[0].legend()

    ax[1].plot(y.index, y.z_Sales, label="de-trended data")
    ax[1].plot(y.z_Sales.rolling(window=150).mean(), label="rolling mean");
    ax[1].plot(y.z_Sales.rolling(window=150).std(), label="rolling std (x10)");
    ax[1].legend()

    ax[2].plot(y.index, y.zp_Sales, label="12 lag differenced de-trended data")
    ax[2].plot(y.zp_Sales.rolling(window=150).mean(), label="rolling mean");
    ax[2].plot(y.zp_Sales.rolling(window=150).std(), label="rolling std (x10)");
    ax[2].legend()

    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()

plot_rolling(y)
