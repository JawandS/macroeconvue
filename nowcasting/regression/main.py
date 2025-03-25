# Perform a regression

# Dependencies
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv', index_col='sasdate')
# Set index as datetime
data.index = pd.to_datetime(data.index)
# Standardize
df = (data - data.mean()) / data.std()

# Create DFM
model = sm.tsa.DynamicFactor(
    endog=df,       # Input time series
    k_factors=1,    # Number of latent factors
    factor_order=2  # Number of lags in the factor model
)

# Fit the model
res = model.fit(method='bfgs')

# Extract estimated factors
df_factors = res.factors.filtered

# Plot the extracted factor
plt.figure(figsize=(10, 4))
plt.plot(df_factors, label="Estimated Factor")
plt.title("Dynamic Factor Model - Extracted Factor")
plt.legend()
plt.save("df_factors_plot.png")

# Make forecasts
n_forecast = 12  # Forecast next 12 months
forecast = res.forecast(steps=n_forecast)

# Plot forecast
plt.figure(figsize=(10, 4))
plt.plot(df.index, df.iloc[:, 0], label="Actual Data")
plt.plot(pd.date_range(df.index[-1], periods=n_forecast+1, freq='M')[1:], forecast, label="Forecast", linestyle="dashed")
plt.title("DFM Forecast")
plt.legend()
plt.save("forecast_plot.png")


