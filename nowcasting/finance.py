# Dependencies
import macroecon_tools as mt
import numpy as np
# ML Dependencies
from MacroRandomForest import MRF
import matplotlib.pyplot as plt

# Get data from FRED 
srcs = {
    "SP500": "S&P 500",
    "UNRATE": "Unemp Rate",
    "FEDFUNDS": "Fed Funds Rate",
    "CPIAUCSL": "CPI",
    "GDP": "GDP",
    "M2SL": "Money Supply (M2)",
    "UMCSENT": "Consumer Sentiment",
    "VIXCLS": "VIX Volatility",
    "RECPROUSM156N": "P of Recession",
    "BAA10Y": "Yield Spread",
    "PAYEMS": "Nonfarm Payrolls",
    "INDPRO": "Industrial Production",
    "HSN1F": "New Home Sales",
    "PCE": "Personal Cons Exp",
    "BUSINV": "Business Inventories",
    "ISRATIO": "Inv/Sales Ratio",
    "EXPGS": "Exports",
    "IMPGS": "Imports",
    "TCU": "Capacity Utilization",
    "WTISPLC": "WTI Crude Oil Price"
}


# fetch
data = mt.get_fred(srcs)

data = data.aggregate('QS', 'mean').df.dropna(how='any')
print(data.head(2))

# Dependent variable
y_var = "S&P 500"
y_pos = data.columns.get_loc(y_var)

# Exogenous variables
# s_vars = ["UNRATE", "FEDFUNDS", "RPI", "UMCSENTX", "BOGMBASE", "BUSLOANS", "GS1", "S&P 500"]
s_vars = [col for col in data.columns if col != y_var]
S_pos = np.array([data.columns.get_loc(v) for v in s_vars])

# Independent variables
x_vars = ["Unemp Rate", "Fed Funds Rate", "CPI", "GDP", "Consumer Sentiment"]
x_pos = np.array([data.columns.get_loc(v) for v in x_vars])

# Predict last 4 observations (1 year) (out of sample position)
data_len = len(data)
num_oos = 4
oos_pos = np.arange(data_len - num_oos, data_len)
# copy test data
test = data[[y_var]].copy().iloc[oos_pos]

# Build model
model = MRF.MacroRandomForest(
    data = data,
    x_pos = x_pos,
    oos_pos = oos_pos,
    S_pos = S_pos,
    y_pos = y_pos,
)

# Run model
output = model._ensemble_loop()

# Get predictions
forecasts = output['pred']
# betas = output['betas']

# Map forecasts to date using data index
forecasts.dropna(inplace=True)
forecasts.index = data.index[oos_pos]

# Plot
fig, ax = plt.subplots()
plt.rcParams['figure.figsize'] = (20, 8)

# Plotting actual versus original
ax.plot(data[0], label = 'Actual', linewidth = 3, color ='mediumseagreen', linestyle = '--')
ax.plot(forecasts, color = 'lightcoral', linewidth = 3, label = "MRF Ensemble")

ax.legend(fontsize = 15)
ax.set_ylabel("Value", fontsize = 15)
ax.grid()
ax.set_xlabel(r"$t$", fontsize = 16)
ax.set_title("OOS predictions of MRF", fontsize = 15)

# Save plot
plt.savefig("stock.png", dpi = 300)
