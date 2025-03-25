import pandas as pd
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline

# =============================================================================
# Load the FRED-MD Data
# =============================================================================
df = pd.read_csv("data.csv", index_col='sasdate', parse_dates=True)
target_series = df['CPIAUCSL']

# =============================================================================
# Prepare Train-Test Split and Format Data for Chronos
# =============================================================================
# Split the target series into training and test parts.
# Here we reserve the last 12 months as test data.
train_series = target_series.iloc[:-12]
test_series = target_series.iloc[-12:]

# Create the Chronos model
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cpu",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)

quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(df["CPIAUCSL"].values, dtype=torch.bfloat16),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Visualize the forecast
forecast_index = range(len(df), len(df) + 12)
low, median, high = quantiles[0, :, 0], quantiles[0, :, 1], quantiles[0, :, 2]

plt.figure(figsize=(8, 4))
plt.plot(df["CPIAUCSL"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()
