# Dependencies
import pandas as pd

# Load the dataset
data = pd.read_csv('data.csv', index_col='sasdate', parse_dates=True)
target = 'CPIAUCSL'

# Metadata
num_months = 6
lags = 5

