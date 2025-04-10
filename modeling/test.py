# Dependencies
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error

# # Load data (FRED-MD)
# df = pd.read_csv('current.csv')

# # Remove the first row (transformation codes)
# transformation_codes = df.iloc[0]  # Transformation codes can be applied if needed
# df = df.iloc[1:]

# # Set the first column as the index and datetime
# df.set_index(df.columns[0], inplace=True)
# df.index = pd.to_datetime(df.index)

# # Dropna
# data = df.dropna()

# # Create train data and target
# target = (data['CPIAUCSL'].diff(12) / data['CPIAUCSL'].shift(12)) * 100
# target = target.shift(-12).dropna()
# data = data.loc[target.index]
# train = data.dropna()

# print("Data Shape:", data.shape)

# Update modeling data
data = pd.read_csv('modeling/var.csv')
# add a new column in between the third and fourth column
data['normalized'] = False
data.insert(3, 'normalized', data.pop('normalized'))
data['normalized'] = data['normalized'].astype(bool)
data.to_csv('modeling/var.csv', index=False)