# Imports
import pandas as pd
import numpy as np
# ML
from MacroRandomForest import MRF

# Generate simulated data
n = 1000
vars = 15
data = pd.DataFrame(index=range(n))
time = np.arange(n)

# df 
x_vars = pd.DataFrame(index=range(n))

# Add independent variables
for i in range(vars):
    x_vars['x' + str(i)] = np.random.normal(0, 1, n)

# Add dependent variable
weights = np.random.normal(0, 0.5, vars)
weights[0] = 0.8
weights[1] = 0.8
weights[2] = 0.8
dependent_data = np.dot(x_vars, weights) + np.random.normal(0, 1, n)
data['y'] = dependent_data

# Create dataset
data = pd.concat([data, x_vars], axis=1)

# Dependent variables
y_var = 'y'
y_pos = data.columns.get_loc(y_var)

# Exogenous variables
S_vars = [x for x in data.columns if x != y_var]
S_pos = np.array([data.columns.get_loc(x) for x in S_vars])

# Independent variables
x_vars = ['x0', 'x1', 'x2']
x_pos = np.array([data.columns.get_loc(x) for x in x_vars])

# Predict last 50 observations
oos_pos = np.arange(len(data) - 50 , len(data)) # lower should be oos start, upper the length of your dataset

# Print
print(f"Data:\n{data.head()}")
print(f"y_pos: {y_pos}")
print(f"S_pos: {S_pos}")
print(f"x_pos: {x_pos}")
print(f"oos_pos: {oos_pos}")

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

print(output)
