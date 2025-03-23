# Machine Learning Model to Forecast CPAIUCSL

# DEPENDENCIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lars
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. LOAD DATA (FRED-MD)
# Assuming 'data.csv' has a date index column named 'sasdate'
data = pd.read_csv('data.csv', index_col='sasdate', parse_dates=True)
target = 'CPAIUCSL'  # Target variable

# Separate features and target
y = data[target]
X = data.drop(columns=[target])

# Split data into training and testing sets (e.g., 80% train, 20% test)
train_size = int(len(data) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 2. RUN LARS
# Least Angle Regression to get an initial forecasting model
lars_model = Lars(n_nonzero_coefs=10)  # You can adjust the max number of coefficients
lars_model.fit(X_train, y_train)
lars_pred = lars_model.predict(X_test)
lars_mse = mean_squared_error(y_test, lars_pred)
print("LARS Mean Squared Error:", lars_mse)

# 3. RUN PCA
# Reduce the dimensionality of features for the LSTM model.
# The number of components is adjustable based on data characteristics.
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 4. RUN LSTM
# Prepare data for LSTM: reshape to [samples, timesteps, features].
# Here we assume a single timestep per sample.
X_train_lstm = X_train_pca.reshape((X_train_pca.shape[0], 1, X_train_pca.shape[1]))
X_test_lstm = X_test_pca.reshape((X_test_pca.shape[0], 1, X_test_pca.shape[1]))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the LSTM model
history = lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

# Predict using the LSTM model
lstm_pred = lstm_model.predict(X_test_lstm)
lstm_mse = mean_squared_error(y_test, lstm_pred)
print("LSTM Mean Squared Error:", lstm_mse)

# 5. LOOK AT OUTPUT
# Plot true vs predicted values for both models
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='True', color='black')
plt.plot(y_test.index, lars_pred, label='LARS Predictions', linestyle='--')
plt.plot(y_test.index, lstm_pred, label='LSTM Predictions', linestyle=':')
plt.legend()
plt.title("Forecasting CPAIUCSL")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()
