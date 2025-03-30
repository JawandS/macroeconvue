import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Load and process data (FRED-MD)
df = pd.read_csv(
    '/home/js/macroeconvue/nowcasting/current.csv',
    index_col='sasdate',
    parse_dates=True
)

def transform_series(series, code):
    if code == 1:
        return series  # No transformation
    elif code == 2:
        return series.diff().dropna()  # First difference
    elif code == 3:
        return series.diff().diff().dropna()  # Second difference
    elif code == 4:
        return np.log(series).dropna()  # Logarithm
    elif code == 5:
        return np.log(series).diff().dropna()  # First difference of logarithm
    elif code == 6:
        return np.log(series).diff().diff().dropna()  # Second difference of logarithm
    elif code == 7:
        return series.pct_change().dropna()  # Percentage change
    else:
        raise ValueError(f"Unknown transformation code: {code}")

transformed_data = {}
transformation_codes = df.iloc[0]  # Assuming the first row contains the codes
data = df.iloc[1:]  # The actual data starts from the second row

for column in data.columns:
    code = transformation_codes[column]
    transformed_data[column] = transform_series(data[column], code)

df = pd.DataFrame(transformed_data).dropna()

# Normalize the data
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Capture target variable (CPIAUCSL)
target = df_scaled['CPIAUCSL']
df_scaled = df_scaled.drop(columns=['CPIAUCSL'])

# Use existing model if available
model = None
try:
    from tensorflow.keras.models import load_model
    model = load_model('/home/js/macroeconvue/nowcasting/lstm_model.h5')
    print("Loaded existing model.")
except Exception as e:
    print(f"ERROR: {e}")
    print("No existing model found, training a new one.")

if not model:
    # Apply PCA
    pca = PCA(n_components=0.90)
    data_pca = pca.fit_transform(df_scaled)

    def create_sequences(data, target, time_steps=12):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])
            y.append(target[i+time_steps])
        return np.array(X), np.array(y)

    # Prepare data for LSTM
    sequence_length = 12  # Use past 12 months for prediction
    X, y = create_sequences(data_pca, target.values, sequence_length)

    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, X.shape[2]), kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

# Generate predictions
y_pred = model.predict(X_test)

# Plot original vs predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual CPIAUCSL', color='blue')
plt.plot(y_pred, label='Predicted CPIAUCSL', color='red', linestyle='dashed')
plt.legend()
plt.title('Actual vs Predicted CPIAUCSL')
# Save
plt.savefig('/home/js/macroeconvue/nowcasting/predictions.png')
