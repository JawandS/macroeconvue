# Dependencies
import pandas as pd
import numpy as np
# Stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# ML Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Load and process data (FRED-MD)
# Load data (FRED-MD)
df = pd.read_csv(
    '/home/js/macroeconvue/nowcasting/current.csv',
    index_col='sasdate'
    )
# Drop target variable (CPIAUCSL)
target = df['CPIAUCSL']
df = df.drop(columns=['CPIAUCSL'])