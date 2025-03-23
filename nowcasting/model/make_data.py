# Dependencies
import pandas as pd


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
    