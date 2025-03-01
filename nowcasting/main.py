# referencing: https://github.com/baptiste-meunier/NowcastingML_3step/blob/main/11-Github/Main.R
# referencing: https://mrf-web.readthedocs.io/en/latest/index.html

# Dependencies
import pandas as pd
import macroecon_tools as mt

### Data Preprocessing ###
# Import data
data = pd.read_csv('/home/js/macroeconvue/nowcasting/fred-md.csv').s
# Adjust datetime
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')
# Set date as index
data.set_index('date', inplace=True)
# Drop missing values
data.dropna(inplace=True)

### Data Transformation ###
data = mt.TimeseriesTable(data)
print(data)
