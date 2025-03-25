# Script to create the LSTM model for data prediction

def process_data():
    import pandas as pd

    
    # Load the data
    df = pd.read_csv(
        '/home/js/macroeconvue/nowcasting/current.csv',
        index_col='sasdate',
        parse_dates=True
    )
    
    transformed_data = {}
    transformation_codes = df.iloc[0]  # Assuming the first row contains the codes
    data = df.iloc[1:]  # The actual data starts from the second row

    for column in data.columns:
        code = transformation_codes[column]
        transformed_data[column] = transform_series(data[column], code)

    df = pd.DataFrame(transformed_data).dropna()

    # Save the data
    df.to_csv('/home/js/macroeconvue/nowcasting/processed_data.csv')