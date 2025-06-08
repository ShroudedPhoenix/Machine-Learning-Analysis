import pandas as pd
import os
import datetime as dt
import numpy as np


def load_csv(path):
    df = pd.read_csv(path, usecols=[0, 1, 2, 3, 4])

    raw = df['time'].astype(str).str.strip()

    # numeric bits to Unix seconds to UTC
    as_num = pd.to_numeric(raw, errors='coerce')
    ts_num = pd.to_datetime(as_num, unit='s', utc=True)

    # string bits to normal parse to UTC
    ts_str = pd.to_datetime(raw, utc=True, errors='coerce')

    # stitch to pure datetime64[ns, UTC]
    ts_final = ts_num.combine_first(ts_str)

    # convert to Central and drop tz tag
    df['time'] = (
        ts_final
        .dt.tz_convert('America/Chicago')
        .dt.tz_localize(None)
    )

    return df.sort_values('time')


def calculate_rsi(data, periods):
    close_delta = data['close'].diff()
    gain = (close_delta.where(close_delta > 0, 0)
            ).rolling(window=periods).mean()
    loss = (-close_delta.where(close_delta < 0, 0)
            ).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_indicators(data):
    for period in range(5, 22):  # RSI from 5 to 21
        data[f'RSI_{period}'] = calculate_rsi(data, period)

    for period in range(50, 251, 10):  # EMA from 50 to 250 in steps of 10
        data[f'EMA_{period}'] = data['close'].ewm(
            span=period, adjust=False).mean()

    for period in range(50, 251, 10):  # SMA from 50 to 250 in steps of 10
        data[f'SMA_{period}'] = data['close'].rolling(window=period).mean()

    return data


def combine_data(data_folder='import'):
    # Loop through all files in the import folder
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            # Load the data from the file
            import_data = load_csv(os.path.join(data_folder, file))

            # Loop through all files in the data folder
            for data_file in os.listdir('data'):
                if file[:12] == data_file[:12]:
                    # Load the data from the file
                    data = load_csv(os.path.join('data', data_file))
                    data = pd.concat([data, import_data])  # Combine the data
                    data = data.drop_duplicates()  # Drop duplicates
                    # Calculate the indicators
                    data = calculate_indicators(data)
                    data.to_csv(os.path.join('data', data_file),
                                index=False)  # Save the combined data
                    print(f"Combined data for {file} with {data_file}")
                    break
            else:
                import_data = calculate_indicators(
                    import_data)  # Calculate the indicators
                import_data.to_csv(os.path.join('data', file),
                                   index=False)  # Save the data
                print(f"Saved data for {file} as a new file")
    return
