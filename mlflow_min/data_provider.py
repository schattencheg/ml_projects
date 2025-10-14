import pandas as pd
import numpy as np

def get_data(num_rows=100):
    """
    A dummy data provider that returns a pandas DataFrame with OHLC data.
    Replace this with your actual data provider.
    """
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=num_rows))
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 102, num_rows),
        'High': 0.0,
        'Low': 0.0,
        'Close': np.random.uniform(100, 102, num_rows),
        'Volume': np.random.randint(100000, 500000, num_rows)
    }
    df = pd.DataFrame(data)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 2, num_rows)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 2, num_rows)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.set_index('Date', inplace=True)
    return df

if __name__ == '__main__':
    ohlc_data = get_data()
    print("Sample OHLC Data:")
    print(ohlc_data.head())
