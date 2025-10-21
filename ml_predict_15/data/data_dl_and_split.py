import pandas as pd
import kagglehub
import os

os.makedirs('data', exist_ok=True)

path_to_data = 'BTC-USD_data_1min.csv'
#path_to_data = kagglehub.dataset_download("aklimarimi/bitcoin-historical-data-1min-interval", force_download=True)

df = pd.read_csv(os.path.join('data', path_to_data))
if 'Date' in df.columns:
    df.columns = [x if x != 'Date' else 'Timestamp' for x in df.columns]
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna()
min_year = df.index.min().year
max_year = df.index.max().year
for year in range(min_year, max_year + 1):
    print(f'Processing year {year}')
    df_year = df[df.index.year == year]
    df_year.to_csv(os.path.join('data', f'btc_{year}.csv'))