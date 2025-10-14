import pandas as pd
df = pd.read_csv('data/raw/ohlc_data.csv')
print('Data loaded successfully')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('First 3 rows:')
print(df.head(3))