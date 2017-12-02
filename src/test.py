import pandas as pd

raw_data = pd.read_csv("coincheck.csv").dropna()

# append bolinger band
base = raw_data[["Close"]].rolling(window=25).mean()
std = raw_data[["Close"]].rolling(window=25).std()
band1_upper = base + 2 * std
band1_lower = base - 2 * std

print(band1_lower)