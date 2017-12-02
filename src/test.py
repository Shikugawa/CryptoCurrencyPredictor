import pandas as pd
import numpy as np

raw_data = pd.read_csv("coincheck.csv").dropna()

high = raw_data[["High"]].rolling(window=9 * 60 * 24, center=False).max().dropna()
low = raw_data[["Low"]].rolling(window=9 * 60 * 24, center=False).min().dropna()
high = high.values
low = low.values
data = np.reshape(high + low, (-1, ))
print(data)