from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

SERIES_LENGTH = 1000

# Prepare dataset
dataset = np.arange(1, SERIES_LENGTH+1, 1).reshape(SERIES_LENGTH, 1).astype(np.float)
# print(dataset)
# Transform
scaler = preprocessing.MinMaxScaler()
dataset = scaler.fit_transform(dataset)
# print(dataset)
# Split dataset into train and test subsets
train_dataset = dataset[0:int(len(dataset)*0.8), :]
test_dataset =  dataset[len(train_dataset):len(dataset), :]

look_back = 1
x, y = [], []
for i in range(len(dataset) - look_back):
    a = i + look_back
    x.append(dataset[i:a, 0])
    y.append(dataset[a, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
model = Sequential()
model.add(LSTM(128, input_shape=(1, look_back)))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
print(x)
model.fit(x, y, batch_size=2, epochs=10, verbose=1)