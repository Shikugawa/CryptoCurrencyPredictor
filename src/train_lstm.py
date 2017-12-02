from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import datetime
import re

class Model():
    def __init__(self, term):
        self.hidden_neurons = 256
        self.in_out_newrons = 3
        self.term = term

    def create_data(self, data, label_data):
        created_data = []
        nested_data = []
        label = []

        for index, dt in enumerate(data):
            nested_data.append(dt)

            if len(nested_data) == self.term:
                created_data.append(nested_data)
                label.append(label_data[index])
                nested_data = []

        # convert to one hot encoding
        label = np.reshape(np.array(label), (-1, 1))
        label = np_utils.to_categorical(label)
        return np.array(created_data), label

    def create_model(self, y):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.term, y)))
        model.add(Dropout(0.5))
        model.add(Dense(self.in_out_newrons))
        model.add(Activation("softmax"))
        return model

class TechnicalTerm():
    @classmethod
    def bolinger_band(self, raw_data):
        bolinger_option = ["bolinger_upper1", "bolinger_lower1", "bolinger_upper2", "bolinger_lower2"]
        base = raw_data[["Close"]].rolling(window=25).mean()
        std = raw_data[["Close"]].rolling(window=25).std()

        for opt in bolinger_option:
            if opt == "bolinger_upper1":
                raw_data[opt] = base + std
            elif opt == "bolinger_lower1":
                raw_data[opt] = base - std
            elif opt == "bolinger_upper2":
                raw_data[opt] = base + 2 * std
            elif opt == "bolinger_lower2":
                raw_data[opt] = base - 2 * std

        data = raw_data.dropna()
        return data

    @classmethod
    def conversion(self, raw_data):
        raw_data["rol_high"] = raw_data[["High"]].rolling(window=9*60*24).max()
        raw_data["rol_low"] = raw_data[["Low"]].rolling(window=9*60*24).min()
        raw_data = raw_data.dropna()

        high = raw_data[["rol_high"]].values
        low = raw_data[["rol_low"]].values
        raw_data["conversion"] = np.reshape((high + low) / 2, (-1, ))
        data = raw_data

        return data

raw_data = pd.read_csv("coincheck.csv").dropna()

# append hour
hour = []
hour_data = np.reshape(raw_data[["Timestamp"]].values, (-1, ))
for timestamp in hour_data:
    time = datetime.datetime.fromtimestamp(int(timestamp))
    hour.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(time)).group(1))
raw_data["Hour"] = hour

# append updown
updown_elem = {"up": 0, "down": 1, "flat": 2, "unknown": 9}
weighed_price = np.reshape(raw_data[["Weighted_Price"]].values, (-1, ))
updown_data = []

for index, data in enumerate(weighed_price):
    if (index == len(weighed_price) - 1):
        break
    else:
        if (data < weighed_price[index + 1]):
            updown_data.append(updown_elem["up"])
        elif (data == weighed_price[index+1]):
            updown_data.append(updown_elem["flat"])
        else:
            updown_data.append(updown_elem["down"])

raw_data = raw_data.drop(len(weighed_price)-1)
raw_data["updown"] = updown_data

raw_data = raw_data.drop(["Timestamp"], axis=1)

# append bolinger band
raw_data = TechnicalTerm.bolinger_band(raw_data)

# append conversion line
raw_data = TechnicalTerm.conversion(raw_data)

options = ["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price", "Hour",
           "bolinger_upper1", "bolinger_lower1", "bolinger_upper2", "bolinger_lower2", "conversion"]
df_train = raw_data[options]
df_label = raw_data[["updown"]]

lstm_model = Model(10)

x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=0)
x_train = x_train[options].values
y_train = y_train.values

# normalize
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_train, y_train = lstm_model.create_data(x_train, y_train)
model = lstm_model.create_model(len(options))
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

# -------------training-------------
check = ModelCheckpoint("model.hdf5")
output = model.fit(x_train, y_train, epochs=30, callbacks=[check], verbose=1)

# -------------evaluation-------------
x_test = x_test[options].values
y_test = y_test.values

# test data normalize
scaler = preprocessing.MinMaxScaler()
x_test = scaler.fit_transform(x_test)

x_test, y_test = lstm_model.create_data(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test)
print("\nloss:{} accuracy:{}".format(loss, accuracy))

