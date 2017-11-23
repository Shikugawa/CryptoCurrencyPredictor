from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import datetime
import re

class Model():
    def __init__(self, term):
        self.hidden_neurons = 128
        self.in_out_newrons = 1
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

        return np.array(created_data), np.reshape(np.array(label), (-1, 1))

    def create_model(self, x, y):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, x, y)))
        model.add(Dropout(0.5))
        model.add(Dense(self.in_out_newrons))
        model.add(Activation("relu"))
        return model

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
        updown_data.append(updown_elem["unknown"])
        break
    else:
        if (data < weighed_price[index + 1]):
            updown_data.append(updown_elem["up"])
        elif (data == weighed_price[index+1]):
            updown_data.append(updown_elem["flat"])
        else:
            updown_data.append(updown_elem["down"])

raw_data["updown"] = updown_data

raw_data = raw_data.drop(["Timestamp"], axis=1)

options = ["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price", "Hour"]
df_train = raw_data[options]
df_label = raw_data[["updown"]]

lstm_model = Model(5)

x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=0)
x_train = x_train[options].values
y_train = y_train.values

# normalize
scaler = preprocessing.MinMaxScaler()
x_train = scaler.fit_transform(x_train)

x_train, y_train = lstm_model.create_data(x_train, y_train)

print(x_train)
