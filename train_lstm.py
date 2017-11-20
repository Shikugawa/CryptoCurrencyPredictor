from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re

class Model():
    def __init__(self):
        self.hidden_neurons = 128
        self.in_out_newrons = 1
        self.epochs = 25

    def create_model(self):
        model = Sequential()
        model.add(LSTM(self.hidden_neurons, batch_input_shape=(None, self.in_out_newrons, self.epochs)))
        model.add(Dropout(0.5))
        model.add(Dense(self.in_out_newrons))
        model.add(Activation("sigmoid"))
        return model

    @property
    def get_epoch(self):
        return self.epochs

raw_data = pd.read_csv("data.csv").dropna()

options = ["lowprice", "closeprice", "volume", "hour", "highprice", "openprice", "averageprice"]
df_train = raw_data[options]
df_label = raw_data[["updown"]]

# datetimeを切ってhoutのみ抽出
time_array = []
for value in np.reshape(raw_data[["datetime"]].values, (-1, )):
    time_array.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(value)).group(1))

raw_data["hour"] = pd.Series(time_array)
raw_data.drop([np.size(raw_data["openprice"].values)-1])

x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=1)

model_obj = Model()
model = model_obj.create_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

output = model.fit(x_train, y_train, epochs=model_obj.get_epoch())