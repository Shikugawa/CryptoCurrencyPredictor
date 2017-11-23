from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

raw_data = pd.read_csv("data.csv").dropna()
raw_data = raw_data.drop(["closetime", "datetime"], axis=1)

options = ["lowprice", "closeprice", "volume", "hour", "highprice", "openprice", "averageprice"]
df_train = raw_data[options]
df_label = raw_data[["updown"]]

lstm_model = Model(5)

x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=0)
x_train = x_train[options].values
y_train = y_train.values

x_train, y_train = lstm_model.create_data(x_train, y_train)

