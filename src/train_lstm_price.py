from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers.recurrent import LSTM
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import re

plt.style.use('ggplot')

def create_data(data, label_data, term):
    created_data = []
    nested_data = []
    label = []

    label_data = np.reshape(label_data, (-1,))
    for index, dt in enumerate(data):
        nested_data.append(dt)

        if len(nested_data) == term:
            created_data.append(nested_data)
            label.append(label_data[index])
            nested_data = []

    label = np.reshape(np.array(label), (-1,))
    return np.array(created_data), label


def split_data(train, label, testing_rate=0.9):
    train_x, test_x = train[1:int(len(train) * testing_rate)], train[1 + int(len(train) * testing_rate):len(train)]
    train_y, test_y = label[1:int(len(label) * testing_rate)], label[1 + int(len(label) * testing_rate):len(label)]
    return train_x, train_y, test_x, test_y


def training(x_train, y_train, x_test, y_test, term, option_length, neurons=128, dropout=0.25, epoch=20):
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(None, term, option_length),
                   recurrent_regularizer=regularizers.l2(0.), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mean_squared_error", optimizer="adam")

    # -------------training-------------
    output = model.fit(x_train, y_train, epochs=epoch, verbose=1)

    predicted_price = model.predict(x_test)
    datas = pd.DataFrame(
        data={
            'real_price': np.reshape(y_test, (-1,)),
            'predicted_price': np.reshape(predicted_price, (-1,))
        }
    )

    plt.plot(output.history['loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss'], loc='lower right')

    print(predicted_price)
    datas.plot(figsize=(12, 7))
    plt.show()
    print(mean_absolute_error(np.reshape(y_test, (-1,)),
                              np.reshape(predicted_price, (-1,))))

raw_data = pd.read_csv("C:\\Users\\shikugawa\\CryptoCurrencyPredictor\\src\\coin_refine.csv").dropna()

# append hour
hour = []
hour_data = np.reshape(raw_data[["Timestamp"]].values, (-1,))
for timestamp in hour_data:
    time = datetime.datetime.fromtimestamp(int(timestamp))
    hour.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(time)).group(1))

raw_data["Hour"] = hour
raw_data = raw_data.drop(["Timestamp"], axis=1)

options = ["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)", "Hour"]
df_train = raw_data[options]
df_label = raw_data[["Weighted_Price"]]
term = 10

x_train, y_train, x_test, y_test = split_data(df_train, df_label)

# ----train data noramlization---------
x_train = x_train[options].values
y_train = y_train.values
x_train, y_train = create_data(x_train, y_train, term)
# -------------------------------------

# ----train data noramlization---------
x_test = x_test[options].values
y_test = y_test.values
x_test, y_test = create_data(x_test, y_test, term)
# -------------------------------------

training(x_train, y_train, x_test, y_test, term, len(options), neurons=256, dropout=0.25, epoch=35)