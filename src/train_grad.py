import pandas as pd
import numpy as np
import re
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn import preprocessing
from keras.utils import np_utils

class Model():
    def __init__(self, term):
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

def main():
  raw_data = pd.read_csv("/Users/kiyohoshi/CryptoCurrencyPredictor/coincheck.csv").dropna()

  # append hour
  hour = []
  hour_data = np.reshape(raw_data[["Timestamp"]].values, (-1,))
  for timestamp in hour_data:
    time = datetime.datetime.fromtimestamp(int(timestamp))
    hour.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(time)).group(1))
  raw_data["Hour"] = hour

  # append updown
  updown_elem = {"up": 0, "down": 1, "flat": 2, "unknown": 9}
  weighed_price = np.reshape(raw_data[["Weighted_Price"]].values, (-1,))
  updown_data = []

  for index, data in enumerate(weighed_price):
    if (index == len(weighed_price) - 1):
      break
    else:
      if (data < weighed_price[index + 1]):
        updown_data.append(updown_elem["up"])
      elif (data == weighed_price[index + 1]):
        updown_data.append(updown_elem["flat"])
      else:
        updown_data.append(updown_elem["down"])

  raw_data = raw_data.drop(len(weighed_price) - 1)
  raw_data["updown"] = updown_data

  raw_data = raw_data.drop(["Timestamp"], axis=1)

  options = ["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price", "Hour"]
  df_train = raw_data[options]
  df_label = raw_data[["updown"]]

  x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=0)
  x_train = x_train[options].values
  y_train = y_train.values

  # normalize
  scaler = preprocessing.MinMaxScaler()
  x_train = scaler.fit_transform(x_train)


  x_test = x_test[options].values
  y_test = y_test.values

  # test data normalize
  x_test = scaler.fit_transform(x_test)

  # 最新の価格データを取得し、updownカラムに上がり下がりデータを追記させればよい
  model = GradientBoostingClassifier(n_estimators=100, verbose=1)
  output = model.fit(x_train, y_train).predict(x_test)

  print(output)
  print(accuracy_score(y_test, output))
  
  joblib.dump(output, 'model.pkl')

if __name__ == '__main__':
  main()
  