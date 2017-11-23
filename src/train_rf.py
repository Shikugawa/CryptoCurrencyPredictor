import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
  raw_data = pd.read_csv("data.csv").dropna()
  data = raw_data[["openprice", "datetime"]].dropna()

  df = pd.DataFrame({
    'time': pd.to_datetime(data["datetime"].astype(datetime)),
    'price': [ int(d) for d in data["openprice"] ] 
  })

  # datetimeを切ってhoutのみ抽出
  time_array = []
  for value in np.reshape(raw_data[["datetime"]].values, (-1, )):
    print(value)
    time_array.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(value)).group(1))
  raw_data["hour"] = pd.Series(time_array)

  raw_data.drop([np.size(raw_data["openprice"].values)-1])
  raw_data = raw_data.dropna()

  options = ["lowprice","closeprice","volume","hour","highprice","openprice", "averageprice"]
  df_train = raw_data[options]
  df_label = raw_data[["updown"]]
  
  x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=1)

  model = RandomForestClassifier(n_estimators=10, verbose=1)
  output = model.fit(x_train[options].values, np.reshape(y_train.values, (-1, ))).predict(x_test[options].values)

  print(accuracy_score(np.reshape(y_test.values, (-1, )), output))
  fig = plt.figure()
  ax = fig.add_subplot(111)

  # x = range(0, len(np.reshape(x_test["datetime"].astype(datetime).values, (-1, ))))
  # plt.xticks(x, np.reshape(x_test["datetime"].astype(datetime).values, (-1, )))
  # ax.plot(np.reshape(df['price'].values, (-1, )))
  # ax.plot(x, output, color="r")
  # plt.show()

if __name__ == '__main__':
  main()
  