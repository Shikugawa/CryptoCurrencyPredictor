import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
  raw_data = pd.read_csv("data.csv")
  data = raw_data[["openprice", "datetime"]].dropna()

  df = pd.DataFrame({
    'time': pd.to_datetime(data["datetime"].astype(datetime)),
    'price': [ int(d) for d in data["openprice"] ] 
  })

  # 価格の上がり下がりなんかを記録するカラムを作成する
  updown = []
  updown_elem = {"up": 1, "down": 2, "flat": 3}
  raw_data_val = raw_data["openprice"].values

  for index, current_data in enumerate(raw_data_val):
    if(index == np.size(raw_data_val)-1):
      break
    else:
      if(current_data < raw_data_val[index+1]):
        updown.append(updown_elem["up"])
      elif(current_data is raw_data_val[index+1]):
        updown.append(updown_elem["flat"])
      else:
        updown.append(updown_elem["down"])
  
  raw_data.drop([np.size(raw_data["openprice"].values)-1])
  print(np.size(raw_data["openprice"].values)-1)
  print(np.size(updown))
  raw_data["updown"] = pd.Series(updown)
  raw_data = raw_data.dropna()
  print(raw_data)

  df_train = raw_data[["lowprice","closeprice","volume","datetime", "highprice", "updown"]]
  df_label = raw_data[["openprice"]]
  
  x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.8, random_state=1)

  model = RandomForestClassifier(n_estimators=10, verbose=1)
  predict_options = ["highprice","lowprice","closeprice","volume", "updown"]
  output = model.fit(x_train[predict_options].values, np.reshape(y_train.values, (-1, ))).predict(df_train[predict_options].values)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  # x = range(0, len(np.reshape(x_test["datetime"].astype(datetime).values, (-1, ))))
  # plt.xticks(x, np.reshape(x_test["datetime"].astype(datetime).values, (-1, )))
  # ax.plot(np.reshape(df['price'].values, (-1, )))
  # ax.plot(x, output, color="r")
  plt.show()

if __name__ == '__main__':
  main()
  