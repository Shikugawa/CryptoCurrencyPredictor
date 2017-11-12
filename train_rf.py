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

  df_train = raw_data[["openprice","lowprice","closeprice","volume","datetime"]]
  df_label = raw_data[["highprice"]]
  
  x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.8, random_state=1)

  # print(y_train.values)
  model = RandomForestClassifier(n_estimators=10, verbose=1)
  output = model.fit(x_train[["openprice","lowprice","closeprice","volume"]].values, np.reshape(y_train.values, (-1, ))).predict(x_test[["openprice","lowprice","closeprice","volume"]].values)
  # print(output)
  # fig = plt.figure()
  # ax = fig.add_subplot(111)

  # x = range(0, len(np.reshape(x_test["datetime"].astype(datetime).values, (-1, ))))
  # plt.xticks(x, np.reshape(x_test["datetime"].astype(datetime).values, (-1, )))
  # ax.plot(x, np.reshape(y_test.values, (-1, )), 'o')
  # ax.plot(x, output, color="r")
  # plt.show()

if __name__ == '__main__':
  main()
  