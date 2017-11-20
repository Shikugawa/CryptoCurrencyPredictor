import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def main():
  raw_data = pd.read_csv("data.csv").dropna()
  
  # datetimeを切ってhoutのみ抽出
  time_array = []
  for value in np.reshape(raw_data[["datetime"]].values, (-1, )):
    time_array.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(value)).group(1))
  raw_data["hour"] = pd.Series(time_array)

  raw_data.drop([np.size(raw_data["openprice"].values)-1])

  # openprice...一定期間の最初の価格, closeprice..一定期間の最後の価格
  options = ["lowprice","closeprice","volume","hour","highprice","openprice", "averageprice"]
  df_train = raw_data[options]
  df_label = raw_data[["updown"]]
  
  x_train, x_test, y_train, y_test = train_test_split(df_train, df_label, train_size=0.9, random_state=1)

  # 最新の価格データを取得し、updownカラムに上がり下がりデータを追記させればよい
  model = GradientBoostingClassifier(n_estimators=100, verbose=1)
  output = model.fit(x_train[options].values, np.reshape(y_train.values, (-1, ))).predict(x_test[options].values)

  print(output)
  print(accuracy_score(np.reshape(y_test.values, (-1, )), output))
  
  joblib.dump(output, 'model.pkl')

if __name__ == '__main__':
  main()
  