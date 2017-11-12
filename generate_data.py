import urllib.request
import json
import csv
import datetime

def generate_csv(dict_data, periods):
  datetime_ary = []
  origin_date_str = '2017/01/01 00:00:00'
  origin = datetime.datetime.strptime(origin_date_str, '%Y/%m/%d %H:%M:%S')
  
  while origin < datetime.datetime.now():
    datetime_ary.append(origin)
    origin += datetime.timedelta(hours=int(periods)/3600)
  
  for data_ary, date_data in zip(dict_data, datetime_ary):
    data_ary.append(date_data)

  with open("data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["closetime", "openprice", "highprice", "lowprice", "closeprice", "volume", "datetime"])
    writer.writerows(dict_data)

  print("succeeded!")

def main():
  after = "1483196400"
  periods = "21600"
  url = "https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=" + periods + "&after=" + after

  with urllib.request.urlopen(url) as response:
    result = json.loads(response.read().decode('UTF-8'))
    # print(result['result'][periods])
    generate_csv(result['result'][periods], periods)

if __name__ == '__main__':
  main()