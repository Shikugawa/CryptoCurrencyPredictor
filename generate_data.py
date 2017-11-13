import urllib.request
import json
import csv
import datetime

def generate_csv(dict_data, periods, after):
  datetime_ary = []
  origin_date_str = datetime.datetime.fromtimestamp(int(after))
  print(origin_date_str)
  origin = datetime.datetime.strptime(str(origin_date_str), '%Y-%m-%d %H:%M:%S')
  
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
  after = "1501513200"
  periods = "300"
  url = "https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=" + periods + "&after=" + after

  with urllib.request.urlopen(url) as response:
    result = json.loads(response.read().decode('UTF-8'))
    # print(result['result'][periods])
    generate_csv(result['result'][periods], periods, after)

if __name__ == '__main__':
  main()