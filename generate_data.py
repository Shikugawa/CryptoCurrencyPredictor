import urllib.request
import json
import csv
import datetime
import time 

def generate_csv(dict_data, periods, after):
  for date in dict_data:
    date.append(datetime.datetime.fromtimestamp(int(date[0])))
  # for index, data in enumerate(dict_data):
  #   data.append(after)

  #   if index == len(dict_data)-1:
  #     before_time = after
  #   else:
  #     after += datetime.timedelta(hours=int(periods)/3600)

  with open("data.csv", "a") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(dict_data)

  print("succeeded!")


def main():
  now = datetime.datetime.now()

  after = datetime.datetime(2017, 6, 1, 0, 0, 0)
  periods = "3600"

  before = str(time.mktime(now.timetuple()))

  with open("data.csv", "a") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["closetime", "openprice", "highprice", "lowprice", "closeprice", "volume", "datetime"])
    
  # while now > after:
  print(after)
  print(int(time.mktime(after.timetuple())))

  url = "https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=" + periods + "&after=" + str(int(time.mktime(after.timetuple())))

  print(url)
  with urllib.request.urlopen(url) as response:
    result = json.loads(response.read().decode('UTF-8'))
    print(result)
    after = generate_csv(result['result'][periods], periods, after)
    print(after)

if __name__ == '__main__':
  main()