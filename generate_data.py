import urllib.request
import json
import csv
import datetime
import time 
import re

def generate_csv(dict_data):
  for date in dict_data:
    date.append(datetime.datetime.fromtimestamp(int(date[0])))

  for middle_prices in dict_data:
    middle = (middle_prices[2] + middle_prices[3]) / 2
    middle_prices.append(middle)

  updown, raw_data_val = [], []
  updown_elem = {"up": 1, "down": 2, "flat": 3}
  
  for price_data in dict_data:
    raw_data_val.append(price_data[-1])

  for value in dict_data:
    value.append(re.match(r"\d{4}-\d{1,2}-\d{1,2} (\d\d):\d\d:\d\d", str(value[6])).group(1))

  for index, data in enumerate(dict_data):
    if(index == len(dict_data)-1):
      break
    else:
      if(data[-1] < dict_data[index+1][-1]):
        data.append(updown_elem["up"])
      elif(data[-1] == dict_data[index+1][-1]):
        data.append(updown_elem["flat"])
      else:
        data.append(updown_elem["down"])

  print(dict_data)

  with open("data.csv", "a") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(dict_data)

  print("succeeded!")


def main():
  now = datetime.datetime.now()

  after = datetime.datetime(2017, 6, 1, 0, 0, 0)
  periods = "3600"

  with open("data.csv", "a") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["closetime", "openprice", "highprice", "lowprice", "closeprice", "volume", "datetime", "averageprice", "hour", "updown"])

  print(after)
  print(int(time.mktime(after.timetuple())))

  url = "https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc?periods=" + periods + "&after=" + str(int(time.mktime(after.timetuple())))

  print(url)
  with urllib.request.urlopen(url) as response:
    result = json.loads(response.read().decode('UTF-8'))
    after = generate_csv(result['result'][periods])

if __name__ == '__main__':
  main()