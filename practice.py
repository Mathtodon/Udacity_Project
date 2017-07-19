import csv


with open("sp500tickers.csv", 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	tickers = list(reader)

import pandas as pd

# Read the CSV into a pandas data frame (df)
#   With a df you can do many things
#   most important: visualize data with Seaborn
#df = pd.read_csv('sp500tickers.csv', delimiter=',')	
#tickers = list(df.values):
#	tickers.append(ticker)
#	#print(ticker)
#print(tickers)


x = [[0.84  ,  0.47 ,   0.55  ,  0.46  ,  0.76  ,  0.42  ,  0.24  ,  0.75],
[0.43  ,  0.47  ,  0.93  ,  0.39   , 0.58 ,   0.83  ,  0.35  ,  0.39],
[0.12  ,  0.17  ,  0.35  ,  0.00   , 0.19  ,  0.22  ,  0.93  ,  0.73],
[0.95  ,  0.56  ,  0.84  ,  0.74   , 0.52   , 0.51  ,  0.28  ,  0.03],
[0.73  ,  0.19  ,  0.88  ,  0.51   , 0.73   , 0.69  ,  0.74  ,  0.61],
[0.18  ,  0.46  ,  0.62  ,  0.84  ,  0.68   , 0.17  ,  0.02  ,  0.53],
[0.38  ,  0.55  ,  0.80  ,  0.87 ,   0.01   , 0.88  ,  0.56  ,  0.72]]
X = pd.DataFrame(x)	

start_date = ['2000-01-01','2000-01-02','2000-01-03','2000-01-04','2000-01-05','2000-01-06','2000-01-07','2000-01-08']

Date = pd.to_datetime(start_date)
print(Date)
Date = Date.date

	
y = [0.84  ,  0.47 ,   0.55  ,  0.46  ,  0.76  ,  0.42  ,  0.24  ,  0.75]

import matplotlib.pyplot as plt

train_plt = plt.subplot2grid((3,6), (0,0), rowspan =2 , colspan=4)
test_plt = plt.subplot2grid((3,6), (0,4), rowspan =2, colspan=2, sharey = train_plt)
future_plt = plt.subplot2grid((3,6), (2,0), rowspan =1, colspan=2)

train_plt.plot(y)
test_plt.plot(Date,y)
future_plt.plot(y,label = y)
for tick in test_plt.get_xticklabels():
	tick.set_rotation(90)
future_plt.legend(bbox_to_anchor =(1.05,1.05), loc = 2)
plt.show()