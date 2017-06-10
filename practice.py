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
print(tickers)
	
	
