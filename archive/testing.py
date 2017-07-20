import numpy as np 
import scipy as sp 
import pandas as pd
from pandas import Series
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from datetime import datetime
import models
from sklearn.decomposition import PCA


##############################################################################
# 									DATA PREP
##############################################################################
print("Loading Data")
df = pd.read_csv('sp500_joined_closes.csv')
tickers = df.columns.values.tolist()
df.dropna(1, inplace = True)
df['Day_N'] = df['Date'].index

df.drop('Date',inplace=True,axis=1)

# Variables
ticker = 'MMM'
start_date = '2000-01-01'
end_date = '2016-12-30'
days = 7

ticker_price = df[ticker].copy()

df['30ma'] = ticker_price.rolling(window=30).mean()
df['20ma'] = ticker_price.rolling(window=20).mean()
df['10ma'] = ticker_price.rolling(window=10).mean()
df = df[30:]


y = df[ticker]
X = df

y_shifted = df[ticker].shift(-days)[:-days]
X = df[:-days]
X_for_future = df[-days:]

test_size = int(len(X)*.15)

#X.drop('Date',inplace=True,axis=1)
#X_for_future.drop('Date',inplace=True,axis=1)

X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y_shifted[:-test_size]
y_test = y_shifted[-test_size:]

X_train_benchmark = X_train[ticker]
X_test_benchmark = X_test[ticker].values.reshape(-1,1)

#print(X_train.shape)
#print(X_train_benchmark.shape)
#print(X_test.shape)
#print(X_test_benchmark.shape)
#collin = pd.DataFrame()
collin = pd.DataFrame()
collin['x']= pd.Series(X_train_benchmark, index=collin.index)
collin['y']=y_train

print(collin)
collin.corr()