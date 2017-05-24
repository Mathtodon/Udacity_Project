'''
INVESTMENT AND TRADING CAPSTONE PROJECT
This is Capstone Project for Udacity's Machine Learning Engineer Nanodegree 
The purpose of this project is to Build a Stock Price Indicator

Stock data will be pulled from Quandl's API

We will be trying to predict the Adjusted Close price of the designated stock
'''


#Import

import numpy as np 
import scipy as sp 
import pandas as pd 
import quandl
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#Querry data from Quandl
quandl.ApiConfig.api_key = 'Ri7BNHzdujUt3zYB7NQr'

#Data consists of 	Open
#					High
#					Low
#					Close
#					Volume
#					Ex-Dividend
#					Split Ratio
#					Adj. Open
#					Adj. High
#					Adj. Low
#					Adj Close
#					Adj Volume


start = "2014-01-01"
end = "2016-12-31"
df = quandl.get("WIKI/FB", start_date= start, end_date= end)


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput





#graph_data = data['Adj. High']
import matplotlib
matplotlib.style.use('ggplot')

#test_stationarity(graph_data)

#plt.plot(graph_data, color = 'blue')
#
#
#moving_avg_20 = graph_data.rolling(window = 20, center =False).mean()
#plt.plot(moving_avg_20, color='red')
#
#moving_avg_10 = graph_data.rolling(window = 10, center =False).mean()
#plt.plot(moving_avg_10, color='pink')
#
#plt.show()
#
#df_ohlc = df['Adj. Close'].resample('10D').ohlc()
#df_volume = df['Volume'].resample('10D').sum()
#
#print(df_ohlc.head())
#
#ax1 = plt.subplot2grid((6,1), (0,0), rowspan =5, colspan=1)
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan =1, colspan=1, sharex = ax1)

# plot Adj Close with 100 Moving Average plus Volume underneath

#df['100ma'] = df['Adj. Close'].rolling(window=100).mean()
#ax1.plot(df.index, df['Adj. Close'])
#ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index, df['Volume'])

#lt.show()


ax1 = plt.subplot2grid((8,1), (0,0), rowspan =5, colspan=1)
ax2 = plt.subplot2grid((8,1), (5,0), rowspan =2, colspan=1, sharex = ax1)
ax3 = plt.subplot2grid((8,1), (7,0), rowspan =1, colspan=1, sharex = ax1)

# plot Adj Close with 100 Moving Average plus Volume underneath
df['percent_change'] = (df['Adj. Close'].shift(1)-df['Adj. Close'])/df['Adj. Close']
df['100ma'] = df['Adj. Close'].rolling(window=100).mean()

ax1.plot(df.index, df['Adj. Close'])
ax1.plot(df.index, df['100ma'])
ax2.plot(df.index, df['percent_change'])
ax3.bar(df.index, df['Volume'])

plt.show()