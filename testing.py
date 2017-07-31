##############################################################################
# 									USERS Guide
##############################################################################
'''
This script will take in the {ticker} and {days} variables set by the user and 
will create 5 different models using Lasso Regression and Stochastic Gradient 
Decent Regression to predict the stock price.

The data will be split into parts for training, testing, and future predictions.

The ouput will be a 3 plots showing these different sections of the data with 
the models' predictions. 

Historically the script takes around 560 seconds to finish running which is 
equivalent to 9.5 minutes.
'''
##############################################################################
# 									Load Modules
##############################################################################

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import	r2_score
from sklearn.metrics import mean_absolute_error


##############################################################################
# 									DATA PREP
##############################################################################
print("Loading Data")
df = pd.read_csv('sp500_joined_closes.csv')
tickers = df.columns.values.tolist()
df.dropna(1, inplace = True)
df['Day_N'] = df['Date'].index
Dates = pd.to_datetime(df['Date'].values)
dates = Dates.date

# Variables
ticker = 'MMM'
start_date = '2000-01-01'
end_date = '2015-12-30'
days = 14

training_end = df.loc[df['Date']==end_date].index.values[0]
print(training_end)

df.drop('Date',inplace=True,axis=1)

print("Model will predict stock {} {} days into the future.".format(ticker,days))
ticker_price = df[ticker].copy()

df['30ma'] = ticker_price.rolling(window=30).mean()
df['20ma'] = ticker_price.rolling(window=20).mean()
df['10ma'] = ticker_price.rolling(window=10).mean()
df = df[30:]
dates = dates[30:]

y = df[ticker]
X = df

y_shifted = df[ticker].shift(-days)[:-days]
X = df[:-days]

X_dates = dates[:-days]
X_for_future = df[-days:]
future_dates = dates[-days:]

test_size = int(len(X)*.15)

X_train = X[:-training_end]
X_test = X[-training_end:]
y_train = y_shifted[:-training_end]
y_test = y_shifted[-training_end:]

train_dates = X_dates[:-training_end]
test_dates = X_dates[-training_end:]