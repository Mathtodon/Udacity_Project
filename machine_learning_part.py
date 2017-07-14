'''
This script classifies a day when in 7 days into the future the price has risen or follen by x%
then it trains on this set and tries to learn what characteristics will predict the rise or fall
'''



from collections import Counter
import numpy as np 
import pandas as pd 
import pickle
from sklearn import	svm, neighbors
from sklearn.model_selection import	train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
	hm_days = 7
	df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
	tickers = df.columns.values.tolist()
	df.fillna(0, inplace = True)

	for i in range(1, hm_days+1):
		df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

	df.fillna(0, inplace =True)
	return tickers, df


def buy_sell_hold(*args):
	cols = [c for c in args]
	requirement = 0.05
	for col in cols:
		if col > requirement:
			return 1
		if col < -requirement:
			return -1
	return 0

def extract_featuresets_for_clf(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = vals
    
    return X, y, df

def extract_featuresets_for_reg(ticker):
	hm_days = 7
	df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
	tickers = df.columns.values.tolist()
	df.fillna(0, inplace = True)

	for i in range(1, hm_days+1):
		df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

	df.fillna(0, inplace =True)
	
    vals = df[ticker].values.tolist()
    str_vals = [str(i) for i in vals]
    
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = vals
    
    return X, y

def do_clf(ticker):
	X, y, df = extract_featuresets(ticker)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
	
	clf = VotingClassifier([('lsvc', svm.LinearSVC()),
							('knn', neighbors.KNeighborsClassifier()),
							('rfor', RandomForestClassifier())])


	clf.fit(X_train, y_train)

	confidence = clf.score(X_test, y_test)
	print('Accuracy: ',confidence)
	predictions =  clf.predict(X_test)
	print('Predicted spread:', Counter(predictions))

	return confidence

print(extract_featuresets_for_reg('AAPL'))
#do_ml('BAC')

#from sklearn import svm
#X = [[0, 0], [2, 2]]
#y = [0.5, 2.5]
#clf = svm.SVR()
#clf.fit(X, y) 