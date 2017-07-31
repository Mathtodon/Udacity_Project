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

df.drop('Date',inplace=True,axis=1)

# Variables
ticker = 'MMM'
start_date = '2000-01-01'
end_date = '2016-12-30'
days = 28

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

X_train = X[:-test_size]
X_test = X[-test_size:]
y_train = y_shifted[:-test_size]
y_test = y_shifted[-test_size:]

train_dates = X_dates[:-test_size]
test_dates = X_dates[-test_size:]

X_train_benchmark = X_train[ticker].values.reshape(-1,1)
X_test_benchmark = X_test[ticker].values.reshape(-1,1)

scaler = StandardScaler()
X_train =  pd.DataFrame(scaler.fit_transform(X_train.copy()), columns = X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test.copy()), columns = X_test.columns)
X_for_future = pd.DataFrame(scaler.transform(X_for_future.copy()), columns = X_for_future.columns) 
##############################################################################
# 								     Feature Selection
##############################################################################
print("Performing Feature Selection")
n_components = models.choose_components(X_train)

pca = PCA(n_components=n_components)
fit = pca.fit(X_train)
X_train_pca = pd.DataFrame(fit.transform(X_train))
X_test_pca = pd.DataFrame(fit.transform(X_test))
X_for_future_pca = pd.DataFrame(fit.transform(X_for_future))

top_features = models.feature_ranking(X_train,y_train)

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
X_for_future_top = X_for_future[top_features]
X_for_future_benchmark = X_for_future[ticker].values.reshape(-1,1)

print("Top Features Selected are", top_features)
print("The number of Principle Componenets selected = ", n_components)
##############################################################################
# 								     LEARN
##############################################################################
print("Training the Models")

# Create Benchmark Model as Linear Regression
benchmark_model = models.benchmark_model(X_train_benchmark,y_train)
benchmark_score = np.round(benchmark_model.score(X_test_benchmark,y_test),5)
benchmark_preds = benchmark_model.predict(X_test_benchmark.reshape(-1,1))
benchmark_future_preds = benchmark_model.predict(X_for_future_benchmark)

# Create Lasso Model that will select the best number of variable within itself
lasso , names = models.LASSO_model(X_train,y_train)

#txt = "Lasso model: " + models.pretty_print_linear(lasso.coef_, names, sort = True)

X_train_few = X_train[names]
X_test_few = X_test[names]
X_for_future_few = X_for_future[names]

lasso_train_preds = lasso.predict(X_train_few)
lasso_preds = lasso.predict(X_test_few)

# Create Models that uses the top variables  
lasso2 = models.LASSO_model2(X_train_top,y_train)
sgd = models.SGD_model(X_train_top,y_train)

lasso2_train_preds = lasso2.predict(X_train_top)
lasso2_preds = lasso2.predict(X_test_top)

# Crate Models that uses pca data
lasso3 = models.LASSO_model2(X_train_pca,y_train)
sgd2 = models.SGD_model(X_train_pca,y_train)

# Predict over the Testing data
sgd_train_preds = sgd.predict(X_train_top)
sgd_preds = sgd.predict(X_test_top)

sgd2_preds = sgd2.predict(X_test_pca)
sgd2_train_preds = sgd2.predict(X_train_pca)

lasso3_preds = lasso3.predict(X_test_pca)
lasso3_train_preds = lasso3.predict(X_train_pca)

# Score the models

lasso_score = np.round(lasso.score(X_test_few,y_test),5)
lasso2_score = np.round(lasso2.score(X_test_top,y_test),5)
lasso3_score = np.round(lasso3.score(X_test_pca,y_test),5)
sgd_score = np.round(sgd.score(X_test_top,y_test),5)
sgd2_score = np.round(sgd2.score(X_test_pca,y_test),5)

print("benchmark score = ", benchmark_score)
print("lasso score =" , lasso_score)
print("lasso2 score =" , lasso2_score)
print("lasso3 score =" , lasso3_score)
print("sgd score =" , sgd_score)
print("sgd2 score =" , sgd2_score)


# Create Future Predictions
future_lasso_preds = lasso.predict(X_for_future_few)
future_lasso2_preds = lasso2.predict(X_for_future_top)
future_lasso3_preds = lasso3.predict(X_for_future_pca)
future_sgd_preds = sgd.predict(X_for_future_top)
future_sgd2_preds = sgd2.predict(X_for_future_pca)


##############################################################################
#									ENSEMBLE
##############################################################################
train_compare = pd.DataFrame(index=X_train.index)
train_compare['Lasso_1'] = lasso_train_preds
train_compare['Lasso_2'] = lasso2_train_preds
train_compare['Lasso_3'] = lasso3_train_preds
train_compare['SGD_1'] = sgd_train_preds
train_compare['SGD_2'] = sgd2_train_preds
#test_compare['Average'] = train_compare.mean(axis=1)
Average_train = pd.DataFrame(train_compare.mean(axis=1), columns = ['Average'])

ensemble_model = models.ENSEMBLE_model(train_compare,y_train)
average_model = models.ENSEMBLE_model(Average_train,y_train)

test_compare = pd.DataFrame(index=X_test.index)
test_compare['Lasso_1'] = lasso_preds
test_compare['Lasso_2'] = lasso2_preds
test_compare['Lasso_3'] = lasso3_preds
test_compare['SGD_1'] = sgd_preds
test_compare['SGD_2'] = sgd2_preds
#test_compare['Average'] = test_compare.mean(axis=1)
Average_test = pd.DataFrame(test_compare.mean(axis=1), columns = ['Average'])

ensemble_preds = ensemble_model.predict(test_compare)
ensemble_score = np.round(ensemble_model.score(test_compare,y_test),5)

average_preds = average_model.predict(Average_test)
average_score = np.round(average_model.score(Average_test,y_test),5)

print("ensemble score =" , ensemble_score)
print("average score =" , average_score)


future_compare = pd.DataFrame(index=X_for_future.index)
future_compare['Lasso_1'] = future_lasso_preds
future_compare['Lasso_2'] = future_lasso2_preds
future_compare['Lasso_3'] = future_lasso3_preds
future_compare['SGD_1'] = future_sgd_preds
future_compare['SGD_2'] = future_sgd2_preds
Average_future = pd.DataFrame(future_compare.mean(axis=1), columns=['Average'])

future_ensemble_preds = ensemble_model.predict(future_compare)
future_average_preds = average_model.predict(Average_future)

bottom = min(future_compare.min())
top = min(future_compare.max())

preds = [future_lasso_preds,future_lasso2_preds,future_lasso3_preds,future_sgd_preds,future_sgd2_preds,future_ensemble_preds]
up_down = 0
for i in preds:
	if np.max(i) > X_test[ticker].values[-1]:
		up_down += 1
	else: 
		up_down += -1

if up_down < 0:
	in_dec = "decrease"
elif up_down > 0:
	in_dec = "increase"
else:
	in_dec = "stay the same"

##############################################################################
#									PLOT
##############################################################################
print("")
print("Predicting the Future")

#txt = txt + "\n"
txt = ''
txt = txt + "\n Benchmark score = " + str(benchmark_score)
txt = txt + "\n Lasso model 1 score = " + str(lasso_score)
txt = txt + "\n Lasso model 2 score = " + str(lasso2_score)
txt = txt + "\n Lasso model 3 score = " + str(lasso3_score)
txt = txt + "\n SGD model 1 score = " + str(sgd_score)
txt = txt + "\n SGD model 2 score = " + str(sgd2_score)
txt = txt + "\n Ensemble model score = " + str(ensemble_score)
txt = txt + "\n Average model score = " + str(average_score)
txt = txt + "\n"
txt = txt + "\n The model predicts that the stock price will {} in the next {} days".format(in_dec, days)

train_plt = plt.subplot2grid((3,6), (0,0), rowspan =2 , colspan=4)
test_plt = plt.subplot2grid((3,6), (0,4), rowspan =2, colspan=2, sharey = train_plt)
future_plt = plt.subplot2grid((3,6), (2,0), rowspan =1, colspan=2)
future_plt.text(days+13,(top-bottom)/2 + bottom,txt)

train_plt.plot(train_dates,y_train)
train_plt.set_title('{} Stock Price Over Traing Dates'.format(ticker))
train_plt.set_ylabel('Price')

test_plt.plot(test_dates,lasso_preds)
test_plt.plot(test_dates,lasso2_preds)
test_plt.plot(test_dates,lasso3_preds)
test_plt.plot(test_dates,sgd_preds)
test_plt.plot(test_dates,sgd2_preds)
test_plt.plot(test_dates,ensemble_preds)
test_plt.plot(test_dates,y_test.values)
test_plt.set_title('{} Stock Price and Predictions \n Over Testing Dates'.format(ticker))
test_plt.set_ylabel('Price')
for tick in test_plt.get_xticklabels():
	tick.set_rotation(45)

future_plt.plot(range(1,days +1),future_lasso_preds, label = "lasso 1")
future_plt.plot(range(1,days +1),future_lasso2_preds, label = "lasso 2")
future_plt.plot(range(1,days +1),future_lasso3_preds, label = "lasso 3")
future_plt.plot(range(1,days +1),future_sgd_preds, label = 'sgd 1')
future_plt.plot(range(1,days +1),future_sgd2_preds, label = 'sgd 2')
future_plt.plot(range(1,days +1),future_ensemble_preds, label = 'ensemble')
future_plt.plot(range(1,days +1),future_average_preds, label = 'average')
future_plt.legend(bbox_to_anchor =(1.05,1.05), loc = 2)
future_plt.set_title('Price of {} Stock in the future'.format(ticker))
future_plt.set_xlabel('Number of Days Into The Future')
future_plt.set_ylabel('Price')

plt.show()