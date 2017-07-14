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
print "Loading Data"
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
##############################################################################
# 								     Feature Selection
##############################################################################
print "Performing Feature Selection"
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

print "Top Features Selected are", top_features
print "The number of Principle Componenets selected = ", n_components
##############################################################################
# 								     LEARN
##############################################################################
print "Training the Models"
lasso , names = models.LASSO_model(X_train,y_train)

txt = "Lasso model: " + models.pretty_print_linear(lasso.coef_, names, sort = True)

X_train_few = X_train[names]
X_test_few = X_test[names]
X_for_future_few = X_for_future[names]
lasso_predictions = lasso.predict(X_test_few)

sgd = models.SGD_model(X_train_top,y_train)
#sgd2 = models.SGD_model(X_train_few,y_train)
#svr = models.SVR_model(X_train_top,y_train)

sgd_preds = sgd.predict(X_test_top)
#sgd_predictions_few = sgd2.predict(X_test_few)
#sgd_w_pca_predictions = sgd_w_pca.predict(X_test_pca)
#svr_preds = svr.predict(X_test_top)

print "lasso score =" , lasso.score(X_test_few,y_test)
#print "svr score = ", svr.score(X_test_top,y_test)
print "sgd score =" , sgd.score(X_test_top,y_test)
#print "sgd2 score =" , sgd2.score(X_test_few,y_test)
#print "sgd_wPCA score =" , sgd_w_pca.score(X_test_pca,y_test)

print ""
print "Predicting the Future"

future_lasso_preds = lasso.predict(X_for_future_few)
future_sgd_preds = sgd.predict(X_for_future_top)
#future_sgd_w_pca_preds = sgd_w_pca.predict(X_for_future_pca)
#future_svr_preds = svr.predict(X_for_future_top)

##############################################################################
#									PLOT
##############################################################################

train_plt = plt.subplot2grid((3,6), (0,0), rowspan =2 , colspan=4)
test_plt = plt.subplot2grid((3,6), (0,4), rowspan =2, colspan=2, sharey = train_plt)
future_plt = plt.subplot2grid((3,6), (2,0), rowspan =1, colspan=1)
future_plt.text(days+1,(future_lasso_preds.max()-future_lasso_preds.min())/2 +future_lasso_preds.min(),txt)

train_plt.plot(y_train)

test_plt.plot(lasso_predictions)
test_plt.plot(y_test.values)

future_plt.plot(range(1,days +1),future_lasso_preds)
future_plt.plot(range(1,days +1),future_sgd_preds)
#future_plt.plot(range(1,days +1),future_svr_preds)
#future_plt.plot(range(1,days +1),future_sgd2_predictions)
#future_plt.plot(range(1,days +1),future_sgd_w_pca_predictions)
#future_plt.title('Price of Stock # of Days into the future')
plt.show()