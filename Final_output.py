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

X_train_benchmark = X_train[ticker].values.reshape(-1,1)
X_test_benchmark = X_test[ticker].values.reshape(-1,1)

##############################################################################
# 								     Feature Selection
##############################################################################
print("Performing Feature Selection")
#n_components = models.choose_components(X_train)

#pca = PCA(n_components=n_components)
#fit = pca.fit(X_train)
#X_train_pca = pd.DataFrame(fit.transform(X_train))
#X_test_pca = pd.DataFrame(fit.transform(X_test))
#X_for_future_pca = pd.DataFrame(fit.transform(X_for_future))

top_features = models.feature_ranking(X_train,y_train)

X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
X_for_future_top = X_for_future[top_features]
X_for_future_benchmark = X_for_future[ticker].values.reshape(-1,1)

print("Top Features Selected are", top_features)
#print("The number of Principle Componenets selected = ", n_components)
##############################################################################
# 								     LEARN
##############################################################################
print("Training the Models")

benchmark_model = models.benchmark_model(X_train_benchmark,y_train)
benchmark_score = benchmark_model.score(X_test_benchmark,y_test)
benchmark_preds = benchmark_model.predict(X_test_benchmark.reshape(-1,1))
benchmark_future_preds = benchmark_model.predict(X_for_future_benchmark)

lasso , names = models.LASSO_model(X_train,y_train)

txt = "Lasso model: " + models.pretty_print_linear(lasso.coef_, names, sort = True)

X_train_few = X_train[names]
X_test_few = X_test[names]
X_for_future_few = X_for_future[names]

lasso_train_preds = lasso.predict(X_train_few)
lasso_preds = lasso.predict(X_test_few)

lasso2 = models.LASSO_model2(X_train_top,y_train)
sgd = models.SGD_model(X_train_top,y_train)
#sgd2 = models.SGD_model(X_train_few,y_train)

lasso2_train_preds = lasso2.predict(X_train_top)
lasso2_preds = lasso2.predict(X_test_top)
sgd_train_preds = sgd.predict(X_train_top)
sgd_preds = sgd.predict(X_test_top)
#sgd2_preds = sgd2.predict(X_test_few)
#sgd_w_pca_preds = sgd_w_pca.predict(X_test_pca)

print("benchmark score = ", benchmark_score)
print("lasso score =" , lasso.score(X_test_few,y_test))
print("lasso2 score =" , lasso2.score(X_test_top,y_test))
print("sgd score =" , sgd.score(X_test_top,y_test))
#print("sgd2 score =" , sgd2.score(X_test_few,y_test))
#print "sgd_wPCA score =" , sgd_w_pca.score(X_test_pca,y_test)


future_lasso_preds = lasso.predict(X_for_future_few)
future_lasso2_preds = lasso2.predict(X_for_future_top)
future_sgd_preds = sgd.predict(X_for_future_top)
#future_sgd2_preds = sgd2.predict(X_for_future_few)
#future_sgd_w_pca_preds = sgd_w_pca.predict(X_for_future_pca)

##############################################################################
#									ENSEMBLE
##############################################################################
train_compare = pd.DataFrame(index=X_train.index)
train_compare['Lasso_1'] = lasso_train_preds
train_compare['Lasso_2'] = lasso2_train_preds
train_compare['SGD'] = sgd_train_preds
#test_compare['Average'] = test_compare.mean(axis=1)
#test_compare['True'] = y_train

ensemble_model = models.ENSEMBLE_model(train_compare,y_train)

test_compare = pd.DataFrame(index=X_test.index)
test_compare['Lasso_1'] = lasso_preds
test_compare['Lasso_2'] = lasso2_preds
test_compare['SGD'] = sgd_preds
#test_compare['Average'] = test_compare.mean(axis=1)

ensemble_preds = ensemble_model.predict(test_compare)
ensemble_score = ensemble_model.score(test_compare,y_test)

print("ensemble score =" , ensemble_score)

future_compare = pd.DataFrame(index=X_for_future.index)
future_compare['Lasso_1'] = future_lasso_preds
future_compare['Lasso_2'] = future_lasso2_preds
future_compare['SGD'] = future_sgd_preds
future_ensemble_preds = ensemble_model.predict(future_compare)

##############################################################################
#									PLOT
##############################################################################
print("")
print("Predicting the Future")

txt = txt + "ensemble score =" , ensemble_score

train_plt = plt.subplot2grid((3,6), (0,0), rowspan =2 , colspan=4)
test_plt = plt.subplot2grid((3,6), (0,4), rowspan =2, colspan=2, sharey = train_plt)
future_plt = plt.subplot2grid((3,6), (2,0), rowspan =1, colspan=2)
future_plt.text(days+1,(future_lasso_preds.max()-future_lasso_preds.min())/2 +future_lasso_preds.min(),txt)

train_plt.plot(y_train)

test_plt.plot(lasso_preds)
test_plt.plot(lasso2_preds)
test_plt.plot(sgd_preds)
test_plt.plot(ensemble_preds)
test_plt.plot(y_test.values)

future_plt.plot(range(1,days +1),future_lasso_preds)
future_plt.plot(range(1,days +1),future_lasso2_preds)
future_plt.plot(range(1,days +1),future_sgd_preds)
future_plt.plot(range(1,days +1),future_ensemble_preds)
#future_plt.plot(range(1,days +1),future_sgd_w_pca_predictions)
future_plt.set_title('Price of Stock # of Days into the future')
plt.show()