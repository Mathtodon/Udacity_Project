import numpy as np 
import scipy as sp 
import pandas as pd 
import quandl
import matplotlib.pyplot as plt
#%matplotlib inline 
from matplotlib import style
style.use('ggplot')
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn import linear_model



######################################################################
######################################################################
# 							DATA PREP
######################################################################
######################################################################
ticker = 'AAPL'
#days = input('How many days into the future would you like to predict for AAPL?: ')
days = 1
df = pd.read_csv('sp500_joined_closes.csv', index_col = 0)
tickers = df.columns.values.tolist()
#dates = df.index.
df.fillna(0, inplace = True)

df = df.replace([np.inf, -np.inf], np.nan)
#df.dropna(inplace=True)
df.fillna(0, inplace=True)

y_shifted = df[ticker].shift(-days)[:-days]
X = df[:-days]
X_for_future = df[-days:]

X_train = X[:-20]
X_train.fillna(0, inplace=True)
X_test = X[-20:]
y_train = y_shifted[:-20]
y_test = y_shifted[-20:]

######################################################################
######################################################################
#						Regression Functions
######################################################################
######################################################################

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


def SVR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = svm.SVR()

    # Create a dictionary for the parameters
    params = {'kernel':['rbf','linear'], 
              'C':[1.0,10.0,100.0,1000.0],
              'gamma':['auto',.05,.075,.1,1]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_



def DTR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameters
    params = {'max_depth':range(1,11)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def RFR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    #cv_sets = TimeSeriesSplit(n_splits = 10).split(X_train)

    # Create a decision tree regressor object
    regressor = RandomForestRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'max_depth':range(1,20)}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    #grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def KNR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = KNeighborsRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'k':range(1,20), 'weights':['uniform','distance']}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def RNR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    #cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)

    # Create a decision tree regressor object
    regressor = RadiusNeighborsRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'radius':range(1,20),'weights':['uniform','distance']}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


def Linear_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    #cv_sets = TimeSeriesSplit(n_splits = 10)

    # Create a decision tree regressor object
    regressor = linear_model.LinearRegression(normalize = True, copy_X = True, n_jobs = -1)

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Return the optimal model after fitting the data
    return regressor.fit(X, y)


######################################################################
######################################################################
#						
######################################################################
######################################################################

predictions = y_test

model_1 = Linear_model(X_train, y_train)
preds_1 = pd.Series(model_1.predict(X_test))
predictions['predictions_1'] = preds_1.values 

#model_2 = RNR_model(X_train, y_train)
#preds_2 = model_2.predict(X_test)
#predictions['predictions_2'] = preds_2.values

#model_3 = KNR_model(X_train, y_train)
#preds_3 = model_3.predict(X_test)
#predictions['predictions_3'] = preds_3.values

model_4 = RFR_model(X_train, y_train)
preds_4 = model_4.predict(X_test)
predictions['predictions_4'] = preds_4.values

print(predictions[0])