from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np 
import pandas as pd

def DTR_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = TimeSeriesSplit(n_splits = 10)
    cv_sets.split(X,y)

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'max_depth':range(5,60), 'max_features':['auto']}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def RFR_model(X, y):
    """ Performs grid search over the 'n_estimators' parameter for a 
        random forest regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = TimeSeriesSplit(n_splits = 10)
    cv_sets.split(X,y)

    # Create a decision tree regressor object
    regressor = RandomForestRegressor()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'n_estimators':range(1,len(X.columns))}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def SGD_model(X, y):

    #scaler = StandardScaler()
    #names = X_names
    
    # Create cross-validation sets from the training data
    cv_sets = TimeSeriesSplit(n_splits = 10)
    cv_sets.split(X,y)

    # Create a decision tree regressor object
    regressor = linear_model.SGDRegressor(n_iter=500)

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'loss':['squared_loss','huber'],'penalty':['none','l2','l1'],'alpha':[.0001,.001,.01,.1],'l1_ratio':[.15,.30,.5,.65]}

    # Transform 'performance_metric' into a scoring function using 'make_scorer' 
    #scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def LASSO_model2(X,y):
    cv_sets = TimeSeriesSplit(n_splits = 10)
    cv_sets.split(X,y)
      
    regressor = Lasso()
    params = {'alpha':[.3,.4,.5]}

    grid = GridSearchCV(regressor, params, cv=cv_sets)

    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def ENSEMBLE_model(X,y):
    regressor = LinearRegression()
    return regressor.fit(X,y)


def LASSO_model(X,y):
    n = 3
    names = list(X.columns)
    #use linear regression as the model
    lr = LinearRegression()
    #scaler = StandardScaler()
    #rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X,y)

    X_train_top = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))[:n]
    X_names = ['Day_N']
    for i in range(n):
        X_names = X_names+[X_train_top[i][1]]
        
    names = X_names
    X_few = X[names]
      
    lasso = Lasso(alpha=.3)
    lasso_model = lasso.fit(X_few, y)
    return lasso_model, names
    
def SVR_model(X, y):
    # Create cross-validation sets from the training data
    cv_sets = TimeSeriesSplit(n_splits = 10)
    cv_sets.split(X,y)

    # Create a decision tree regressor object
    regressor = svm.SVR()

    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 100
    params = {'kernel':['linear','poly'],
              'degree':[2,3],
              'C':[1.0,10.0],
              'gamma':['auto']}

    # Create the grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def benchmark_model(X, y):
    # Create a decision tree regressor object
    regressor = linear_model.LinearRegression()

    # Fit the benchmark model to the data 
    benchmark = regressor.fit(X, y)

    return benchmark

def choose_components(X):
    components = range(1,10)
    scores = []
    
    for n in components:
        pca = PCA(n_components=n)
        fit = pca.fit(X)
        # summarize components
        #print("Explained Variance: %s") % fit.explained_variance_ratio_
        scores.append(sum(fit.explained_variance_ratio_))

    values = [scores[0]]
    for i in range(1,len(scores)):
        additional_value = (scores[i]-scores[i-1])
        if additional_value <.01:
            n_components = i
            break
        values.append(additional_value)
    return n_components

def feature_ranking(X,y):
    names = list(X.columns)
    #use linear regression as the model
    lr = LinearRegression()
    cv_sets = TimeSeriesSplit(n_splits = 10)
    #scaler = StandardScaler()
    #rank all features, i.e continue the elimination until the last one
    rfecv = RFECV(lr, cv = cv_sets)
    rfecv.fit(X,y)
     
    feature_ranks = sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), names))
    feature_ranks_df = pd.DataFrame(feature_ranks)
    top_features = list(feature_ranks_df[feature_ranks_df[0]==1][1])
    return top_features[:21]

def pretty_print_linear(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 4), name) for coef, name in lst)

def visualize_data():
    ## code from https://pythonprogramming.net/stock-price-correlation-table-python-programming-for-finance/
    ## after going through his videos for ideas, I found that this was a useful visualization
    import matplotlib.pyplot as plt
    df = pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr = df.corr()
    #print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()
    
#visualize_data()