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


model_list = [LASSO_model2(), LASSO_model(), SVR_model(), SGD_model(), KNR_model(), benchmark_model()]

i = 1
for model in model_list:
	model_ i = 
	model(X)