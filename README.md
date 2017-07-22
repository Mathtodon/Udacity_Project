# Udacity_Project
This contains my Udacity Capstone Project

This script will take in the {ticker} and {days} variables set by the user and 
will create 5 different models using Lasso Regression and Stochastic Gradient 
Decent Regression to predict the stock price.

The data will be split into parts for training, testing, and future predictions.

The ouput will be a 3 plots showing these different sections of the data with 
the models' predictions. 

Historically the script takes around 560 seconds to finish running which is 
equivalent to 9.5 minutes.


User Inputs:

ticker = 'MMM'
	
	This will be any Ticker for a company which had full data between 2000-01-01 and 2016-12-31

start_date = '2000-01-01'
	
	This will be any date between 2000-01-01 and 2016-12-31

end_date = '2016-12-30'

	This will be any date between 2000-01-01 and 2016-12-31 but after start_date

days = 28
	
	This will be the number of days into the future the user would like to predict
	it can be any integer greater than 0