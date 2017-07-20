import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib import style

import numpy as np
import urllib.request
import datetime as dt

style.use('fivethirtyeight')

def first_plot():
	x = [1,2,3]
	y = [5,7,4]

	x2 = [1,2,3]
	y2 = [10,14,12]

	plt.plot(x,y, label = 'first line')
	plt.plot(x2,y2, label = 'second line')
	plt.xlabel('Plot Number')
	plt.ylabel('Important Var')


	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()

def bar_chart():
	x = [2,4,6,8,10]
	y = [6,7,8,2,4]

	x2 = [1,3,5,7,9]
	y2 = [7,8,2,4,2]

	plt.bar(x,y, label='Bars1', color = 'r')
	plt.bar(x2,y2, label='Bars2', color = 'c')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()



def histogram():
	population_ages = [22,55,62,45,21,22,34,42,42,4,99,102,110,120,121,130,111,115,112,80,75,65,54,44,43,42,48]

	#ids = [x for x in range(len(population_ages))]

	bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

	plt.hist(population_ages, bins, histtype='bar', rwidth=.8)

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()


def scatter_plot():
	x = [1,2,3,4,5,6,7,8]
	y = [5,2,4,2,1,4,5,2]

	plt.scatter(x,y, label='skitscat', color='c', s=100, marker='*')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()


def stack_plot():
	days = [1,2,3,4,5]

	sleeping = [7,8,6,11,7]
	eating =   [2,3,4,3,2]
	working =  [7,8,7,2,2,]
	playing =  [8,5,7,8,13]

	plt.plot([],[], color ='m', label='Sleeping', linewidth = 5)
	plt.plot([],[], color ='c', label='Eating', linewidth = 5)
	plt.plot([],[], color ='r', label='Working', linewidth = 5)
	plt.plot([],[], color ='k', label='Playing', linewidth = 5)


	plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])


	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()



def pie_chart():
	days = [1,2,3,4,5]

	sleeping = [7,8,6,11,7]
	eating =   [2,3,4,3,2]
	working =  [7,8,7,2,2,]
	playing =  [8,5,7,8,13]

	slices = [7,2,2,13]
	activities = ['sleeping','eating','working','playing']
	cols = ['c','m','r','b']

	plt.pie(slices, 
		labels=activities, 
		colors=cols, 
		startangle=90, 
		shadow=True, 
		explode=(0,0.1,0,0),
		autopct='%1.1f%%')

	#plt.xlabel('X')
	#plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	#plt.legend()
	plt.show()


def csv_file():
	import csv

	x = []
	y = []

	with open('example.txt','r') as csvfile:
		plots = csv.reader(csvfile, delimiter = ',')
		for row in plots:
			x.append(int(row[0]))
			y.append(int(row[1]))

	plt.plot(x,y, label='Loaded from file!')


	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()





def csv_file_with_np():
	import numpy as np

	x, y = np.loadtxt('example.txt', delimiter=',' , unpack=True)

	plt.plot(x,y, label='Loaded from file!')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Interesting Graph\nCheck it out')
	plt.legend()
	plt.show()


def bytespdate2num(fmt, encoding='utf-8'):
	strconverter = mdates.strpdate2num(fmt)
	def bytesconverter(b):
		s = b.decode(encoding)
		return strconverter(s)
	return bytesconverter



def graph_data(stock):

	fig = plt.figure()
	ax1 = plt.subplot2grid((1,1), (0,0))

	stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
	source_code = urllib2.urlopen(stock_price_url).read().decode()
	stock_data = []
	split_source = source_code.split('\n')
	for line in split_source:
		split_line = line.split(',')
		if len(split_line) == 6:
			if 'values' not in line and 'labels' not in line:
				stock_data.append(line)

	date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
														  delimiter = ',',
														  unpack=True,
														  # %Y = full year. 2015
														  # %y = partial year 15
														  # %m = number month
														  # %d = number day
														  # %H = hours
														  # %M = minutes
														  # %S = seconds
														  # 12-06-2014   %m-%d-%Y
														  converters={0: bytespdate2num('%Y%m%d')})

	# date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
	# 													  delimiter = ',',
	# 													  unpack=True)

	# dateconv = np.vectorize(dt.datetime.fromtimestamp)
	# date = dateconv(date)


	ax1.plot_date(date, closep,'-', label = stock)
	ax1.axhline(closep[0], color='k', linewidth=5)
	ax1.fill_between(date, closep, closep[0], where=(closep >closep[0]), facecolor='g', alpha =0.5)
	ax1.fill_between(date, closep, closep[0], where=(closep <closep[0]), facecolor='r', alpha =0.5)

	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)
	ax1.grid(True, color ='g', linestyle='-' )
	#ax1.xaxis.label.set_color('c')
	#ax1.yaxis.label.set_color('r')
	ax1.set_yticks([0,25,50,75])

	ax1.spines['left'].set_color('c')
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.spines['left'].set_linewidth(5)

	ax1.tick_params(axis='x', colors='#f06215')



	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title(stock)
	plt.legend()

	plt.subplots_adjust(left=.10,bottom=.18,right=.94,top=.87,wspace=.2,hspace=0)
	plt.show()


MA1 = 5
MA2 = 20

def moving_average(values, window):
	weights = np.repeat(1.0, window)/window
	smas = np.convolve(values, weights, 'valid')
	return smas

def high_minus_low(highs, lows):
	return highs-lows


def candlestick_graph(stock):

	fig = plt.figure(facecolor='#f0f0f0')
	ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 1, colspan = 1)
	plt.title(stock)
	plt.ylabel('H-L')
	ax2 = plt.subplot2grid((6,1), (1,0), rowspan = 4, colspan = 1, sharex=ax1)
	plt.ylabel('Price')
	ax2v = ax2.twinx()
	ax3 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex=ax1)
	plt.ylabel('MAVGs')

	stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=1y/csv'
	source_code = urllib.request.urlopen(stock_price_url).read().decode()
	stock_data = []
	split_source = source_code.split('\n')
	for line in split_source:
		split_line = line.split(',')
		if len(split_line) == 6:
			if 'values' not in line and 'labels' not in line:
				stock_data.append(line)

	date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
														  delimiter = ',',
														  unpack=True,
														  converters={0: bytespdate2num('%Y%m%d')})

	x = 0
	y = len(date)
	ohlc = []

	while x <y:
		append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
		ohlc.append(append_me)
		x+=1

	ma1 = moving_average(closep,MA1)
	ma2 = moving_average(closep,MA2)
	start = len(date[MA2-1:])

	h_1 = list(map(high_minus_low, highp, lowp))

	ax1.plot_date(date[-start:],h_1[-start:],'-', linewidth= 2, label='H-L')
	ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='lower'))


	candlestick_ohlc(ax2, ohlc[-start:], width=0.4, colorup='#77d879', colordown='#db3f3f')

	#ax1.plot(date,closep)
	#ax1.plot(date,openp)

	#for label in ax2.xaxis.get_ticklabels():
	#	label.set_rotation(45)

	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune='upper'))
	ax2.grid(True)

	bbox_props = dict(boxstyle='larrow',fc='w', ec='k')
	ax2.annotate(str(closep[-1]), (date[-1], closep[-1]),
				 xytext = (date[-1]+4,closep[-1]), bbox=bbox_props)

#	# Annotation example with arrow
#	ax2.annotate('Bad News!',(date[6],highp[6]),
#				 xytext=(0.8, 0.9), textcoords='axes fraction',
#				 arrowprops = dict(facecolor='grey', color ='grey'))

	# Font dict Example
#	font_dict = {'family':'serif',
#				 'color': 'darkred',
#				 'size':15}
#	ax2.text(date[10], closep[1],'Text Example', fontdict=font_dict)
	ax2v.plot([],[], color = '#0079a3', alpha=0.4, label='Volume')
	ax2v.fill_between(date[-start:],0, volume[-start:], facecolor = '#0079a3', alpha=.4)
	ax2v.axes.yaxis.set_ticklabels([])
	ax2v.grid(False)
	ax2v.set_ylim(0, 3*volume.max())

	ax3.plot(date[-start:], ma1[-start:], linewidth=1, label=(str(MA1)+'MA'))
	ax3.plot(date[-start:], ma2[-start:], linewidth=1,label=(str(MA2)+'MA'))

	ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], 
					 where=(ma1[-start:]<ma2[-start:]), 
					 facecolor='r', 
					 edgecolor= 'g',
					 alpha=.5)
	ax3.fill_between(date[-start:], ma2[-start:], ma1[-start:], 
					 where=(ma1[-start:]>ma2[-start:]), 
					 facecolor='g', 
					 edgecolor= 'g',
					 alpha=.5)

	ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='upper'))
	
	for label in ax3.xaxis.get_ticklabels():
		label.set_rotation(45)

	plt.setp(ax2.get_xticklabels(), visible = False)
	plt.subplots_adjust(left=.10,bottom=.25,right=.88,top=.87,wspace=.2,hspace=0)

	ax1.legend()
	leg = ax1.legend(loc=9, ncol=2,prop={'size':11})
	leg.get_frame().set_alpha(0.4)
	
	ax2v.legend()
	leg = ax2v.legend(loc=9, ncol=2,prop={'size':11})
	leg.get_frame().set_alpha(0.4)
	
	ax3.legend()
	leg = ax3.legend(loc=9, ncol=2,prop={'size':11})
	leg.get_frame().set_alpha(0.4)

	plt.show()
	#fig.savefig('AMZN.png', facecolor=fig.get_facecolor())


# graph_data('ebay')
candlestick_graph('AMZN')


