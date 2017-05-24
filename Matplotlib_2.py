import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from matplotlib import style
import random

style.use('fivethirtyeight')

# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)

# def animate(i):
#     graph_data = open('example.txt','r').read()
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#     for line in lines:
#         if len(line) > 1:
#             x, y = line.split(',')
#             xs.append(x)
#             ys.append(y)
#     ax1.clear()
#     ax1.plot(xs, ys)

# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()

fig = plt.figure()

def create_plots():
	xs = []
	ys = []
	
	for i in range(10):
		x = i
		y = random.randrange(10)

		xs.append(x)
		ys.append(y)
	return xs, ys


ax1 = plt.subplot2grid((6,1),(0,0), rowspan=1, colspan =1)
ax2 = plt.subplot2grid((6,1),(1,0), rowspan=4, colspan =1)
ax3 = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan =1)

# add_subplot syntax
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(212)

x,y = create_plots()
ax1.plot(x,y)

x,y = create_plots()
ax2.plot(x,y)

x,y = create_plots()
ax3.plot(x,y)

plt.show()