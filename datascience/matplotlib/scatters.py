import numpy as np
import matplotlib.pyplot as plt

# generate data
x = np.random.rand(100)
y = np.random.rand(100)


fig = plt.figure(figsize=(15, 5))


# default
subplot = fig.add_subplot(2, 3, 1)
subplot.scatter(x, y)


# change marker & color
subplot = fig.add_subplot(2, 3, 2)
subplot.scatter(x, y, s=600, c='pink', alpha=.6, linewidths=2, edgecolor='red')

subplot = fig.add_subplot(2, 3, 3)
subplot.scatter(x, y, marker='^', color='orchid')


# show grid , title, axis label
subplot = fig.add_subplot(2, 3, 4)
subplot.scatter(x, y)
subplot.grid(True)
subplot.set_title('Title')
subplot.set_xlabel('x-axis')
subplot.set_ylabel('y-axis')


# gradiation
subplot = fig.add_subplot(2, 3, 5)
im = subplot.scatter(x, y, c=x, cmap='Reds')
fig.colorbar(im)

subplot = fig.add_subplot(2, 3, 6)
im = subplot.scatter(x, y, c=x, cmap='Blues', vmax=0.4, vmin=0.6)
fig.colorbar(im)

plt.show()
