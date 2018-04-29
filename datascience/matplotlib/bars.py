import numpy as np
import matplotlib.pyplot as plt

# generate data
x = np.array([1, 2, 3, 4, 5])
y = np.array([100, 200, 300, 400, 500])


# default
# plt.bar(x, y)


# no separate
# plt.bar(x, y, width=1.0)


# without borderline
# plt.bar(x, y, linewidth=0, color='magenta')

# with borderline
# plt.bar(x, y, linewidth=4, color='magenta', edgecolor='red')


# align
# plt.bar(x, y, align='center')


# with labels
# label = ['A', 'B', 'C', 'D', 'E']
# plt.bar(x, y, tick_label=label, align='center')
# plt.title('Title')
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.grid(True)


# error range
# plt.bar(x, y, xerr=0.5, ecolor='red')

# yerror = [10, 20, 30, 40, 50]
# plt.bar(x, y, yerr=yerror, ecolor='red')
# plt.bar(x, y, yerr=yerror, ecolor='red', capsize=10)


# log scale
# plt.bar(x, [100, 1000, 10000, 100000, 1000000], log=True)


# stacked charts
y2 = np.array([100, 200, 300, 400, 500])
y3 = np.array([1000, 800, 600, 400, 200])
p1 = plt.bar(x, y2)
p2 = plt.bar(x, y3, bottom=y2)
plt.legend((p1[0], p2[0]), ('class 1', 'class 2'))

plt.show()
