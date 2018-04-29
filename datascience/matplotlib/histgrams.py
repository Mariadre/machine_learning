import numpy as np
import matplotlib.pyplot as plt

# generate data
x = np.random.normal(50, 10, 1000)


# default
# plt.hist(x)


# bin(#bars)
# plt.hist(x, bins=16)


# range
# plt.hist(x, range=(30, 90))


# normalize
# plt.hist(x, normed=True)


# accumulate
# plt.hist(x, cumulative=True)


# log scale
# plt.hist(x, log=True)


# bar width(= separate)
# plt.hist(x, rwidth=.8, color='cyan')


# not fill
# plt.hist(x, histtype='step')


# align
# fig = plt.figure(figsize=(10, 5))
#
# subplot = fig.add_subplot(1, 2, 1)
# subplot.hist(x, align='right')
#
# subplot = fig.add_subplot(1, 2, 2)
# subplot.hist(x, align='left')


# orientation
# plt.hist(x, orientation='horizontal')


# bar stacked
x2 = np.random.normal(20, 10, 1000)
# plt.hist([x, x2], histtype='barstacked')
# plt.hist([x, x2], stacked=True, histtype='step')

# bar non-stacked
# plt.hist([x, x2], stacked=False)


# label, legend
plt.hist([x, x2], label=['cat1', 'cat2'])
plt.legend()

plt.show()
