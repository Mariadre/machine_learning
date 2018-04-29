import numpy as np
import matplotlib.pyplot as plt


# generate data
x = np.arange(1, 6)
y = np.array([100, 300, 200, 500, 400])
print(y)

# default
# plt.plot(x, y)


# linewidth, color
# plt.plot(x, y, linewidth=4, color='red')


# linestyle
# plt.plot(x, y, linestyle='solid')      # '-'
# plt.plot(x, y/2, linestyle='dashed')   # '--'
# plt.plot(x, y/3, linestyle='dashdot')  # '-.'
# plt.plot(x, y/4, linestyle='dotted')   # ':'


# marker
# plt.plot(x, y, marker='D', markersize='12', markerfacecolor='lightblue',
#          markeredgewidth=3, markeredgecolor='blue')
# plt.plot(x, y, marker='s', markersize='20', markeredgewidth=2, markeredgecolor='black',
#          markerfacecolor='blue', markerfacecoloralt='yellow', fillstyle='left')


# anti-aliase
# plt.plot(x, y, linewidth=5, antialiased=False)
# plt.plot(x, y/2, linewidth=5)


# label, legend
plt.title('Title')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
p1 = plt.plot(x, y)
p2 = plt.plot(x, y/2, linestyle='--')
plt.legend((p1[0], p2[0]), ('Class 1', 'Class 2'), loc='best')


plt.show()
