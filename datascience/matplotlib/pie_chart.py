import numpy as np
import matplotlib.pyplot as plt

x = np.arange(100, 600, 100)


# default
label = ['A', 'B', 'C', 'D', 'E']
# plt.pie(x, labels=label)


# 時計回りで開始位置を垂直から
# plt.pie(x, startangle=90, counterclock=False)


# 'cut off' expression
# plt.pie(x, labels=label, counterclock=False, startangle=90, explode=[0.2, 0, 0, 0, 0])


# color
grays = ['0.1', '0.3', '0.5', '0.7', '0.9']
# plt.pie(x, labels=label, counterclock=False, startangle=90, colors=grays)


# shadow
# plt.pie(x, shadow=True)


# label settings
# plt.pie(x, labels=label,counterclock=False, startangle=90,
#         labeldistance=0.50, textprops={'color': 'white', 'weight': 'bold'})


# show each components value
# plt.pie(x, labels=label, autopct='%1.1f%%', pctdistance=0.7)


# donuts graph
# plt.pie(x, labels=label, counterclock=False, startangle=90)
# center_circle = plt.Circle((0,0), 0.6, color='black', fc='white', linewidth=0   )
# fig = plt.gcf()
# fig.gca().add_artist(center_circle)


# double donuts graph
inner = np.array([150, 250, 300, 350, 450])
fmt = '%1.1f%%'

plt.pie(x, labels=label, counterclock=False, startangle=90, autopct=fmt, pctdistance=0.85)
plt.pie(inner, radius=0.7, counterclock=False, startangle=90, autopct=fmt, pctdistance=0.75)

center_circle = plt.Circle((0, 0), 0.4, fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(center_circle)














# 正円になるように調整
plt.axis('equal')

plt.show()
