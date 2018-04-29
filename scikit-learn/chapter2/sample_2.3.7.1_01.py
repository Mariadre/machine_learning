import matplotlib.pyplot as plt
import mglearn
from sklearn.svm import LinearSVC
from mglearn.datasets import make_blobs

X, y = make_blobs(centers=4, random_state=8)
y = y % 2

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(loc=2)
plt.show()
