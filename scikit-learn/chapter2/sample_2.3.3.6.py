import numpy as np
import mglearn
import matplotlib.pyplot as plt
from mglearn.datasets import make_blobs
from sklearn.svm import LinearSVC

X, y = make_blobs(random_state=42, centers=3)

linear_svc = LinearSVC().fit(X, y)
print('Coefficient Shape', linear_svc.coef_.shape)
print('Intercept Shape', linear_svc.intercept_.shape)


mglearn.plots.plot_2d_classification(linear_svc, X, fill=True, alpha=.6)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)

for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)

plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line Class 1', 'Line Class 2', 'Line Class 3'], loc=(1.01, 0.3))
plt.show()
