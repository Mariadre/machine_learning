import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder


X, y = make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

tree = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
lr = LinearRegression().fit(X, y)
print('Tree Score: {:.3f}'.format(tree.score(X, y)))
print('Linear Regression Score: {:.3f}'.format(lr.score(X, y)))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(line, tree.predict(line), label='Decision Tree')
axes[0].plot(line, lr.predict(line), label='Linear Regression', linestyle='--')
axes[0].plot(X[:, 0], y, 'o', c='k')
axes[0].set_ylabel('Regression output')
axes[0].set_xlabel('Input feature')
axes[0].legend(loc='best')


bins = np.linspace(-3, 3, 11)
print('bins: {}'.format(bins))

which_bin = np.digitize(X, bins=bins)
print('\nData points:\n', X[:5])
print('\nBin membership for data points:\n', which_bin[:5])


encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
print('X_binned.shape: {}'.format(X_binned.shape))


line_binned = encoder.transform(np.digitize(line, bins=bins))
lr = LinearRegression().fit(X_binned, y)
tree = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
print('Tree Score: {:.3f}'.format(tree.score(X_binned, y)))
print('Linear Regression Score: {:.3f}'.format(lr.score(X_binned, y)))

axes[1].plot(line, tree.predict(line_binned), label='Decision Tree Binned')
axes[1].plot(line, lr.predict(line_binned), label='Linear Regression Binned', linestyle='--')
axes[1].plot(X[:, 0], y, 'o', c='k')
axes[1].vlines(bins, -3, 3, linewidth=1, alpha=.2)
axes[1].legend(loc='best')
axes[1].set_ylabel('Regression output')
axes[1].set_xlabel('Input feature')
plt.show()
