import matplotlib.pyplot as plt
import mglearn
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston


X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

for a, m in zip([1, 10, 0.1], ['s', '^', 'v']):
    ridge = Ridge(alpha=a).fit(X_train, y_train)
    print('Training set(alpha={}): {:.2f}'.format(a, ridge.score(X_train, y_train)))
    print('Test set(alpha={}): {:.2f}\n'.format(a, ridge.score(X_test, y_test)))

    plt.plot(ridge.coef_, m, label='Ridge alpha={}'.format(a))


plt.plot(lr.coef_, 'o', label='Linear Regression')
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()

mglearn.plots.plot_ridge_n_samples()
plt.show()
