import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from mglearn.datasets import load_extended_boston


X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge(alpha=0.1).fit(X_train, y_train)

for a, m in zip([1, 0.1, 0.0001], ['s', '^', 'v']):
    lasso = Lasso(alpha=a, max_iter=100000).fit(X_train, y_train)
    print('\nalpha={}'.format(a))
    print('Training set score: {:.2f}'.format(lasso.score(X_train, y_train)))
    print('Test set score: {:.2f}'.format(lasso.score(X_test, y_test)))
    print('#Features used: {}'.format(np.sum(lasso.coef_ != 0)))

    plt.plot(lasso.coef_, m, label='Lasso alpha={}'.format(a))

plt.plot(ridge.coef_, 'o', label='Ridge alpha=0.1')
plt.legend(ncol=2, loc='best')
plt.ylim(-25, 25)
plt.xlabel('Coefficient index')
plt.ylabel('Coefficient magnitude')
plt.show()
