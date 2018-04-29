import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
# print(np.bincount(X[:, 0]))

fig, axes = plt.subplots(1, 2, figsize=(10, 6))
bins = np.bincount(X[:, 0])
axes[0].bar(range(len(bins)), bins, color='purple')
axes[0].set_ylabel('Number of appearances')
axes[0].set_xlabel('Value')
axes[0].set_title('Original data')


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ridge = Ridge().fit(X_train, y_train)
print('Test Score: {:.3f}'.format(ridge.score(X_test, y_test)))


X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

ridge = Ridge().fit(X_train_log, y_train)
print('Test Score: {:.3f}'.format(ridge.score(X_test_log, y_test)))

axes[1].hist(X_train_log[:, 0], bins=25, color='lightgray')
axes[1].set_ylabel('Number of appearances')
axes[1].set_xlabel('Value')
axes[1].set_title('Logarithm data')

plt.show()



