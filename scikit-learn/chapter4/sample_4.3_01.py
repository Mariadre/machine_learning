import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


X, y = make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins=bins)

encoder = OneHotEncoder(sparse=False)
X_binned = encoder.fit_transform(which_bin)
line_binned = encoder.transform(np.digitize(line, bins=bins))


X_product = np.hstack([X_binned, X * X_binned])
line_product = np.hstack([line_binned, line * line_binned])

lr = LinearRegression().fit(X_product, y)
plt.plot(line, lr.predict(line_product), label='Linear Regression Product')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input Feature')
plt.legend(loc='best')
plt.show()
