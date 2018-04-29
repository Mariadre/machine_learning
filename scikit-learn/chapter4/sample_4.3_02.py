import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_wave
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


X, y = make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)


poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X)
line_poly = poly.transform(line)
print(poly.get_feature_names())


lr = LinearRegression().fit(X, y)
lr_poly = LinearRegression().fit(X_poly, y)

plt.plot(line, lr.predict(line), label='Linear Regression')
plt.plot(line, lr_poly.predict(line_poly), label='Polynomial Linear Regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel('Regression output')
plt.xlabel('Input feature')
plt.legend(loc='best')
plt.show()
