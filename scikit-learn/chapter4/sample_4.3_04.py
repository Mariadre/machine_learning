import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
# print(poly.get_feature_names())


ridge = Ridge().fit(X_train_scaled, y_train)
print('Ridge(scaled):')
print('Score without interactions: {:.3f}'.format(ridge.score(X_train_scaled, y_train)))
print('Test Score without interactions: {:.3f}'.format(ridge.score(X_test_scaled, y_test)))

ridge = Ridge(alpha=1.0).fit(X_train_poly, y_train)
print('\nRidge(polynomial):')
print('Score with interactions: {:.3f}'.format(ridge.score(X_train_poly, y_train)))
print('Test Score with interactions: {:.3f}'.format(ridge.score(X_test_poly, y_test)))

lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train)
print('\nLasso(polynomial):')
print('Score with interactions: {:.3f}'.format(lasso.score(X_train_poly, y_train)))
print('Test Score with interactions: {:.3f}'.format(lasso.score(X_test_poly, y_test)))
print('Feature Used: {}'.format(np.sum([lasso.coef_ != 0])))
