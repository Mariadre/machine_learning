from sklearn.linear_model import LinearRegression
from mglearn.datasets import make_wave, load_extended_boston
from sklearn.model_selection import train_test_split

# Linear Regression with simple data
X, y = make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)

print('wave dataset:')
print('coefficent: {}'.format(lr.coef_))
print('intercept:  {}'.format(lr.intercept_))

print('Train Set: {:.2f}'.format(lr.score(X_train, y_train)))
print('Test  Set: {:.2f}'.format(lr.score(X_test, y_test)))


# Linear Regression with complex data
X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print('\nextended Boston dataset:')
print('Train Set: {:.2f}'.format(lr.score(X_train, y_train)))
print('Test  Set: {:.2f}'.format(lr.score(X_test, y_test)))












