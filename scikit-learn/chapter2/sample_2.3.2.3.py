from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave


X, y = make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

print('Accuracy Of Train Set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy Of Test  Set: {:.2f}'.format(knn.score(X_test, y_test)))
