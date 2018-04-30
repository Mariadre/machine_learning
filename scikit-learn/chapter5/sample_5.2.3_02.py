from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
model = grid_search.fit(X_train, y_train)


print('Best parameters:', grid_search.best_params_)
print('Best score: {:.3f}'.format(grid_search.best_score_))
print('Best Estimator:\n', grid_search.best_estimator_)
print('Test set score: {:.3f}'.format(grid_search.score(X_test, y_test)))
