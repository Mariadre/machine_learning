import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)


param_C = [0.001, 0.01, 0.1, 1, 10, 100]
param_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = [{'kernel': ['rbf'],
               'C': param_C,
               'gamma': param_gamma},
              {'kernel': ['linear'],
               'C': param_C}]

grid_search = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


print('Best parameters:', grid_search.best_params_)
print('Best cv score: {:.3f}'.format(grid_search.best_score_))
print('Test set score: {:.3f}'.format(grid_search.score(X_test, y_test)))

results = pd.DataFrame(grid_search.cv_results_)
print(results[results['param_kernel'] != 'rbf'].head())



