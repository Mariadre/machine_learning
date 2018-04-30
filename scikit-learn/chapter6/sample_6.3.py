from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)


pipe = Pipeline([("scaler", MinMaxScaler()),
                 ('svm', SVC())])


param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: {:.3f}'.format(grid.best_score_))
print('Test set accuracy: {:.3f}'.format(grid.score(X_test, y_test)))
print('Best cross-validation parameters:', grid.best_params_)
