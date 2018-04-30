from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [
    {'classifier': [SVC()],
     'preprocessing': [StandardScaler(), None],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None],
     'classifier__max_features': [1, 2, 3]}]

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)


print('Best parameters:', grid.best_params_)
print('Best cross-validation score: {:.3f}'.format(grid.best_score_))
print('Test Accuracy: {:.3f}'.format(grid.score(X_test, y_test)))
