from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score


iris = load_iris()
logreg = LinearRegression()

kfold = KFold(n_splits=3)
print('Cross-validation scores: {}'.format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)
))


shuffled_kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffled_kfold)
print('Cross-validation scores: {}'.format(scores))
print('Mean score: {:.3f}'.format(scores.mean()))
