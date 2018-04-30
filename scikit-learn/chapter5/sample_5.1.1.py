from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print('Cross-validation scores: {}'.format(scores))
print('Mean score: {:.3f}'.format(scores.mean()))
