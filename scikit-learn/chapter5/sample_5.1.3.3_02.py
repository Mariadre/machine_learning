from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedShuffleSplit


iris = load_iris()
logreg = LogisticRegression()

shuffle_split = ShuffleSplit(n_splits=10, test_size=.5, train_size=.5)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)

print('Cross-validation scores: {}'.format(scores))
print('Mean accuracy: {:.3f}'.format(scores.mean()))


stratified_shuffle = StratifiedShuffleSplit(n_splits=10, test_size=.5, train_size=.5)
scores = cross_val_score(logreg, iris.data, iris.target, cv=stratified_shuffle)

print('Cross-validation scores: {}'.format(scores))
print('Mean accuracy: {:.3f}'.format(scores.mean()))
