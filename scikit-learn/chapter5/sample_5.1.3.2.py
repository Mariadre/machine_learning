from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut


iris = load_iris()
logreg = LogisticRegression()


loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)

print('#Cross-validation iterations: {}'.format(len(scores)))
print('Mean score: {:.3f}'.format(scores.mean()))
