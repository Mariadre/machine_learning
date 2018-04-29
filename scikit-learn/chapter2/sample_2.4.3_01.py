import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=12)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)

print('Decision Function shape: {}'.format(gbrt.decision_function(X_test).shape))
print('Decision Function:\n{}'.format(gbrt.decision_function(X_test)[:6]))

am = np.argmax(gbrt.decision_function(X_test), axis=1)
p = gbrt.predict(X_test)
print('Argmax of decision function:\n{}'.format(am))
print('Predictions:\n{}'.format(p))
print('prediction is equal to argmax of decision function: {}'.format(np.all(am == p)))

print()
print('Predicted Probabilities:\n{}'.format(gbrt.predict_proba(X_test)[:6]))
print('Sums: {}'.format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))
print('Argmax of predicted probabilities:\n{}'.format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
print('Predictions:\n{}'.format(gbrt.predict(X_test)))


lr = LogisticRegression()
named_target = iris.target_names[y_train]
lr.fit(X_train, named_target)
print('unique classes in training data: {}'.format(lr.classes_))
print('predictions: {}'.format(lr.predict(X_train)[:15]))
print('argmax of decision function: {}'.format(np.argmax(lr.decision_function(X_test)[:15], axis=1)))
am = np.argmax(lr.predict_proba(X_test), axis=1)
print('argmax of predicted propabilities: {}'.format(lr.classes_[am][:10]))
