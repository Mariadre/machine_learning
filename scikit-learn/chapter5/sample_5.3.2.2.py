import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)


# predict by frequency
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

print('Unique predicted labels: {}'.format(np.unique(pred_most_frequent)))
print('Test score(Most Frequent): {:.2f}'.format(dummy_majority.score(X_test, y_test)))


# predict by decision tree
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print('Test score(Decision Tree): {:.2f}'.format(tree.score(X_test, y_test)))


# predict by random
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print('Test score(Dummy): {:.2f}'.format(dummy.score(X_test, y_test)))


# predict by Logistic Regression
logreg = LogisticRegression().fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print('Test score(Logistic Regression): {:.2f}'.format(logreg.score(X_test, y_test)))




