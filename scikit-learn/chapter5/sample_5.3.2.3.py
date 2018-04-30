import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, classification_report


digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)


# predict by frequency
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)

# predict by decision tree
tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)

# predict by random
dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

# predict by Logistic Regression
logreg = LogisticRegression().fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)


print('Confusion Matrix')
print('Logistic Regression:\n{}'.format(confusion_matrix(y_test, pred_logreg)))
print('Most Frequent:\n{}'.format(confusion_matrix(y_test, pred_most_frequent)))
print('Dummy:\n{}'.format(confusion_matrix(y_test, pred_dummy)))
print('Decision Tree:\n{}'.format(confusion_matrix(y_test, pred_tree)))

print('f1 Score')
print('Logistic Regression:\n{}'.format(f1_score(y_test, pred_logreg)))
print('Most Frequent:\n{}'.format(f1_score(y_test, pred_most_frequent)))
print('Dummy:\n{}'.format(f1_score(y_test, pred_dummy)))
print('Decision Tree:\n{}'.format(f1_score(y_test, pred_tree)))


print(classification_report(y_test, pred_logreg, target_names=['not nine', 'nine']))
print(classification_report(y_test, pred_most_frequent, target_names=['not nine', 'nine']))
print(classification_report(y_test, pred_dummy, target_names=['not nine', 'nine']))
print(classification_report(y_test, pred_tree, target_names=['not nine', 'nine']))
