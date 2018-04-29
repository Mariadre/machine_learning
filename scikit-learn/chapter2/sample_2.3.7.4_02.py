import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svc = SVC().fit(X_train, y_train)

# scale fitting していないため精度が低い
print('Accuracy of Training Set: {: 4.3f}'.format(svc.score(X_train, y_train)))
print('Accuracy of Test Set: {: 10.3f}'.format(svc.score(X_test, y_test)))


plt.plot(X_train.min(axis=0), 'o', label='min')
plt.plot(X_train.max(axis=0), '^', label='max')
plt.legend(loc=4)
plt.xlabel('Feature index')
plt.ylabel('Feature magnitude')
plt.yscale('log')
plt.show()
