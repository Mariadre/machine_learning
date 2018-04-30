from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# valid data split
iris = load_iris()
X_dev, X_test, y_dev, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_cv, y_train, y_cv = train_test_split(X_dev, y_dev, random_state=1)

print('Size of training set: {}'.format(X_train.shape[0]))
print('Size of cv set: {}'.format(X_cv.shape[0]))
print('Size of test set: {}'.format(X_test.shape[0]))


params = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
best_parameters = {}

for gamma in params:
    for C in params:
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_train, y_train)
        score = svm.score(X_cv, y_cv)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}


svm = SVC(**best_parameters)
svm.fit(X_dev, y_dev)

print('Best score on validation set: {:.2f}'.format(best_score))
print('Best parameters:', best_parameters)
print('Test set score with best parameters: {:.2f}'.format(svm.score(X_test, y_test)))
