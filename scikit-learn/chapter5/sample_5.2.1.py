from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# invalid data split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print('Size of training set: {}'.format(X_train.shape[0]))
print('Size of test set: {}'.format(X_test.shape[0]))


params = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
best_parameters = {}

# handmaid grid search
for gamma in params:
    for C in params:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}


print('Best Score: {:.2f}'.format(best_score))
print('Best Parameters: {}'.format(best_parameters))
