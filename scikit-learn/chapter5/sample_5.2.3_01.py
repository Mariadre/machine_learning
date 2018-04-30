from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split


iris = load_iris()
X_dev, X_test, y_dev, y_test = train_test_split(iris.data, iris.target, random_state=0)


params = [0.001, 0.01, 0.1, 1, 10, 100]
best_score = 0
best_params = {}


# グリッドサーチはデータ分割のされ方に影響される
# →データ分割をならすためには交差検証が有効
for gamma in params:
    for C in params:
        scores = cross_val_score(SVC(C=C, gamma=gamma), X_dev, y_dev, cv=5)

        score = scores.mean()
        if score > best_score:
            best_score = score
            best_params = {'C': C, 'gamma': gamma}


svm = SVC(**best_params)
svm.fit(X_dev, y_dev)

print('Best parameters:', best_params)
print('Test accuracy: {:.3f}'.format(svm.score(X_test, y_test)))
