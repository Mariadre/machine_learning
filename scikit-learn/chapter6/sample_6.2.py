from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

pipe = Pipeline([("scaler", MinMaxScaler()),
                 ('svm', SVC())])

pipe.fit(X_train, y_train)

print('Test Accuracy: {:.3f}'.format(pipe.score(X_test, y_test)))
