from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X_train, y_train)

svc_origin = SVC().fit(X_train, y_train)

print('Test Score for Original Data: {:.3f}'.format(svc_origin.score(X_test, y_test)))
print('Test Score for Scaled Data: {:.3f}'.format(pipe.score(X_test, y_test)))
