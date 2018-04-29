import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)


for c, m in zip([1, 100, 0.001], ['o', '^', 'v']):
    logreg = LogisticRegression(C=c).fit(X_train, y_train)
    print('\nC={}'.format(c))
    print('Training Set Score: {:.2f}'.format(logreg.score(X_train, y_train)))
    print('Test Set Score: {:.2f}'.format(logreg.score(X_test, y_test)))
    print('#Features used: {}'.format(np.sum(logreg.coef_ != 0)))
    plt.plot(logreg.coef_.T, m, label='C={}'.format(c))


plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel('Feature')
plt.ylabel('Coefficient maginitude')
plt.legend(loc='best')
plt.show()
