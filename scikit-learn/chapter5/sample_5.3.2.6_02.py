import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)


plt.figure()

for ls, gamma in zip(['-', '-.', '--'], [1, 0.1, 0.01]):
    svm = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    auc = roc_auc_score(y_test, svm.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svm.decision_function(X_test))
    print('gamma: {:.2f}, accuracy: {:.3f}, AUC: {:.3f}'.format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label='gamma={:.2f}'.format(gamma), linestyle=ls)

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)
plt.legend(loc='best')
plt.show()
