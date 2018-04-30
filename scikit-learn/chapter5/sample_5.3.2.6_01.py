import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# SVC
svc = SVC(gamma=0.05)
svc.fit(X_train, y_train)

# RF
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)


fpr, tpr, threshold = roc_curve(y_test, svc.decision_function(X_test))
fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])


plt.plot(fpr, tpr, label='SVC')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('FPR')
plt.ylabel('TPR')

close_zero = np.argmin(np.abs(threshold))
close_zero_rf = np.argmin(np.abs(threshold_rf - 0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, c='k', fillstyle='none', label='threshold 0')
plt.plot(fpr_rf[close_zero_rf], tpr_rf[close_zero_rf], '^', markersize=10, fillstyle='none', label='threshold 0.5')
plt.legend(loc='best')
plt.show()


svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
print('AUC(SVC): {:.3f}'.format(svc_auc))
print('AUC(RF) : {:.3f}'.format(rf_auc))
