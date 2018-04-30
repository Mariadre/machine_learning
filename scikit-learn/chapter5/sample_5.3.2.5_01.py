import numpy as np
import matplotlib.pyplot as plt
from mglearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score


X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# SVC
svc = SVC(gamma=0.05)
svc.fit(X_train, y_train)

precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))

# RF
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:, 1])


# default threshold
close_zero = np.argmin(np.abs(thresholds))
close_zero_rf = np.argmin(np.abs(thresholds_rf - 0.5))

# draw graph
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
        label='threshold zero', fillstyle='none', c='k', mew=2)
ax.plot(precision_rf[close_zero_rf], recall_rf[close_zero_rf], '^', markersize=10,
        label='threshold 0.5', fillstyle='none', mew=2)
ax.plot(precision, recall, label='SVC')
ax.plot(precision_rf, recall_rf, label='Randam Forest')
ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.legend(loc='best')


# f1 score
print('f1 score')
print('  SVC: {:.3f}'.format(f1_score(y_test, svc.predict(X_test))))
print('  RF : {:.3f}'.format(f1_score(y_test, rf.predict(X_test))))


# average precision
print('\nAverage precision score')
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
print('  SVC: {:.3f}'.format(ap_svc))
print('  RF : {:.3f}'.format(ap_rf))

plt.show()
