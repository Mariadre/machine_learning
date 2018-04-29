import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(max_depth=1, learning_rate=0.2)
gbrt.fit(X_train, y_train)

print('Accuracy Of Training Set: {: 6.3f}'.format(gbrt.score(X_train, y_train)))
print('Accuracy Of Test Set: {: 10.3f}'.format(gbrt.score(X_test, y_test)))


def plot_feature_importances(model, dataset):
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()


plot_feature_importances(gbrt, cancer)
