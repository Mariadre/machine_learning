import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print('Accuracy Of Training Set: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy Of Testing Set: {:.3f}'.format(tree.score(X_test, y_test)))


# export tree graph
# export_graphviz(tree, out_file='tree.dot', class_names=['malignant', 'benign'],
#                 feature_names=cancer.feature_names, filled=True, impurity=False)


def plot_feature_importances_cancer(model, dataset):
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()


plot_feature_importances_cancer(tree, cancer)
