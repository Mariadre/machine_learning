import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mglearn
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)


param_C = [0.001, 0.01, 0.1, 1, 10, 100]
param_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
grid_param = {'C': param_C,
              'gamma': param_gamma}

grid_search = GridSearchCV(SVC(), grid_param, cv=5)
cross_val_score()
grid_search.fit(X_train, y_train)


results = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results.mean_test_score).reshape(len(param_gamma), len(param_C))

# mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=grid_param['gamma'],
#                       ylabel='gamma', yticklabels=grid_param['C'], cmap='viridis')
sns.heatmap(scores, xticklabels=grid_param['gamma'], yticklabels=grid_param['C'])
plt.xlabel('gamma')
plt.ylabel('C')
plt.show()
