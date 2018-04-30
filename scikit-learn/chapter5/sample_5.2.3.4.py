from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV


iris = load_iris()

param_C = [0.001, 0.01, 0.1, 1, 10, 100]
param_gamma = [0.001, 0.01, 0.1, 1, 10, 100]
param_grid = {'C': param_C,
              'gamma': param_gamma}

# nested cross-validation
scores = cross_val_score(GridSearchCV(SVC(), param_grid=param_grid, cv=5),
                         iris.data, iris.target, cv=5)

# あるデータセットに対する、あるアルゴリズムの評価精度のポテンシャルがうかがえる
print('Mean accuracy: {:.3f}'.format(scores.mean()))
