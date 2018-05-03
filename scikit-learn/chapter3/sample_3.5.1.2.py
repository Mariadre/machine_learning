import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
distance_features = kmeans.transform(X)
y_pred = kmeans.predict(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     distance_features, y_pred, random_state=0)
#
#
# param_grid = {'svc__C': [0.001, 0.01, 0.1, 1, 10, 100],
#               'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
# pipe = make_pipeline(StandardScaler(), SVC())
# grid = GridSearchCV(pipe, param_grid=param_grid, cv=15)
# grid.fit(X_train, y_train)
#
# print('Best Parameters:', grid.best_params_)
# print('Best cross-validation Score: {:.3f}'.format(grid.best_score_))
# print('Train set Score: {:.3f}'.format(grid.score(X_train, y_train)))
# print('Test set Score: {:.3f}'.format(grid.score(X_test, y_test)))

_, ax = plt.subplots(1, 1, figsize=(6, 6))

ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Blues', alpha=0.5)
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           c=range(kmeans.n_clusters),
           s=60, marker='^', linewidth=2, cmap='bone')
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
plt.show()
