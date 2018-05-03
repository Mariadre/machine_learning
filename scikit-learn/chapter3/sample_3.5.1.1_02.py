import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


X, y = make_blobs(n_samples=600, random_state=170)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)


_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           c=np.unique(kmeans.labels_),
           cmap='rainbow', marker='^', s=100, linewidth=2)
ax.set_xlabel('Feature 0')
ax.set_ylabel('Feature 1')
plt.show()
