import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


X, y = make_blobs(random_state=1)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

cluster_centers = kmeans.cluster_centers_
labels = np.unique(kmeans.labels_)

print('Cluster memberships:\n{}'.format(labels))
print('Cluster centers:\n{}'.format(cluster_centers))

mglearn.discrete_scatter(X[:, 0], X[:, 1], y, markers=['o'], alpha=.7)
mglearn.discrete_scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                         labels, markers=['^'], markeredgewidth=2)
plt.show()
