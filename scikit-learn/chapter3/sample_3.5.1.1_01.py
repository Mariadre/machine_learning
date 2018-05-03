import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


X_varied, y_varied = make_blobs(n_samples=200,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
plt.legend(['cluster 0', 'cluster 1', 'cluster 2'], loc='best')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
