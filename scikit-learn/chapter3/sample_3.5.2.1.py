import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, ward


X, y = make_blobs(n_samples=12, random_state=0)

linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center')
ax.text(bounds[1], 4, ' three clusters', va='center')
ax.set_xlabel('Sample index')
ax.set_ylabel('Cluster distance')
plt.show()
