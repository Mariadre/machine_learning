import matplotlib.pyplot as plt
import mglearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons


X, y = make_blobs(random_state=1)
# X, y = make_moons(n_samples=400, noise=0.05, random_state=0)

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
assignment = agg.fit_predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=assignment, cmap='Paired')
mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
