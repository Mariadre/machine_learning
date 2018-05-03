import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN().fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan, cmap='cividis', s=60)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
