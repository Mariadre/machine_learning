import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


cancer = load_breast_cancer()

pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
pipe.fit(cancer.data, cancer.target)

reduced_data = pipe.transform(cancer.data)


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
mglearn.discrete_scatter(reduced_data[:, 0], reduced_data[:, 1], cancer.target)
plt.legend(loc='best')
ax.set_aspect('equal')
ax.set_xlabel('1st Principle Component')
ax.set_ylabel('2nd Principle Component')
plt.show()
