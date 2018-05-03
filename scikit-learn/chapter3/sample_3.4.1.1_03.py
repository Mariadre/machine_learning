import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cancer = load_breast_cancer()

pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])
pipe.fit(cancer.data)


plt.matshow(pipe.named_steps['pca'].components_, cmap='bone')
plt.yticks([0, 1], ['First Component', 'Second Component'])
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel('Feature')
plt.ylabel('Principal Components')
plt.colorbar()
plt.show()
