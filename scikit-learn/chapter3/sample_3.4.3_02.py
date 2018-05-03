import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


digits = load_digits()

pca = PCA(n_components=2)
digits_pca = pca.fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

for i in range(len(digits.data)):
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]))

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
