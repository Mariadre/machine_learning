import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


digits = load_digits()

tsne = TSNE(n_components=2, random_state=42)
digits_tsne = tsne.fit_transform(digits.data)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.set_xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
ax.set_ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]))

ax.set_xlabel('t-SNE feature 0')
ax.set_ylabel('t-SNE feature 1')
plt.show()
