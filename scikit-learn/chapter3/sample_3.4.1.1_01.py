import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

fig, axes = plt.subplots(10, 3, figsize=(15, 25))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color='orchid', alpha=0.4)
    ax[i].hist(benign[:, i], bins=bins, color='gray', alpha=0.4)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks([])

ax[0].set_xlabel('Feature malignant')
ax[0].set_ylabel('Frequency')
ax[0].legend(['malignant', 'benign'], loc='best')
fig.tight_layout()
plt.show()
