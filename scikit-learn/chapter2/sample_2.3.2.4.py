import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave


X, y = make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.scatter(X_train, y_train, marker='^', c='pink', s=18, alpha=.4, edgecolor='red')
    ax.scatter(X_test, y_test, marker='o', c='lightblue', s=18, alpha=.4, edgecolor='blue')

    ax.set_title('{} neighbor(s)\ntrain score: {:.2f} / test score: {:.2f}'.format(
        n_neighbors,
        reg.score(X_train, y_train),
        reg.score(X_test, y_test)
    ))

    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')


axes[0].legend(['Model Predictions', 'Training Data/Target', 'Test Data/Target'], loc='best')
plt.show()
