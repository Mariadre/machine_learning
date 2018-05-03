import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)

X_train, X_test = train_test_split(X, random_state=0, test_size=.1)


def plot_scatter(ax, x, y, title):
    ax.scatter(x[:, 0], x[:, 1], c='red', label='Training set', s=60, alpha=.5)
    ax.scatter(y[:, 0], y[:, 1], c='blue', marker='^', label='Test set', s=60, alpha=.5)
    ax.legend(loc='best')
    ax.set_title(title)


fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Original Data
plot_scatter(axes[0], X_train, X_test, 'Original Data')

# Properly Scaled Data
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

plot_scatter(axes[1], X_train_scaled, X_test_scaled, 'Scaled Data')

# Improperly Scaled Data
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

plot_scatter(axes[2], X_train_scaled, X_test_scaled, 'Badly Scaled Data')


for ax in axes:
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')

plt.show()
