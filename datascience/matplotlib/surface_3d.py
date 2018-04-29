import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = 2
mean = np.zeros(m)
sigma = np.eye(m)

N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
Y_plot = Y_plot.reshape(X1.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Y_plot, cmap='viridis', linewidth=0, alpha=0.8)
fig.colorbar(surf)
ax.set_title("Surface Plot")
plt.show()
