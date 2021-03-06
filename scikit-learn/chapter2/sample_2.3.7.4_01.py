import matplotlib.pyplot as plt
import mglearn

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(['Class 0', 'Class 1', 'SV Class 0', 'SV Class 1'],
                  ncol=4, loc=(.9, 1.2))
plt.show()
