import numpy as np
import matplotlib.pyplot as plt


def plot_feature_importances(model, dataset):
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()
