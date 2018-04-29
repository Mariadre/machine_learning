import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

mlp = MLPClassifier(random_state=42, max_iter=1000)
mlp.fit(X_train, y_train)

# スケールフィッティング前
print('NOT scaled')
print('Accuracy of Training set: {:.3f}'.format(mlp.score(X_train, y_train)))
print('Accuracy of Test set: {:.3f}'.format(mlp.score(X_test, y_test)))


mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=42, max_iter=1000, alpha=1.01)
mlp.fit(X_train_scaled, y_train)
print('\nscaled')
print('Accuracy of Training set: {:.3f}'.format(mlp.score(X_train_scaled, y_train)))
print('Accuracy of Test set: {:.3f}'.format(mlp.score(X_test_scaled, y_test)))


plt.figure(figsize=(20, 10))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel('Columns in weight matrix')
plt.ylabel('Input Feature')
plt.colorbar()
plt.show()