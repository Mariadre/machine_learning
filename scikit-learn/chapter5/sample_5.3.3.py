import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

lr = LogisticRegression()
lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, pred)))
print('\nConfusion Matrix:\n{}'.format(confusion_matrix(y_test, pred)))

print('\nClassification Report:\n{}'.format(classification_report(y_test, pred)))

scores_image = mglearn.tools.heatmap(
    confusion_matrix(y_test, pred), xlabel='Predicted label', ylabel='Actual label',
    xticklabels=digits.target_names, yticklabels=digits.target_names, cmap='Blues', fmt='%d')
plt.title('Confusion Matrix')
plt.show()
