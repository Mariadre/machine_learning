import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# 特徴量 30 のデータセット
cancer = load_breast_cancer()

# 乱数で生成した 50 の特徴量を追加する
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5)


# RandomForest を用いて有効な特徴量を選択させる
# 閾値として median を指定しているので 50 % の特徴量を選び出す
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                         threshold='median')
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
# print(X_train.shape)
# print(X_train_l1.shape)


# 選択された特徴量の可視化
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel('Sample index')
# plt.show()


# 選択された特徴量を用いることでロジスティック回帰の精度が向上する
lr = LogisticRegression().fit(X_train, y_train)
print('Test score(All): {:.3f}'.format(lr.score(X_test, y_test)))

X_test_l1 = select.transform(X_test)
lr = LogisticRegression().fit(X_train_l1, y_train)
print('Test score(Selected): {:.3f}'.format(lr.score(X_test_l1, y_test)))
