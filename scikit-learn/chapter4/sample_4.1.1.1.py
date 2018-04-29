import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# データのロードと射影
data = pd.read_csv('adult.csv', header=None, index_col=False,
                   names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                          'marital-status', 'occupation', 'relationship', 'race', 'gender',
                          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]


# ワンホットエンコード
print('Original Features:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Features after get_dummies:\n', list(data_dummies.columns))


# 特徴量とラベルに分割
# ワンホットエンコードの後に行うこと！
features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values


# 訓練と結果表示
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)

print('Test Score: {:.3f}'.format(logreg.score(X_test, y_test)))
