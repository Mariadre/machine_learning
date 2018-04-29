import pandas as pd


df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                   'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
print('Original:\n', df)


dummies = pd.get_dummies(df)
print('Dummies:\n', dummies)


df['Integer Feature'] = df['Integer Feature'].astype(str)  # 非可逆
dummies = pd.get_dummies(df)
print('Dummies:\n', dummies)
