import pandas as pd
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('train/train_df.csv')
test_df = pd.read_csv('test/test_df.csv')

X_train = train_df.drop('cnt', axis=1)
y_train = train_df['cnt']

X_test = test_df.drop('cnt', axis=1)
y_test = test_df['cnt']

X_train_no_date = X_train.drop('dteday', axis=1)
X_test_no_date = X_test.drop('dteday', axis=1)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_no_date)
X_test_sc = scaler.transform(X_test_no_date)

train_dfsc = pd.DataFrame(X_train_sc, columns=X_train_no_date.columns)
train_dfsc[y_train.name] = y_train
test_dfsc = pd.DataFrame(X_test_sc, columns=X_test_no_date.columns)
test_dfsc[y_test.name] = y_test

train_dfsc.to_csv('train/train_dfsc.csv', index=False)
test_dfsc.to_csv('test/test_dfsc.csv', index=False)