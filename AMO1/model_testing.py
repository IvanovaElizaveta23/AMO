import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump, load


test_df = pd.read_csv('test/test_dfsc.csv')

model = load('rmr_model.pkl')

X_test = test_df.drop(["cnt"], axis=1)
y_test = test_df["cnt"]

pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
print(f"R-squared: {r2:.2f}")
mse = mean_absolute_error(y_test, pred)
print(f"Средняя абсолютная ошибка: {mse:.2f}")