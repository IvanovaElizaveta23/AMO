import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

train_data = pd.read_csv('train/train_dfsc.csv')

X_train = train_data.drop(["cnt"], axis=1)
y_train = train_data["cnt"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

dump(model, 'rmr_model.pkl')
loaded_model = load('rmr_model.pkl')