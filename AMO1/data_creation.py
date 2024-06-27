from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd
import os

bike_sharing = fetch_ucirepo(id=275)

X = bike_sharing.data.features
y = bike_sharing.data.targets

print(bike_sharing.metadata)
print(bike_sharing.variables)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

train_df = pd.concat([pd.DataFrame(X_train),pd.DataFrame(y_train)], axis=1)
test_df = pd.concat([pd.DataFrame(X_test),pd.DataFrame(y_test)], axis=1)

if not os.path.exists("train"):
    os.makedirs("train")
if not os.path.exists("test"):
    os.makedirs("test")

train_df.to_csv('train/train_df.csv', index=False)
test_df.to_csv('test/test_df.csv', index=False)