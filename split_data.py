import pandas as pd
import numpy as np

# To do: implement this script with sklearn functions
df = pd.read_csv('data/train.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
test = df[~msk]

del train['split']
del test['split']

print(train.head())
print("--------------------------------")
print(f"Original file size {df.shape}")
print(f"Train file size {train.shape}")
print(f"Test file size {test.shape}")

train.to_csv('data/train_split.csv', index=False)
test.to_csv('data/test_split.csv', index=False)