import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()
df= pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]=wine.target
print("wine Dataset:")
print(df)
print("--- First Five Rows---")
print(df.head())
print("\n--- Last Five Rows ---")
print(df.tail())
print("\n--- Shape of Dataset ---")
print(df.shape)
print("\n--- Dataset Information---")
print(df.info())
print("\n--- Statistical Summary---")
print(df.describe())
print("\n--- Column Names ---")
print(df.columns)
print("\n--- Missing Values---")
print(df.isnull().sum())
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())
print("\n--- Target Value Counts ---")
print(df["target"].value_counts())


