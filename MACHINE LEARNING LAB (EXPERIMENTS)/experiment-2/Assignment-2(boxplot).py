import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
wine = load_wine()
df= pd.DataFrame(wine.data,columns=wine.feature_names)
numerical_features = wine.feature_names
plt.figure(figsize=(15, 10))
df[numerical_features].boxplot()
plt.title("Boxplots of Wine Dataset Numerical Features")
plt.xticks(rotation=90)
plt.ylabel("Values")
plt.tight_layout()
plt.savefig("a2")
plt.show()