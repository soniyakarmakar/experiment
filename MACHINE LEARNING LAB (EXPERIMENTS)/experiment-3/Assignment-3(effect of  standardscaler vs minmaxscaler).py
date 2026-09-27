import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = {
    "Age": [22, 25, 28, 30, 35],
    "Salary": [25000, 30000, 35000, 40000, 50000],
    "Years Experience": [1, 2, 3, 5, 7]
}
df = pd.DataFrame(data)
standard_scaler = StandardScaler()
standard_data = standard_scaler.fit_transform(df)
minmax_scaler = MinMaxScaler()
minmax_data = minmax_scaler.fit_transform(df)
print("Original Data:")
print(df)
print("\nStandardScaler:")
print(standard_data)
print("\nMinMaxScaler:")
print(minmax_data)
print("\nStandardScaler Range:")
print("Minimum:", standard_data.min())
print("Maximum:", standard_data.max())
print("\nMinMaxScaler Range:")
print("Minimum:", minmax_data.min())
print("Maximum:", minmax_data.max())