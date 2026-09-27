import numpy as np
import pandas as pd
data = {
    "Std_Name":["Amit","Riya","Sourav","Neha","Rahul"],
    "Roll_No":[42,52,55,60,24],
    "Marks":[72,99,87,77,82],
    "Attendance":[88,95,75,80,60]

}
df= pd.DataFrame(data)
print("\n --first five records--")
print(df.head())
print(df.info())
print(df.describe())
print(df[df["Marks"]>80]["Std_Name"])