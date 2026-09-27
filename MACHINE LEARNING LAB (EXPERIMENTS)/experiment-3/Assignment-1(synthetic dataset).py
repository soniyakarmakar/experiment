import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data={
       "Age":[19,20,21,27,None],
       "Salary":[25000,50000,30000,None,40000],
       "Department":["IT","HR","IT",None,"Finance"],
       "Years of Experience":[1,2,5,6,None]
}

df=pd.DataFrame(data)

print("Original Dataset:")
print(df)

imputer=SimpleImputer(strategy="mean")
df[["Age","Salary","Years of Experience"]]=imputer.fit_transform(df[["Age","Salary","Years of Experience"]])

df["Department"]=df["Department"].fillna("Unknown")
print("\nPreprocessed Dataset:")
print(df)