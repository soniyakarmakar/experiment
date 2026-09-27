import numpy as np
import  pandas as pd
data = {
    "Std_Name":["Amit","Riya","Sourav","Neha","Rahul"],
    "Roll_No":[42,52,55,60,24],
    "Marks":[72,99,87,77,82],
    "Attendance":[88,95,75,80,60]

}
df= pd.DataFrame(data)
def Grade(Marks):
    if Marks>=90:
        return("A")
    elif Marks>=80:
        return("B")
    elif Marks>=70:
        return("C")
    elif Marks>=60:
        return("D")
    else:
        return("F")

df["Grade"]=df["Marks"].apply(Grade)
print(df)