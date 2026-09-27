import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
df = pd.DataFrame({
    "Age": [22, 25, 28, 30, 35, 40, 45, 29, 32, 38],
    "Salary": [25000, 30000, 35000, 40000, 50000, 60000, 75000, 42000, 48000, 55000],
    "Department": ["IT", "HR", "IT", "Finance", "HR",
                    "IT", "Finance", "IT", "HR", "Finance"],
    "Years Experience": [1, 2, 3, 5, 7, 10, 15, 4, 6, 9]
})
X = df[["Age", "Salary", "Department", "Years Experience"]]
numeric_features = ["Age", "Salary", "Years Experience"]
categorical_features = ["Department"]
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", MinMaxScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
pipeline = Pipeline([
    ("preprocessor", preprocessor)
])
X_transformed = pipeline.fit_transform(X)
print("Transformed Data:")
print(X_transformed)