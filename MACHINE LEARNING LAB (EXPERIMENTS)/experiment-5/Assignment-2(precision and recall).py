import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
data = {
    "Study_Hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6,  6.5, 7, 7.5, 8, 8.5],

    "Attendance": [50, 55, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 90, 92],

    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Student Dataset:")
print(df)
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_probability = model.predict_proba(X_test)[:, 1]
thresholds = [0.3, 0.5, 0.7]
for threshold in thresholds:
    y_pred = (y_probability >= threshold).astype(int)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print("\nThreshold:", threshold)
    print("Predicted Values:", y_pred)
    print("Precision:", precision)
    print("Recall:", recall)