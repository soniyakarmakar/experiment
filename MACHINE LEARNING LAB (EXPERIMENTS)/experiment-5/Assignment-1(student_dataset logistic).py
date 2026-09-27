import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = {
    "Study_Hours": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5],
    "Attendance": [50, 55, 60, 62, 65, 68, 70, 72, 75, 78, 80, 82, 85, 90, 92],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Student Dataset:")
print(df)
X = df[["Study_Hours", "Attendance"]]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nActual Values:")
print(y_test.values)
print("\nPredicted Values:")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[5, 75]]
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("\nNew Student: PASS")
else:
    print("\nNew Student: FAIL")