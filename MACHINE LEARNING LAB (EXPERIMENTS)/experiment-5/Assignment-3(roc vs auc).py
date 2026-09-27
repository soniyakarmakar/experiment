import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
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
fpr, tpr, thresholds = roc_curve(y_test, y_probability)
auc_score = roc_auc_score(y_test, y_probability)
print("\nAUC Score:", auc_score)
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Student Pass/Fail")
plt.legend()
plt.savefig("ROC_Curve_Student_Pass_Fail.png")
plt.show()