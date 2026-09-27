import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing.csv")
print("First 5 rows:")
print(df.head())
df = df[["area", "price"]]
df = df.dropna()
df["area"] = pd.to_numeric(df["area"], errors="coerce")
df = df.dropna()

x = df[["area"]]   
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("\nActual Prices:")
print(y_test.values)
print("\nPredicted Prices:")
print(y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nMean Squared Error:", mse)
print("R2 Score:", r2)
area = [[1500]]
predicted_price = model.predict(area)
print("\nPredicted price for 1500 sq.ft house:",predicted_price[0])
plt.scatter(x_test, y_test, label="Actual")
plt.plot(x_test, y_pred, label="Regression Line")
plt.xlabel("House Area (sq.ft)")
plt.ylabel("House Price")
plt.title("House Price Prediction using Area")
plt.legend()
plt.show()