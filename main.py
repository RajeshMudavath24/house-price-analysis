import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("housing.csv")

print("Shape:", data.shape)

print("\nColumns:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

data = data.dropna()

print("\nSummary:")
print(data.describe())

corr = data.corr(numeric_only=True)
print("\nCorrelation with house price:")
print(corr["median_house_value"].sort_values(ascending=False))

X = data.drop("median_house_value", axis=1)
X = pd.get_dummies(X)

y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Score:", model.score(X_test, y_test))

data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.3)
plt.show()