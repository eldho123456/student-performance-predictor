import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("student_data.csv")

X = data[['hours', 'attendance', 'sleep']]
y = data['marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Accuracy:", r2_score(y_test, y_pred))

print("\nEnter student details:")
hours = float(input("Study hours: "))
attendance = float(input("Attendance (%): "))
sleep = float(input("Sleep hours: "))

prediction = model.predict([[hours, attendance, sleep]])
print("Predicted Marks:", prediction[0])

plt.scatter(data['hours'], data['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()