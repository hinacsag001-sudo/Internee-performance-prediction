import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("intern_performance.csv")

print("\nDataset Preview\n")
print(df.head())

# -------------------------
# Features & Target
# -------------------------

X = df[[
    "Task_Completion_Time_Hours",
    "Feedback_Rating",
    "Attendance_Percentage"
]]

y = df["Performance_Score"]

# -------------------------
# Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------
# Train Model
# -------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------

predictions = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 40)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("R2  :", round(r2, 3))

# -------------------------
# Save Model
# -------------------------

joblib.dump(model, "random_forest_model.pkl")

print("\nModel saved successfully!")