import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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
# Feature Engineering
# -------------------------

X = df[[
    "Task_Completion_Time_Hours",
    "Feedback_Rating",
    "Attendance_Percentage"
]].copy()

# Add interaction features to improve model
X['Feedback_x_Attendance'] = X['Feedback_Rating'] * X['Attendance_Percentage']
X['Task_Time_x_Feedback'] = X['Task_Completion_Time_Hours'] * X['Feedback_Rating']
X['Task_Time_Squared'] = X['Task_Completion_Time_Hours'] ** 2
X['Feedback_Squared'] = X['Feedback_Rating'] ** 2

y = df["Performance_Score"]

# -------------------------
# Feature Scaling
# -------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nFeatures shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

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
# Train Model (Using Gradient Boosting for better predictions)
# -------------------------

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Predictions
# -------------------------

y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------

train_mae = mean_absolute_error(y_train, y_train_predictions)
train_mse = mean_squared_error(y_train, y_train_predictions)
train_r2 = r2_score(y_train, y_train_predictions)

mae = mean_absolute_error(y_test, y_test_predictions)
mse = mean_squared_error(y_test, y_test_predictions)
r2 = r2_score(y_test, y_test_predictions)

print("\nModel Performance (Training Set)")
print("-" * 40)
print("MAE :", round(train_mae, 2))
print("MSE :", round(train_mse, 2))
print("R2  :", round(train_r2, 3))

print("\nModel Performance (Test Set)")
print("-" * 40)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("R2  :", round(r2, 3))

# Feature importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# -------------------------
# Save Model and Scaler
# -------------------------

joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!")