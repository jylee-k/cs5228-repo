import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('Coursework2/cs5228-Housing.csv')

# --- Preprocessing ---
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 2})

# furnishingstatus: avoid zero coding
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 1, 'semi-furnished': 2, 'furnished': 3})

# --- CW2-1: Regression ---
X_reg = df.drop(['price', 'class_label'], axis=1)
y_reg = df['price']

# Correlation Analysis
corr = X_reg.corrwith(y_reg)
print("Correlation with price:\n", corr)

# Split 90/10
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.1, random_state=110)

# Normalization (StandardScaler)
scaler = StandardScaler()
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Linear Regression (without normalization)
model_reg = LinearRegression()
model_reg.fit(X_train_reg, y_train_reg)
y_pred_train_reg = model_reg.predict(X_train_reg)
y_pred_test_reg = model_reg.predict(X_test_reg)

mse_train = mean_squared_error(y_train_reg, y_pred_train_reg)
mae_train = mean_absolute_error(y_train_reg, y_pred_train_reg)
mse_test = mean_squared_error(y_test_reg, y_pred_test_reg)
mae_test = mean_absolute_error(y_test_reg, y_pred_test_reg)

print("\n--- CW2-1 Regression (No Normalization) ---")
print(f"Train MSE: {mse_train:.2f}, MAE: {mae_train:.2f}")
print(f"Test MSE: {mse_test:.2f}, MAE: {mae_test:.2f}")

# Linear Regression (with normalization)
model_reg_scaled = LinearRegression()
model_reg_scaled.fit(X_train_reg_scaled, y_train_reg)
y_pred_train_reg_scaled = model_reg_scaled.predict(X_train_reg_scaled)
y_pred_test_reg_scaled = model_reg_scaled.predict(X_test_reg_scaled)

mse_train_scaled = mean_squared_error(y_train_reg, y_pred_train_reg_scaled)
mae_train_scaled = mean_absolute_error(y_train_reg, y_pred_train_reg_scaled)
mse_test_scaled = mean_squared_error(y_test_reg, y_pred_test_reg_scaled)
mae_test_scaled = mean_absolute_error(y_test_reg, y_pred_test_reg_scaled)

print("\n--- CW2-1 Regression (With Normalization) ---")
print(f"Train MSE: {mse_train_scaled:.2f}, MAE: {mae_train_scaled:.2f}")
print(f"Test MSE: {mse_test_scaled:.2f}, MAE: {mae_test_scaled:.2f}")

# --- CW2-2: Classification ---
X_clf = df.drop(['price', 'class_label'], axis=1)
y_clf = df['class_label']

# Split 90/10
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.1, random_state=110)

# Decision Tree
dt = DecisionTreeClassifier(random_state=110)
dt.fit(X_train_clf, y_train_clf)
y_pred_train_dt = dt.predict(X_train_clf)
y_pred_test_dt = dt.predict(X_test_clf)

print("\n--- CW2-2 Decision Tree ---")
print(f"Train Accuracy: {accuracy_score(y_train_clf, y_pred_train_dt):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_test_dt):.4f}")
print(f"Test Precision (Macro): {precision_score(y_test_clf, y_pred_test_dt, average='macro'):.4f}")
print(f"Test Recall (Macro): {recall_score(y_test_clf, y_pred_test_dt, average='macro'):.4f}")

# Random Forest
rf = RandomForestClassifier(random_state=110)
rf.fit(X_train_clf, y_train_clf)
y_pred_train_rf = rf.predict(X_train_clf)
y_pred_test_rf = rf.predict(X_test_clf)

print("\n--- CW2-2 Random Forest ---")
print(f"Train Accuracy: {accuracy_score(y_train_clf, y_pred_train_rf):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_test_rf):.4f}")
print(f"Test Precision (Macro): {precision_score(y_test_clf, y_pred_test_rf, average='macro'):.4f}")
print(f"Test Recall (Macro): {recall_score(y_test_clf, y_pred_test_rf, average='macro'):.4f}")
print("Confusion Matrix (RF):\n", confusion_matrix(y_test_clf, y_pred_test_rf))
