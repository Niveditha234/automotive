#!/usr/bin/env python
# coding: utf-8

# In[6]:


# ======================================================
# 🚗 Fuel Efficiency Prediction Model Using ML
# One-click script: Train + Save + Load + Predict + Evaluate
# ======================================================

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------
print("📂 Loading dataset...")
data_path = r"C:\Users\Admin\Downloads\fuel_efficiency_dataset.csv"  # <-- change if needed
data = pd.read_csv(data_path).dropna()
print(f"✅ Dataset loaded successfully with {len(data)} rows and {len(data.columns)} columns.\n")

# ------------------------------------------------------
# 2️⃣ Encode Categorical Columns
# ------------------------------------------------------
print("🔤 Encoding categorical features...")
for col in ['fuel_type', 'transmission']:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# ------------------------------------------------------
# 3️⃣ Split Data
# ------------------------------------------------------
X = data.drop(columns=['mpg'])
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------
# 4️⃣ Scale Features
# ------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------
# 5️⃣ Train Model
# ------------------------------------------------------
print("🧠 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model training complete.\n")

# ------------------------------------------------------
# 6️⃣ Evaluate Model
# ------------------------------------------------------
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}\n")

# ------------------------------------------------------
# 7️⃣ Save Model & Scaler
# ------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fuel_efficiency_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("💾 Model and scaler saved in 'models/' folder.\n")

# ------------------------------------------------------
# 8️⃣ Load Model & Predict Again (Verification)
# ------------------------------------------------------
print("🔁 Loading saved model to verify...")
loaded_model = joblib.load("models/fuel_efficiency_model.pkl")
loaded_scaler = joblib.load("models/scaler.pkl")

sample_input = np.array([[1500, 100, 1100, 4, 9.5, 2018, 0, 1]])  # example test input
sample_scaled = loaded_scaler.transform(sample_input)
sample_pred = loaded_model.predict(sample_scaled)[0]
print(f"🚙 Predicted Fuel Efficiency for sample vehicle: {sample_pred:.2f} km/l or MPG\n")

# ------------------------------------------------------
# 9️⃣ Visualization
# ------------------------------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted Fuel Efficiency")
plt.grid(True)
plt.show()

# ------------------------------------------------------
# 🔟 Feature Importance Visualization
# ------------------------------------------------------
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(True)
plt.show()

print("✅ End of program — model trained, evaluated, saved, and verified successfully.")


# In[ ]:




