 # train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 Load Dataset
data = pd.read_csv("data/fuel_efficiency_dataset.csv")

# 🧹 Data Cleaning
data = data.dropna()

# 🔤 Encode Categorical Columns
categorical_cols = ['fuel_type', 'transmission']
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# 🎯 Features and Target
X = data.drop(columns=['mpg'])
y = data['mpg']

# 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚖️ Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🧠 Model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# 🔍 Feature Importance
features = X.columns if isinstance(X, pd.DataFrame) else np.arange(len(X[0]))
importances = rf_model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 💾 Save Model and Scaler
joblib.dump(rf_model, "models/fuel_efficiency_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Model and Scaler Saved Successfully!")
