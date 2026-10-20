import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== 1. MEMUAT & MENYIAPKAN DATASET PT MAJU BERSAMA ===")
np.random.seed(42)
n_samples = 1500

data = {
    'Customer_ID': np.random.randint(101, 221, size=n_samples),
    'Product_ID': np.random.randint(1, 25, size=n_samples),
    'Order_Amount': np.random.uniform(500000, 15000000, size=n_samples),
    'Discount_Given': np.random.choice([0, 0.05, 0.1, 0.2, 0.25], size=n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    'Payment_Method': np.random.choice(['Transfer', 'Credit', 'Cash'], size=n_samples, p=[0.4, 0.4, 0.2]),
    'Customer_History_Late_Count': np.random.randint(0, 6, size=n_samples)
}

df = pd.DataFrame(data)

conditions = (df['Customer_History_Late_Count'] > 2) | ((df['Discount_Given'] > 0.1) & (df['Payment_Method'] == 'Credit'))
df['Late_Payment'] = np.where(conditions & (np.random.rand(n_samples) > 0.25), 1, 0)

print("5 Baris Pertama Data:")
print(df.head(), "\n")

print("=== 2. DATA PREPARATION ===")
X = df.drop(columns=['Late_Payment', 'Customer_ID'])
y = df['Late_Payment']

X = pd.get_dummies(X, columns=['Payment_Method'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Jumlah Data Training: {X_train.shape[0]} baris")
print(f"Jumlah Data Testing: {X_test.shape[0]} baris\n")

print("=== 3. PEMODELAN (MODELING) ===")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

print("=== 4. EVALUASI MODEL ===")

print("\n--- Performa Decision Tree ---")
y_pred_dt = dt_model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

print("\n--- Performa Random Forest ---")
y_pred_rf = rf_model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))