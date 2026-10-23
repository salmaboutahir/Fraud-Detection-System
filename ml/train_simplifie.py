import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import joblib
import os

os.makedirs('app/models', exist_ok=True)

# 1. Charger les données
df = pd.read_csv('data/creditcard.csv')
print(f"Shape: {df.shape}, Fraudes: {df['Class'].sum()}")

# 2. Normaliser Amount et Time avec DEUX scalers distincts
scaler_amount = StandardScaler()
scaler_time = StandardScaler()
df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler_time.fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# 3. Séparer features et cible
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Entraîner Random Forest avec class_weight='balanced'
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 6. Évaluation rapide
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
print(f"AUC-ROC: {auc:.4f}")
print(f"F1-Score: {f1:.4f}")

# 7. Sauvegarder modèle et scalers
feature_names = X.columns.tolist()
joblib.dump({
    'model': rf,
    'model_name': 'Random Forest (simple)',
    'metrics': {'AUC-ROC': auc, 'F1': f1},
    'feature_names': feature_names,
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time
}, 'app/models/saved_model.pkl')
print("✅ Modèle sauvegardé dans app/models/saved_model.pkl")