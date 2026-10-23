import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/creditcard.csv')
scaler_amount = StandardScaler()
scaler_time = StandardScaler()
df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler_time.fit_transform(df[['Time']])
df.drop(['Amount','Time'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Prendre 50% des normales + toutes les fraudes
normal = X_train[y_train==0].sample(frac=0.5, random_state=42)
fraud = X_train[y_train==1]
X_small = pd.concat([normal, fraud])
y_small = pd.concat([y_train.loc[normal.index], y_train.loc[fraud.index]])

rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_small, y_small)

# Évaluation rapide sur le test set complet
y_proba = rf.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC sur test: {auc:.4f}")

joblib.dump({
    'model': rf,
    'model_name': 'Random Forest (50% normales)',
    'metrics': {'AUC-ROC': auc},
    'feature_names': X.columns.tolist(),
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time
}, 'app/models/saved_model.pkl')
print("✅ Modèle sauvegardé")