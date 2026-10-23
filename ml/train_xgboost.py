import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, classification_report
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

# SMOTE sur l'entraînement seulement
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Après SMOTE : {X_train_res.shape[0]} échantillons, fraudes: {y_train_res.sum()}")

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=1,  # maintenant équilibré grâce à SMOTE
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

y_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")
print(classification_report(y_test, model.predict(X_test), target_names=['Normal','Fraude']))

joblib.dump({
    'model': model,
    'model_name': 'XGBoost + SMOTE',
    'metrics': {'AUC-ROC': auc},
    'feature_names': X.columns.tolist(),
    'scaler_amount': scaler_amount,
    'scaler_time': scaler_time
}, 'app/models/saved_model.pkl')
print("✅ Modèle XGBoost + SMOTE sauvegardé")