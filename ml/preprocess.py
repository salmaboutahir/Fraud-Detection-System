import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess(filepath='data/creditcard.csv'):
    df = pd.read_csv(filepath)
    print(f"Dataset : {df.shape[0]} lignes, fraudes {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

    if df.isnull().sum().any():
        df = df.dropna()

    # Deux scalers indépendants
    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()

    df['Amount_scaled'] = scaler_amount.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler_time.fit_transform(df[['Time']])

    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    os.makedirs('app/models', exist_ok=True)
    joblib.dump(X.columns.tolist(), 'app/models/feature_names.pkl')
    joblib.dump(scaler_amount, 'app/models/scaler_amount.pkl')
    joblib.dump(scaler_time,   'app/models/scaler_time.pkl')

    print(f"Features : {X.columns.tolist()[:5]}... (total {len(X.columns)})")
    return X, y, scaler_amount, scaler_time   # retourne les deux scalers

def split_and_resample(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Train after SMOTE : {X_train_res.shape[0]} ({y_train_res.mean()*100:.1f}% fraudes)")
    return X_train_res, X_test, y_train_res, y_test