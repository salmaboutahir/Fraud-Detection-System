# test_model.py
import joblib
import numpy as np
import pandas as pd

# 1. Charger le modèle, les scalers et l'ordre des features
model_data = joblib.load('app/models/saved_model.pkl')
model = model_data['model']
scaler_amount = model_data['scaler_amount']
scaler_time = model_data['scaler_time']
feature_names = model_data['feature_names']

print("✅ Modèle et scalers chargés avec succès.")
print(f"📋 Ordre des features : {feature_names[:5]} ... {feature_names[-3:]}\n")

def predict_transaction(amount, time, v_features):
    """
    v_features : dictionnaire contenant V1 à V28
    """
    # Normalisation
    amount_scaled = scaler_amount.transform([[amount]])[0, 0]
    time_scaled = scaler_time.transform([[time]])[0, 0]
    
    # Construction du vecteur dans l'ordre exact
    features = []
    for name in feature_names:
        if name == 'Amount_scaled':
            features.append(amount_scaled)
        elif name == 'Time_scaled':
            features.append(time_scaled)
        else:
            features.append(v_features.get(name, 0.0))
    
    X = np.array(features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    is_fraud = proba >= 0.5
    risk = "Élevé" if proba >= 0.7 else ("Modéré" if proba >= 0.3 else "Faible")
    return proba, is_fraud, risk

# ========== DÉFINITION DES TRANSACTIONS À TESTER ==========

# Transaction frauduleuse n°1 (celle que vous avez utilisée)
fraud1 = {
    'amount': 0.0,
    'time': 406.0,
    'v': {
        'V1': -2.3122265423263, 'V2': 1.95199201064158, 'V3': -1.60985073229769,
        'V4': 3.9979055875468, 'V5': -0.522187864667764, 'V6': -1.42654531920595,
        'V7': -2.53738730624579, 'V8': 1.39165724829804, 'V9': -2.77008927719433,
        'V10': -2.77227214465915, 'V11': 3.20203320709635, 'V12': -2.89990738849473,
        'V13': -0.595221881324605, 'V14': -4.28925378244217, 'V15': 0.389724120274487,
        'V16': -1.14074717980657, 'V17': -2.83005567450437, 'V18': -0.0168224681808257,
        'V19': 0.416955705037907, 'V20': 0.126910559061474, 'V21': 0.517232370861764,
        'V22': -0.0350493686052974, 'V23': -0.465211076182388, 'V24': 0.320198198514526,
        'V25': 0.0445191674731724, 'V26': 0.177839798284401, 'V27': 0.261145002567677,
        'V28': -0.143275874698919
    }
}

# Transaction frauduleuse n°2 (extraite du dataset)
fraud2 = {
    'amount': 1.99,
    'time': 406.0,
    'v': {
        'V1': -0.385380473965304, 'V2': 1.28066971600287, 'V3': 0.705551430235619,
        'V4': -0.0188413331502499, 'V5': 0.088097721031291, 'V6': -0.167584435847549,
        'V7': 0.120448307411582, 'V8': -0.583574832748166, 'V9': -0.408774102599696,
        'V10': 0.155280383438354, 'V11': 0.0679715729450539, 'V12': -0.527290406082452,
        'V13': -0.189375510426632, 'V14': 0.386627811192904, 'V15': -0.568651195228865,
        'V16': -0.559978042132319, 'V17': -0.503186741739953, 'V18': 0.101223921925049,
        'V19': 0.188143140515267, 'V20': 0.101647988754028, 'V21': 0.302972148940833,
        'V22': 0.127265838717184, 'V23': 0.159907050745515, 'V24': -0.446663785241098,
        'V25': 0.213807394580108, 'V26': 0.109220877490982, 'V27': -0.128769530044399,
        'V28': 0.0184673874772001
    }
}

# Transactions normales (toutes les V à 0, montants et temps variés)
normals = [
    {'amount': 50.0, 'time': 50000, 'desc': 'Normal typique'},
    {'amount': 5000, 'time': 100000, 'desc': 'Gros montant'},
    {'amount': 0.01, 'time': 0, 'desc': 'Très petit montant'},
    {'amount': 250.0, 'time': 86400, 'desc': 'Montant moyen, temps 24h'}
]

# ========== EXÉCUTION DES TESTS ==========
print("=" * 60)
print("🔍 TEST DU MODÈLE DE DÉTECTION DE FRAUDE")
print("=" * 60)

print("\n--- Transactions frauduleuses ---")
for idx, fraud in enumerate([fraud1, fraud2], 1):
    proba, is_fraud, risk = predict_transaction(fraud['amount'], fraud['time'], fraud['v'])
    print(f"Fraude #{idx}: Montant={fraud['amount']}€, Time={fraud['time']}s")
    print(f"  → Probabilité de fraude : {proba*100:.2f}%")
    print(f"  → Décision : {'FRAUDE' if is_fraud else 'NORMAL'}, Risque : {risk}\n")

print("--- Transactions normales ---")
for normal in normals:
    # Toutes les V à 0
    v_zero = {f'V{i}': 0.0 for i in range(1, 29)}
    proba, is_fraud, risk = predict_transaction(normal['amount'], normal['time'], v_zero)
    print(f"Normal : {normal['desc']} (Montant={normal['amount']}€, Time={normal['time']}s)")
    print(f"  → Probabilité de fraude : {proba*100:.2f}%")
    print(f"  → Décision : {'FRAUDE' if is_fraud else 'NORMAL'}, Risque : {risk}\n")

print("=" * 60)
print("✅ Test terminé.")
print("Pour tester d'autres valeurs, modifiez le dictionnaire dans le script.")