import numpy as np
import shap

class FraudDetector:
    def __init__(self, model, scaler_amount, scaler_time, feature_names):
        self.model = model
        self.scaler_amount = scaler_amount
        self.scaler_time = scaler_time
        self.feature_names = feature_names   # ex: ['V1','V2',...,'V28','Amount_scaled','Time_scaled']

        # SHAP explainer (arbres uniquement)
        try:
            if hasattr(model, 'estimators_') or 'XGB' in str(type(model)):
                self.explainer = shap.TreeExplainer(model)
                self.shap_available = True
            else:
                self.shap_available = False
        except Exception:
            self.shap_available = False

    def predict(self, transaction_dict):
        """
        transaction_dict doit contenir les clés :
            'Amount' (brut, float),
            'Time'   (brut en secondes, float),
            'V1' à 'V28' (déjà normalisées par PCA)
        """
        # ========== DEBOGAGE : afficher la transaction reçue ==========
        print("\n=== [DEBUG] predict() ===")
        print("[DEBUG] Clés du dictionnaire reçu :", list(transaction_dict.keys()))
        print("[DEBUG] Contenu (extrait) :")
        for k in ['Amount', 'Time'] + [f'V{i}' for i in range(1,6)]:
            print(f"  {k} = {transaction_dict.get(k, 'ABSENT')}")

        # 1. Valeurs brutes
        amount_raw = float(transaction_dict.get('Amount', 0.0))
        time_raw   = float(transaction_dict.get('Time', 0.0))
        print(f"[DEBUG] amount_raw = {amount_raw}")
        print(f"[DEBUG] time_raw   = {time_raw}")

        # 2. Normalisation avec les scalers entraînés
        amount_scaled = self.scaler_amount.transform([[amount_raw]])[0, 0]
        time_scaled   = self.scaler_time.transform([[time_raw]])[0, 0]
        print(f"[DEBUG] amount_scaled = {amount_scaled:.6f}")
        print(f"[DEBUG] time_scaled   = {time_scaled:.6f}")

        # 3. Construction du vecteur dans l'ordre exact des features d'entraînement
        features = []
        print("[DEBUG] Ordre des features attendu (10 premiers) :", self.feature_names[:10])
        for idx, name in enumerate(self.feature_names):
            if name == 'Amount_scaled':
                features.append(amount_scaled)
                print(f"[DEBUG] Index {idx}: ajout Amount_scaled = {amount_scaled:.6f}")
            elif name == 'Time_scaled':
                features.append(time_scaled)
                print(f"[DEBUG] Index {idx}: ajout Time_scaled = {time_scaled:.6f}")
            else:
                val = transaction_dict.get(name, 0.0)
                features.append(float(val))
                if idx < 5:   # afficher les 5 premières features V
                    print(f"[DEBUG] Index {idx}: {name} = {val}")

        print(f"[DEBUG] Nombre total de features : {len(features)}")
        print(f"[DEBUG] 5 premières features : {features[:5]}")
        print(f"[DEBUG] 5 dernières features : {features[-5:]}")

        X = np.array(features, dtype=np.float64).reshape(1, -1)

        # 4. Prédiction
        probability = float(self.model.predict_proba(X)[0][1])
        print(f"[DEBUG] Probabilité brute (classe 1 / fraude) = {probability:.6f}")

        is_fraud = probability >= 0.5

        # Niveau de risque
        if probability < 0.3:
            risk_level = "Faible"
            risk_color = "success"
        elif probability < 0.6:
            risk_level = "Modéré"
            risk_color = "warning"
        else:
            risk_level = "Élevé"
            risk_color = "danger"

        # SHAP (optionnel)
        shap_data = {}
        if self.shap_available:
            try:
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]   # classe 1 = fraude
                else:
                    sv = shap_values[0]
                impacts = sorted(zip(self.feature_names, sv.tolist()),
                                 key=lambda x: abs(x[1]), reverse=True)
                shap_data = {k: round(float(v), 4) for k, v in impacts[:5]}
            except Exception as e:
                shap_data = {'error': str(e)[:60]}

        return {
            'is_fraud': bool(is_fraud),
            'probability': round(probability * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'shap_values': shap_data
        }