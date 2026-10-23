# 🛡️ Fraud Detection – Détection de fraude par carte bancaire

Application web de **détection de fraude sur transactions par carte bancaire**, construite avec **Flask** et du **Machine Learning**. Le projet couvre tout le pipeline : prétraitement des données, entraînement et comparaison de plusieurs modèles, puis déploiement dans une interface web avec explication des prédictions (SHAP) et tableau de bord.

## ✨ Fonctionnalités

- Prédiction de fraude en temps réel à partir d'un formulaire (`Time`, `Amount`, `V1` à `V28`)
- Probabilité de fraude et **niveau de risque** (Faible / Modéré / Élevé)
- **Explicabilité SHAP** : les 5 variables qui influencent le plus la décision
- Historique des prédictions enregistré dans une base **SQLite**
- Tableau de bord avec statistiques
- Comparaison de 4 modèles : Logistic Regression, Random Forest, XGBoost, Isolation Forest

## 🧰 Technologies

| Domaine | Outils |
|---|---|
| Backend | Flask, Flask-SQLAlchemy, SQLite |
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Explicabilité | SHAP |
| Data / Visualisation | pandas, NumPy, Matplotlib, Seaborn |
| Frontend | HTML / CSS (templates Jinja2) |

## 📊 Données

Le projet utilise le dataset **Credit Card Fraud Detection** (Kaggle / ULB) :

- 284 807 transactions, dont environ 0,17 % de fraudes (très déséquilibré)
- Variables `V1` à `V28` : composantes PCA anonymisées
- `Time`, `Amount` : variables brutes
- `Class` : 0 = normal, 1 = fraude

📥 Téléchargez `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le dans le dossier `data/`.

> ⚠️ Le fichier fait environ 150 Mo, ce qui dépasse la limite de GitHub (100 Mo). Il n'est donc pas inclus dans le dépôt.

## 🧠 Méthodologie

1. **Prétraitement** (`ml/preprocess.py`) : normalisation de `Amount` et `Time` avec deux `StandardScaler` indépendants, puis séparation train/test stratifiée (80/20).
2. **Gestion du déséquilibre** : SMOTE sur le jeu d'entraînement uniquement, et `class_weight='balanced'` pour certains modèles.
3. **Entraînement** (`ml/train.py`) :
   - Logistic Regression
   - Random Forest (`RandomizedSearchCV`)
   - XGBoost (`RandomizedSearchCV`)
   - Isolation Forest (non supervisé)
4. **Sélection du meilleur modèle supervisé** avec un score composite : `0.6 × F1 + 0.4 × AUC-ROC`.
5. **Évaluation** (`ml/evaluate.py`) : AUC-ROC, F1, précision, rappel, matrices de confusion, courbes ROC, importance des variables.
6. **Sauvegarde** du modèle, des scalers et des noms de variables dans `app/models/saved_model.pkl`.

### Résultats

| Modèle | AUC-ROC | F1 | Précision | Rappel |
|---|---|---|---|---|
| Logistic Regression | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |
| Random Forest | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |
| XGBoost | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |
| Isolation Forest | _à compléter_ | _à compléter_ | _à compléter_ | _à compléter_ |

Les graphiques générés sont dans [`notebooks/`](notebooks/) : matrices de confusion, courbes ROC, comparaison des modèles, importance des variables, etc.

## 📁 Structure du projet

```
fraud_detection2/
├── app/
│   ├── __init__.py            # Factory Flask, chargement du modèle
│   ├── routes.py              # Routes + modèle SQLAlchemy (PredictionLog)
│   ├── models/
│   │   ├── fraud_detector.py  # Classe de prédiction + SHAP
│   │   ├── saved_model.pkl    # Modèle entraîné + scalers
│   │   ├── isolation_forest.pkl
│   │   └── feature_names.pkl
│   └── templates/             # index.html, result.html, dashboard.html
├── ml/
│   ├── preprocess.py          # Prétraitement + SMOTE
│   ├── train.py               # Pipeline complet (4 modèles)
│   ├── train_xgboost.py       # Entraînement rapide XGBoost + SMOTE
│   ├── train_final.py         # Entraînement rapide Random Forest
│   └── evaluate.py            # Métriques et graphiques
├── notebooks/                 # Graphiques d'analyse et d'évaluation
├── data/                      # creditcard.csv (à télécharger)
├── requirements.txt
└── run.py                     # Point d'entrée de l'application
```

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/fraud_detection2.git
cd fraud_detection2

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Utilisation

### 1. Entraîner le modèle

Placez `creditcard.csv` dans `data/`, puis :

```bash
python ml/train.py
```

Ou pour un entraînement plus rapide :

```bash
python ml/train_xgboost.py   # XGBoost + SMOTE
python ml/train_final.py     # Random Forest
```

### 2. Lancer l'application

```bash
python run.py
```

L'application est accessible sur 👉 **http://localhost:5000**

## 🌐 Routes

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Formulaire de saisie d'une transaction |
| `/predict` | POST | Prédiction de fraude |
| `/history` | GET | Historique des prédictions |
| `/stats` | GET | Statistiques des prédictions |
| `/dashboard` | GET | Tableau de bord |

## 🔒 Niveaux de risque

| Probabilité de fraude | Niveau |
|---|---|
| < 30 % | 🟢 Faible |
| 30 % – 60 % | 🟡 Modéré |
| ≥ 60 % | 🔴 Élevé |

Une transaction est classée comme **fraude** si la probabilité est ≥ 50 %.

## ⚠️ Limites et remarques

- Projet à but **pédagogique** : ne pas utiliser tel quel en production.
- Changer la `SECRET_KEY` dans `app/__init__.py` et désactiver `debug=True` dans `run.py` avant tout déploiement.
- Les variables `V1`–`V28` étant anonymisées par PCA, l'interprétation métier des explications SHAP est limitée.

## 🔮 Améliorations possibles

- Ajouter une API REST (JSON) pour l'intégration avec d'autres services
- Dockeriser l'application
- Ajuster le seuil de décision selon le coût métier (faux négatifs vs faux positifs)
- Ajouter des tests unitaires

## 👤 Auteur

**<Votre nom>** – [GitHub](https://github.com/<votre-utilisateur>) – [LinkedIn](https://linkedin.com/in/<votre-profil>)

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE`.
