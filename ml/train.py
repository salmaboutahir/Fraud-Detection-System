import numpy as np
import pandas as pd
import joblib
import os

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, IsolationForest
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

from preprocess import load_and_preprocess
from evaluate   import (
    evaluate_model,
    evaluate_isolation_forest,
    plot_roc_curves,
    plot_feature_importance,
    plot_model_comparison
)

RANDOM_STATE = 42
os.makedirs('app/models', exist_ok=True)
os.makedirs('notebooks',  exist_ok=True)


def train_all_models():
    """
    Pipeline complet :
    Chargement → Prétraitement → Entraînement × 4 → Évaluation → Sauvegarde
    """
    print("=" * 55)
    print("  ÉTAPE 1 — CHARGEMENT ET PRÉTRAITEMENT")
    print("=" * 55)

    # Récupère X, y et les deux scalers distincts
    X, y, scaler_amount, scaler_time = load_and_preprocess('data/creditcard.csv')

    # ── Split train / test stratifié ─────────────────────────────────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        stratify     = y,
        random_state = RANDOM_STATE
    )

    # ── Sous-ensemble rapide pour hyperParam search ───────────────────────────
    fraud_idx  = y_train_full[y_train_full == 1].index
    normal_idx = y_train_full[y_train_full == 0].sample(10000, random_state=42).index
    fast_idx   = fraud_idx.append(normal_idx)
    X_fast     = X_train_full.loc[fast_idx]
    y_fast     = y_train_full.loc[fast_idx]

    # ── SMOTE : uniquement pour les modèles SUPERVISÉS ───────────────────────
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train, y_train = smote.fit_resample(X_train_full, y_train_full)

    print(f"\nDonnées prêtes :")
    print(f"  X_train (SMOTE) : {X_train.shape[0]:>7} lignes")
    print(f"  X_train_full    : {X_train_full.shape[0]:>7} lignes")
    print(f"  X_fast          : {X_fast.shape[0]:>7} lignes")
    print(f"  X_test          : {X_test.shape[0]:>7} lignes")
    print(f"  Fraudes test    : {y_test.sum():>7}")

    results        = {}
    trained_models = {}

    # ════════════════════════════════════════════════════════════════════════
    # MODÈLE 1 — LOGISTIC REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  MODÈLE 1/4 — Logistic Regression")
    print("=" * 55)

    lr = LogisticRegression(
        class_weight = 'balanced',
        max_iter     = 500,
        random_state = RANDOM_STATE
    )
    lr.fit(X_train, y_train)

    results['Logistic Regression'] = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    trained_models['Logistic Regression'] = lr

    # ════════════════════════════════════════════════════════════════════════
    # MODÈLE 2 — RANDOM FOREST
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  MODÈLE 2/4 — Random Forest")
    print("=" * 55)

    rf_params = {
        'n_estimators'     : [100, 200, 300],
        'max_depth'        : [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf' : [1, 2],
    }

    rf_search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight = 'balanced',
            random_state = RANDOM_STATE,
            n_jobs       = -1
        ),
        rf_params,
        n_iter       = 8,
        cv           = 3,
        scoring      = 'roc_auc',
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        verbose      = 0
    )
    rf_search.fit(X_fast, y_fast)
    print(f"  Meilleurs params : {rf_search.best_params_}")

    rf_best = RandomForestClassifier(
        **rf_search.best_params_,
        class_weight = 'balanced',
        random_state = RANDOM_STATE,
        n_jobs       = -1
    )
    rf_best.fit(X_train, y_train)

    results['Random Forest'] = evaluate_model(rf_best, X_test, y_test, "Random Forest")
    trained_models['Random Forest'] = rf_best

    # ════════════════════════════════════════════════════════════════════════
    # MODÈLE 3 — XGBOOST
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  MODÈLE 3/4 — XGBoost")
    print("=" * 55)

    xgb_params = {
        'n_estimators'    : [100, 200, 300],
        'max_depth'       : [3, 5, 7],
        'learning_rate'   : [0.01, 0.05, 0.1],
        'subsample'       : [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    xgb_search = RandomizedSearchCV(
        XGBClassifier(
            tree_method  = 'hist',
            eval_metric  = 'logloss',
            random_state = RANDOM_STATE,
            n_jobs       = -1
        ),
        xgb_params,
        n_iter       = 8,
        cv           = 3,
        scoring      = 'roc_auc',
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        verbose      = 0
    )
    xgb_search.fit(X_fast, y_fast)
    print(f"  Meilleurs params : {xgb_search.best_params_}")

    xgb_best = XGBClassifier(
        **xgb_search.best_params_,
        tree_method  = 'hist',
        eval_metric  = 'logloss',
        random_state = RANDOM_STATE,
        n_jobs       = -1
    )
    xgb_best.fit(X_train, y_train)

    results['XGBoost'] = evaluate_model(xgb_best, X_test, y_test, "XGBoost")
    trained_models['XGBoost'] = xgb_best

    # ════════════════════════════════════════════════════════════════════════
    # MODÈLE 4 — ISOLATION FOREST (NON-SUPERVISÉ)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  MODÈLE 4/4 — Isolation Forest")
    print("=" * 55)

    contamination = float(y_train_full.mean())
    print(f"  Contamination estimée : {contamination:.4f}")

    iso_forest = IsolationForest(
        n_estimators = 200,
        contamination = contamination,
        max_samples = 'auto',
        max_features = 1.0,
        bootstrap = False,
        random_state = RANDOM_STATE,
        n_jobs = -1
    )
    iso_forest.fit(X_train_full)

    iso_result = evaluate_isolation_forest(iso_forest, X_test, y_test)
    trained_models['Isolation Forest'] = iso_forest
    results['Isolation Forest'] = iso_result

    # ════════════════════════════════════════════════════════════════════════
    # COMPARAISON ET SÉLECTION DU MEILLEUR MODÈLE (supervisés uniquement)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  COMPARAISON FINALE DES MODÈLES SUPERVISÉS")
    print("=" * 55)

    comparison_df = pd.DataFrame(results).T
    supervised_df = comparison_df.drop('Isolation Forest').copy()
    supervised_df['Score_composite'] = supervised_df['F1'] * 0.60 + supervised_df['AUC-ROC'] * 0.40

    print(supervised_df[['AUC-ROC', 'F1', 'Precision', 'Recall', 'Score_composite']].to_string())

    best_name = supervised_df['Score_composite'].idxmax()
    best_model = trained_models[best_name]

    print(f"\n  ★ Meilleur modèle : {best_name}")
    print(f"    Score composite : {supervised_df.loc[best_name, 'Score_composite']:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # SAUVEGARDE (avec les deux scalers)
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  SAUVEGARDE DES MODÈLES ET SCALERS")
    print("=" * 55)

    best_metrics = comparison_df.loc[best_name].to_dict()
    best_metrics['Score_composite'] = supervised_df.loc[best_name, 'Score_composite']

    joblib.dump({
        'model'          : best_model,
        'model_name'     : best_name,
        'metrics'        : best_metrics,
        'feature_names'  : X.columns.tolist(),
        'scaler_amount'  : scaler_amount,
        'scaler_time'    : scaler_time,
        'all_metrics'    : comparison_df.to_dict()
    }, 'app/models/saved_model.pkl')
    print("  ✓ saved_model.pkl (avec scalers intégrés)")

    joblib.dump(iso_forest, 'app/models/isolation_forest.pkl')
    print("  ✓ isolation_forest.pkl")

    # ════════════════════════════════════════════════════════════════════════
    # GRAPHIQUES
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 55)
    print("  GÉNÉRATION DES GRAPHIQUES")
    print("=" * 55)

    plot_roc_curves(
        {
            'Logistic Regression': lr,
            'Random Forest'      : rf_best,
            'XGBoost'            : xgb_best,
            'Isolation Forest'   : iso_forest
        },
        X_test, y_test
    )

    if hasattr(best_model, 'feature_importances_'):
        plot_feature_importance(best_model, X.columns.tolist())

    plot_model_comparison(comparison_df)

    # Si la fonction existe dans evaluate.py
    try:
        from evaluate import plot_all_confusion_matrices
        plot_all_confusion_matrices(trained_models, results, X_test, y_test)
    except ImportError:
        print("  (plot_all_confusion_matrices non trouvée, ignorée)")

    print("\n" + "=" * 55)
    print("  ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 55)
    print(f"  Modèle choisi   : {best_name}")
    print(f"  Lancez maintenant : python run.py")

    return best_model


if __name__ == '__main__':
    train_all_models()