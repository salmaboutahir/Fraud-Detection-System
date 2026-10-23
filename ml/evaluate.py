import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score
)
import os

os.makedirs('notebooks', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLES SUPERVISÉS  (Random Forest, XGBoost, Logistic Regression)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Évalue un modèle SUPERVISÉ qui possède predict_proba().
    Calcule AUC-ROC, F1, Precision, Recall et trace la matrice de confusion.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # probabilité classe 1 (fraude)

    auc       = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  AUC-ROC   : {auc:.4f}  {'✓ OK' if auc > 0.95 else '✗ < 0.95 requis'}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Précision : {precision:.4f}")
    print(f"  Rappel    : {recall:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Fraude'])}")

    _plot_confusion_matrix(y_test, y_pred, name)

    return {
        'AUC-ROC'  : auc,
        'F1'       : f1,
        'Precision': precision,
        'Recall'   : recall
    }


# ══════════════════════════════════════════════════════════════════════════════
# ISOLATION FOREST  (modèle NON-SUPERVISÉ)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_isolation_forest(iso_model, X_test, y_test):
    """
    Évalue Isolation Forest (modèle NON-SUPERVISÉ).

    Différences avec les modèles supervisés :
    ┌─────────────────────────────────────────────────────────────┐
    │  predict()          retourne -1 (anomalie) ou +1 (normal)  │
    │  predict_proba()    N'EXISTE PAS                            │
    │  decision_function() retourne un score : plus bas = plus   │
    │                      suspect (anomalie)                     │
    └─────────────────────────────────────────────────────────────┘

    Conversion :
      -1  →  1  (fraude détectée)
      +1  →  0  (transaction normale)

    Score pour AUC-ROC :
      On inverse decision_function() car valeur basse = anomalie
      Puis on normalise entre 0 et 1
    """
    # ── Prédictions : convertir -1/+1 en 1/0 ────────────────────────────────
    raw_pred = iso_model.predict(X_test)
    y_pred   = np.where(raw_pred == -1, 1, 0)

    # ── Score d'anomalie normalisé pour AUC-ROC ──────────────────────────────
    # decision_function() : valeur négative basse = très suspect
    # On inverse avec le signe - pour que haut = fraude
    raw_scores  = -iso_model.decision_function(X_test)
    # Normalisation [0, 1] pour pouvoir calculer l'AUC
    scores_norm = (raw_scores - raw_scores.min()) / \
                  (raw_scores.max() - raw_scores.min() + 1e-10)

    auc       = roc_auc_score(y_test, scores_norm)
    f1        = f1_score(y_test, y_pred,       zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,   zero_division=0)

    print(f"\n{'─'*45}")
    print(f"  Isolation Forest  (NON-SUPERVISÉ)")
    print(f"{'─'*45}")
    print(f"  AUC-ROC   : {auc:.4f}  (attendu ~0.75-0.85 normal pour non-supervisé)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Précision : {precision:.4f}")
    print(f"  Rappel    : {recall:.4f}")
    print(f"\n  Note : AUC plus bas car il n'utilise PAS les labels")
    print(f"         Il détecte les anomalies structurelles (transactions rares)")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Fraude'], zero_division=0)}")

    _plot_confusion_matrix(y_test, y_pred, "Isolation Forest")

    return {
        'AUC-ROC'  : auc,
        'F1'       : f1,
        'Precision': precision,
        'Recall'   : recall
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(y_test, y_pred, name):
    """Trace et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_test, y_pred)

    # Calcul des pourcentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Matrice brute
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'],
                ax=axes[0])
    axes[0].set_title(f'Matrice de confusion — {name}')
    axes[0].set_ylabel('Réel')
    axes[0].set_xlabel('Prédit')

    # Matrice en pourcentage
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'],
                ax=axes[1])
    axes[1].set_title(f'Matrice de confusion (%) — {name}')
    axes[1].set_ylabel('Réel')
    axes[1].set_xlabel('Prédit')

    plt.tight_layout()
    fname = f"notebooks/confusion_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Matrice sauvegardée : {fname}")


def plot_roc_curves(models_dict, X_test, y_test):
    """
    Trace les courbes ROC pour tous les modèles.
    Gère automatiquement :
      - predict_proba() pour les modèles supervisés
      - decision_function() pour Isolation Forest
    """
    plt.figure(figsize=(9, 7))
    colors = ['#64748b', '#1a3a8f', '#dc2626', '#059669']
    styles = ['-', '-', '-', '--']   # Isolation Forest en pointillé (non-supervisé)

    for (name, model), color, style in zip(models_dict.items(), colors, styles):

        # Supervisé → predict_proba disponible
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test)[:, 1]

        # Isolation Forest → pas de predict_proba, on utilise decision_function
        else:
            raw    = -model.decision_function(X_test)
            scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)

        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)

        label = f'{name} (AUC = {auc:.3f})'
        if name == 'Isolation Forest':
            label += '  ← non-supervisé'

        plt.plot(fpr, tpr, color=color, lw=2, linestyle=style, label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Aléatoire (AUC = 0.500)')
    plt.axvline(x=0.1, color='orange', lw=1, alpha=0.5, linestyle=':',
                label='FPR = 10% (zone opérationnelle)')

    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC — Comparaison des 4 modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('notebooks/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Courbes ROC sauvegardées : notebooks/roc_curves.png")


def plot_feature_importance(model, feature_names, top_n=15):
    """Visualise les features les plus importantes (RF et XGBoost)."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    colors = ['#1a3a8f' if i < 5 else '#64748b' for i in range(top_n)]
    plt.bar(range(top_n), importances[indices], color=colors)
    plt.xticks(range(top_n),
               [feature_names[i] for i in indices],
               rotation=45, ha='right', fontsize=9)
    plt.title(f'Top {top_n} features importantes', fontsize=13, fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('notebooks/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance sauvegardée : notebooks/feature_importance.png")


def plot_model_comparison(comparison_df):
    """Bar chart comparant les 4 modèles sur 4 métriques."""
    metrics = ['AUC-ROC', 'F1', 'Precision', 'Recall']
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ['#64748b', '#1a3a8f', '#dc2626', '#059669']

    for ax, metric in zip(axes, metrics):
        vals   = comparison_df[metric]
        bars   = ax.bar(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric, fontweight='bold')
        ax.axhline(0.95, color='red', lw=1, linestyle='--', alpha=0.5)

        # Afficher les valeurs sur les barres
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Comparaison des 4 modèles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('notebooks/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Comparaison modèles sauvegardée : notebooks/model_comparison.png")
