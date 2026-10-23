import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. Chargement ──────────────────────────────────────────────────────────
df = pd.read_csv('data/creditcard.csv')

print("Shape:", df.shape)           # (284807, 31)
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())            # Vérifier les valeurs manquantes

# ─── 2. Déséquilibre de classes ─────────────────────────────────────────────
class_counts = df['Class'].value_counts()
print(f"\nNon-fraudes : {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
print(f"Fraudes     : {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")

plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette=['green','red'], legend=False)
plt.xticks([0, 1], ['Normal', 'Fraude'])
plt.title("Distribution des classes")
plt.ylabel("Nombre de transactions")
plt.savefig('notebooks/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── 3. Distribution de Amount et Time ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Amount : très étalé, on vérifie par classe
df_fraud = df[df['Class'] == 1]
df_normal = df[df['Class'] == 0]

axes[0].hist(df_normal['Amount'], bins=50, alpha=0.6, label='Normal', color='steelblue')
axes[0].hist(df_fraud['Amount'], bins=50, alpha=0.8, label='Fraude', color='crimson')
axes[0].set_xlabel('Montant (€)')
axes[0].set_title('Distribution du montant par classe')
axes[0].legend()
axes[0].set_yscale('log')  # Log scale pour mieux voir

axes[1].hist(df_normal['Time'], bins=50, alpha=0.6, label='Normal', color='steelblue')
axes[1].hist(df_fraud['Time'], bins=50, alpha=0.8, label='Fraude', color='crimson')
axes[1].set_xlabel('Temps (secondes)')
axes[1].set_title('Distribution du temps par classe')
axes[1].legend()

plt.tight_layout()
plt.savefig('notebooks/amount_time_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── 4. Heatmap de corrélation ───────────────────────────────────────────────
plt.figure(figsize=(14, 10))
corr = df.corr()
sns.heatmap(corr, cmap='RdBu_r', center=0, linewidths=0.5,
            xticklabels=True, yticklabels=True)
plt.title("Matrice de corrélation des features")
plt.savefig('notebooks/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ─── 5. Statistiques par classe ──────────────────────────────────────────────
print("\n=== Statistiques des fraudes ===")
print(df_fraud[['Amount', 'Time']].describe())

print("\n=== Statistiques normales ===")
print(df_normal[['Amount', 'Time']].describe())

# Features les plus corrélées avec Class
correlations_with_class = df.corr()['Class'].abs().sort_values(ascending=False)
print("\n=== Top 10 features corrélées avec Class ===")
print(correlations_with_class.head(11))  # 11 car Class lui-même est en tête