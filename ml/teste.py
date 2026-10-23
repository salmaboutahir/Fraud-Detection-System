import pandas as pd
import joblib

# Charger les données
df = pd.read_csv('data/creditcard.csv')

# Prendre la première transaction frauduleuse
fraud = df[df['Class']==1].iloc[0]

print("=== Transaction frauduleuse réelle ===")
print(f"Amount brut : {fraud['Amount']}")
print(f"Time brut   : {fraud['Time']}")
for i in range(1, 29):
    print(f"V{i} : {fraud[f'V{i}']}")