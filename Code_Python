# ===================================================
# INSTALLATION & IMPORTS
# ===================================================
# pip install ucimlrepo seaborn matplotlib pandas numpy

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================
# CHARGEMENT DU DATASET
# ===================================================
bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets

# Fusion features + target
df = pd.concat([X, y], axis=1)

print("✅ Aperçu des données :")
print(df.head())

print("\n✅ Informations générales :")
print(df.info())

print("\n✅ Statistiques descriptives :")
print(df.describe())

# ===================================================
# IDENTIFICATION DES VARIABLES NUMÉRIQUES
# ===================================================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# ===================================================
# 1) VISUALISATION DES DISTRIBUTIONS : HISTOGRAMMES
# ===================================================
print("\n--- HISTOGRAMMES ---")

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()

    print(f"📌 Interprétation - {col} :")
    print(f"- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).")
    print(f"- Une asymétrie indique une concentration des valeurs vers une borne.")
    print(f"- La présence de pics suggère des groupes de clients distincts.\n")

# ===================================================
# 2) BOXPLOTS (détection des valeurs aberrantes)
# ===================================================
print("\n--- BOXPLOTS ---")

for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

    print(f"📌 Interprétation - {col} :")
    print("- Les points isolés représentent des valeurs aberrantes (outliers).")
    print("- Une boîte large indique une forte dispersion.")
    print("- Une asymétrie de la boîte suggère une distribution biaisée.\n")

# ===================================================
# 3) HEATMAP DES CORRÉLATIONS
# ===================================================
print("\n--- HEATMAP DES CORRÉLATIONS ---")

plt.figure(figsize=(10,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()

print("📌 Interprétation globale des corrélations :")
print("- Une corrélation proche de 1 ou -1 indique une relation forte.")
print("- Une valeur proche de 0 indique une indépendance.")
print("- Les relations fortes peuvent indiquer une redondance de variables.")
print("- Utile pour la sélection des variables dans les modèles prédictifs.")

# ===================================================
# 4) FEATURE ENGINEERING
# ===================================================
print("\n--- FEATURE ENGINEERING ---")

# 1️⃣ Création variable 'age_group'
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                          labels=['Jeune', 'Adulte', 'Senior', 'Très senior'])

# 2️⃣ Création variable durée par minute
df['duration_min'] = df['duration'] / 60

# 3️⃣ Création variable nombre total de contacts
df['total_contacts'] = df['campaign'] + df['previous']

print("✅ Nouvelles variables créées :")
print(df[['age_group', 'duration_min', 'total_contacts']].head())

print("\n📌 Interprétation des nouvelles variables :")
print("- age_group : segmentation client plus claire.")
print("- duration_min : plus interprétable que les secondes.")
print("- total_contacts : estimation de la pression commerciale sur le client.")

# ===================================================
# 5) VISUALISATION FEATURE ENGINEERING
# ===================================================
plt.figure(figsize=(6,4))
sns.histplot(df['duration_min'], kde=True)
plt.title("Distribution de la durée de contact (minutes)")
plt.show()
print("📌 Interprétation : Les appels courts dominent — peu d’appels longs très influents.")

plt.figure(figsize=(6,4))
sns.countplot(x='age_group', data=df)
plt.title("Répartition par groupe d'âge")
plt.show()
print("📌 Interprétation : Les adultes dominent le portefeuille clients.")

plt.figure(figsize=(6,4))
sns.histplot(df['total_contacts'], kde=True)
plt.title("Distribution du nombre total de contacts")
plt.show()
print("📌 Interprétation : La majorité des clients est contactée peu de fois, signe de ciblage sélectif.")

# ===================================================
# FIN
# ===================================================
print("\n✅ ANALYSE TERMINÉE WITH SUCCESS")
