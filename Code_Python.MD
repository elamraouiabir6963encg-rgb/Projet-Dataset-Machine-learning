# 📘 RAPPORT D’ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
## Dataset : Bank Marketing — UCI Machine Learning Repository

---

# 1. INTRODUCTION

## 1.1 Contexte

Dans un environnement bancaire hautement concurrentiel, les institutions doivent optimiser leurs campagnes marketing afin de cibler efficacement les clients susceptibles de souscrire à des produits financiers. Le dataset **Bank Marketing**, utilisé dans ce projet, contient les données issues de campagnes téléphoniques d'une banque portugaise visant la souscription à des dépôts à terme.

---

## 1.2 Problématique

> Comment analyser les données des clients afin d’identifier les facteurs influençant la souscription à un dépôt bancaire ?

---

## 1.3 Objectifs

- Comprendre la structure globale des données  
- Explorer les relations entre variables  
- Détecter les valeurs aberrantes  
- Mettre en œuvre du Feature Engineering  
- Préparer les données pour des modèles prédictifs futurs  

---

# 2. MÉTHODOLOGIE

---

## 2.1 Chargement et préparation des données

Les données sont importées depuis le dépôt **UCI Machine Learning Repository** grâce à la bibliothèque `ucimlrepo`.  
Les variables explicatives (X) et la variable cible (y) sont fusionnées dans un unique DataFrame.

---

## 2.2 Code utilisé

```python
# INSTALLATION & IMPORTS
# pip install ucimlrepo seaborn matplotlib pandas numpy

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CHARGEMENT DU DATASET
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

# IDENTIFICATION DES VARIABLES NUMÉRIQUES
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# 1) VISUALISATION DES DISTRIBUTIONS : HISTOGRAMMES
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()

# 2) BOXPLOTS (détection des valeurs aberrantes)
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# 3) HEATMAP DES CORRÉLATIONS
plt.figure(figsize=(10,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()

# FEATURE ENGINEERING
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                          labels=['Jeune', 'Adulte', 'Senior', 'Très senior'])
df['duration_min'] = df['duration'] / 60
df['total_contacts'] = df['campaign'] + df['previous']

# VISUALISATION FEATURE ENGINEERING
sns.histplot(df['duration_min'], kde=True); plt.show()
sns.countplot(x='age_group', data=df); plt.show()
sns.histplot(df['total_contacts'], kde=True); plt.show()

print("✅ ANALYSE TERMINÉE")
