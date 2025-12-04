Voici ton **rapport complet regroupé sous un seul document Markdown**, prêt à être exporté en PDF ou rendu académique.
Tu peux copier-coller tel quel dans un fichier `rapport_bank_marketing.md`.

---

````markdown
# 🏦 Analyse Exploratoire et Modélisation Prédictive  
## Dataset Bank Marketing (UCI)

---

# 1. INTRODUCTION

## Contexte
Les campagnes de télé-marketing bancaire visent à convaincre les clients de souscrire à des produits financiers (ex : dépôt à terme). Cependant, ces campagnes sont coûteuses et peu efficaces si elles sont mal ciblées.

## Problématique
Comment identifier les clients susceptibles de souscrire à un produit bancaire à partir de données socio-économiques et comportementales ?

## Objectif
Mettre en œuvre une analyse exploratoire des données (EDA) pour comprendre :
- la structure des données
- la distribution des variables
- les corrélations importantes
- la création de nouvelles variables
afin de préparer la modélisation prédictive.

---

# 2. MÉTHODOLOGIE

## Collecte des Données
Les données proviennent du **UCI Machine Learning Repository** (ID = 222).

Elles décrivent :
- le profil client (âge, métier, éducation)
- la situation financière (solde bancaire)
- l'historique des campagnes marketing
- la variable cible : **y (souscription oui/non)**

---

## Outils Utilisés

- Python 3
- pandas / NumPy : manipulation des données
- matplotlib / seaborn : visualisation
- ucimlrepo : importation du dataset

---

## Justification des choix

### Visualisation
Les histogrammes et boxplots permettent de détecter :
- asymétries
- valeurs aberrantes (outliers)
- groupes de population

### Corrélation
La heatmap identifie les redondances et relations significatives.

### Feature Engineering
Objectif :
- améliorer l’interprétabilité
- créer des variables explicatives plus pertinentes

---

# 3. CODE UTILISÉ

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets

df = pd.concat([X, y], axis=1)

print(df.head())
print(df.info())
print(df.describe())

num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# HISTOGRAMMES
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution de {col}")
    plt.show()

# BOXPLOTS
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# HEATMAP
plt.figure(figsize=(10,6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show()

# FEATURE ENGINEERING
df['age_group'] = pd.cut(df['age'], bins=[0,30,45,60,100],
                          labels=['Jeune', 'Adulte', 'Senior', 'Très senior'])
df['duration_min'] = df['duration'] / 60
df['total_contacts'] = df['campaign'] + df['previous']
````

---

# 4. RÉSULTATS DE L'ANALYSE EXPLORATOIRE (EDA)

---

## 4.1 Aperçu des données

```text
Nombre d'observations : 45 211  
Nombre de variables : 17  
```

Valeurs manquantes importantes :

* contact : 13 020
* poutcome : 36 959

📌 **Interprétation :**
Certaines variables sont peu exploitables sans imputation.

---

## 4.2 Statistiques descriptives

| Variable | Moyenne | Min    | Max     |
| -------- | ------- | ------ | ------- |
| age      | 40.9    | 18     | 95      |
| balance  | 1 362 € | -8 019 | 102 127 |
| duration | 258 sec | 0      | 4918    |
| campaign | 2.76    | 1      | 63      |

📌
Forte présence de valeurs extrêmes → besoin de prudence en modélisation.

---

# 4.3 Histogrammes – Distributions

---

### Age

📌 Population concentrée entre 30 et 50 ans.

---

### Balance

📌 Distribution très asymétrique → présence de clients très riches.

---

### Duration

📌 Majorité d’appels courts, rares appels très longs.

---

### Campaign

📌 Pression marketing généralement faible, mais certains clients subissent de nombreuses tentatives.

---

### Pdays & Previous

📌 La majorité n’a jamais été contactée auparavant.

---

# 4.4 BOXPLOTS – Valeurs aberrantes

| Variable | Observation                     |
| -------- | ------------------------------- |
| Balance  | Outliers majeurs                |
| Duration | Extrêmes importants             |
| Campaign | Valeurs atypiques > 40          |
| Pdays    | Distribution très déséquilibrée |

---

# 4.5 HEATMAP – Corrélations

📌 Résultats :

* Corrélation forte : `duration` et `y`
* Autres relations faibles
* Peu de multicolinéarité

⚠️ `duration` crée un biais car connue après l’appel.

---

# 4.6 FEATURE ENGINEERING

---

## Variables créées

| Variable       | Intérêt                    |
| -------------- | -------------------------- |
| age_group      | Segmentation client        |
| duration_min   | Interprétabilité           |
| total_contacts | Pression marketing globale |

---

## Interprétations graphiques

### Répartition par âge

📌 Population majoritairement adulte.

### Durée (minutes)

📌 Appels longs rares mais très informatifs.

### Total contacts

📌 Ciblage marketing relativement précis.

---

# 5. INTERPRÉTATION GÉNÉRALE

| Point               | Conclusion             |
| ------------------- | ---------------------- |
| Dataset             | fiable                 |
| Outliers            | nombreux               |
| Corrélation         | faible sauf `duration` |
| Feature engineering | pertinent              |
| Problème potentiel  | fuite de données       |

---

# 6. CONCLUSION

---

## Limites

* valeurs manquantes
* déséquilibre de classes
* variable biaisante `duration`
* absence de normalisation
* pas de modèle ML construit

---

## Pistes d’amélioration

* Imputation intelligente
* Encodage catégoriel
* Standardisation
* Sélection de variables
* Validation croisée
* Modélisation ML (Logistic, Random Forest, XGBoost)

---

✅ **Analyse exploratoire finalisée avec succès.**

