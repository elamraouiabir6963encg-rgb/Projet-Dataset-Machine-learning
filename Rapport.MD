# EL Amraoui Abir ENCG SETTAT Finance G2
# Apogé: 24010353
![Abir](https://github.com/user-attachments/assets/19b86bed-ac87-4f8e-938c-862cf3722380)

# 🏦 Analyse Exploratoire et Modélisation Prédictive  
## Dataset Bank Marketing (UCI)

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Les banques utilisent des campagnes de télé‑marketing pour convaincre les clients de souscrire à un dépôt à terme. Or :
    *   Ces campagnes coûtent cher,
    *   La majorité des clients ne répondent pas favorablement,
    *   Un mauvais ciblage = perte d’argent + surcharge des centres d’appels. 
*   **Objectif :** Créer un "Assistant IA" pour prédire quels clients sont susceptibles de souscrire afin d'optimiser les campagnes.
*   **L'Enjeu critique :** La matrice des coûts d'erreur est asymétrique.
    *   Faux Positif: On appelle un client qui ne sera jamais intéressé → perte de temps et coût opérationnel.
    *   Faux Négatif: on rate un client réellement intéressé → manque à gagner commercial. **L'IA doit donc maximiser le rappel pour ne pas manquer des clients intéressés.**

### Les Données (L'Input)
On utilise le *Bank Marketing (UCI)*.
*   Profil client : âge, métier, éducation…
*   Situation financière : solde bancaire
*   Historique marketing : nombre de contacts, résultat des anciennes campagnes…
*   cible : y = oui/non (a souscrit ou non)
*   Format : 45 211 observations, 17 colonnes.
---
## 2. Le Code Python 

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
## 3. Analyse Approfondie : Nettoyage 

### Le Problème des données bruts
Le dataset comporte :
    *   Des valeurs manquantes (job, education, contact, poutcome),
    *   Des valeurs extrêmes (ex : balance, duration, previous),
    *   Des distributions fortement asymétriques. 
Ces problèmes perturbent les modèles de Machine Learning car :
    *   Les distributions biaisées trompent les algorithmes,
    *   Les outliers influencent les décisions,
    *   Les valeurs manquantes empêchent certains algorithmes de s’exécuter. 
### La fuite des données
Le modèle final devra toujours éviter :
*   Calcul de moyennes sur l’ensemble du dataset,
*   Transformation avant séparation Train/Test.
Ici, l’objectif étant l’EDA, la priorité est l’analyse, pas la performance.
---
# 4. Justification des choix
*   **Visualisation :** 
    *   Asymétries
    *   Valeurs aberrantes (outliers)
    *   Groupes de population
*   **Corrélation :** La heatmap identifie les redondances et relations significatives.
*   **Feature Engineering :** Objectifs.
    *   Améliorer l’interprétabilité
    *   Créer des variables explicatives plus pertinentes

---

# 5. RÉSULTATS DE L'ANALYSE EXPLORATOIRE (EDA)

---

## 5.1 Aperçu des données
![Aperçu des données](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Aper%C3%A7u%20des%20donn%C3%A9es.png?raw=true)

## 5.2 Statistiques descriptives
![Statistiques descreptives](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Statistiques%20descreptives.png?raw=true)
📌
Forte présence de valeurs extrêmes → besoin de prudence en modélisation.

---

# 5.3 Histogrammes – Distributions

### Age
![Distrtibution de Age](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20Age.png?raw=true)

📌 Interprétation - age :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.


### Balance
![Distribbution de Balance](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20Balance.png?raw=true)
📌 Interprétation - balance :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

### Day of week
![Distribbution de pdays_of_week](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20day_of_week.png?raw=true)

📌 Interprétation - day_of_week :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

### Duration
![Distribbution de duration](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20duration.png?raw=true)

📌 Interprétation - duration :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

### Campaign
![Distribbution de campaign](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20campaign.png?raw=true)

📌 Interprétation - campaign :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

### Pdays
![Distribbution de pdays](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20pdays.png?raw=true)

📌 Interprétation - pdays :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

### Previous
![Distribbution de previous](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20previous.png?raw=true)

📌 Interprétation - previous :
- La distribution permet d’observer la forme générale (symétrique, asymétrique, extrêmes).
- Une asymétrie indique une concentration des valeurs vers une borne.
- La présence de pics suggère des groupes de clients distincts.

---

# 5.4 BOXPLOTS 

### Age
![Boxplot de age](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20age.png?raw=true)

📌 Interprétation - age :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Balance
![Boxplot de Balance](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20Balance.png?raw=true)
📌 Interprétation - balance :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Day of week
![Boxplot de day_of_week](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20day%20of%20week.png?raw=true)

📌 Interprétation - day_of_week :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Duration
![Boxplot de duration](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20duration.png?raw=true)

📌 Interprétation - duration :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Campaign
![Boxplot de campaign](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20campaign.png?raw=true)

📌 Interprétation - campaign :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Pdays
![Boxplot de pdays](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20pdays.png?raw=true)

📌 Interprétation - pdays :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

### Previous
![Boxplot de previous](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Boxplot%20de%20previous.png?raw=true)

📌 Interprétation - previous :
- Les points isolés représentent des valeurs aberrantes (outliers).
- Une boîte large indique une forte dispersion.
- Une asymétrie de la boîte suggère une distribution biaisée.

# 5.5 HEATMAP – Corrélations
![Matrice de corrélation](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Matrice%20de%20corr%C3%A9lation.png?raw=true)

📌 Interprétation globale des corrélations :
- Une corrélation proche de 1 ou -1 indique une relation forte.
- Une valeur proche de 0 indique une indépendance.
- Les relations fortes peuvent indiquer une redondance de variables.
- Utile pour la sélection des variables dans les modèles prédictifs.

--- FEATURE ENGINEERING ---
✅ Nouvelles variables créées :
  age_group  duration_min  total_contacts
0    Senior      4.350000               1
1    Adulte      2.516667               1
2    Adulte      1.266667               1
3    Senior      1.533333               1
4    Adulte      3.300000               1

📌 Interprétation des nouvelles variables :
- age_group : segmentation client plus claire.
- duration_min : plus interprétable que les secondes.
- total_contacts : estimation de la pression commerciale sur le client.

### Répartition par âge
![Répartition par groupe d'âge](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/R%C3%A9partition%20par%20groupe%20d'%C3%A2ge.png?raw=true)

📌 Interprétation : Les adultes dominent le portefeuille clients.

### Durée (minutes)
![Distribution de la durée de contact](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20de%20la%20dur%C3%A9e%20de%20contact.png?raw=true)

📌 Interprétation : Les appels courts dominent — peu d’appels longs très influents.

### Total contacts
![Distribution du nobre total de contacts](https://github.com/elamraouiabir6963encg-rgb/Projet-Dataset-Machine-learning/blob/main/Image/Distribution%20du%20nombre%20total%20de%20contacts.png?raw=true)

📌 Interprétation : La majorité des clients est contactée peu de fois, signe de ciblage sélectif.

---

# 6. INTERPRÉTATION GÉNÉRALE

| Point               | Conclusion             |
| ------------------- | ---------------------- |
| Dataset             | fiable                 |
| Outliers            | nombreux               |
| Corrélation         | faible sauf `duration` |
| Feature engineering | pertinent              |
| Problème potentiel  | fuite de données       |

---

# 7. CONCLUSION

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

