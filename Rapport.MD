# EL Amraoui Abir ENCG SETTAT Finance G2
![Abir](https://github.com/user-attachments/assets/19b86bed-ac87-4f8e-938c-862cf3722380)

# 🏦 Analyse Exploratoire et Modélisation Prédictive  
## Dataset Bank Marketing (UCI)

---

## Le problème: 
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
   age           job  marital  education default  balance housing loan  \
0   58    management  married   tertiary      no     2143     yes   no   
1   44    technician   single  secondary      no       29     yes   no   
2   33  entrepreneur  married  secondary      no        2     yes  yes   
3   47   blue-collar  married        NaN      no     1506     yes   no   
4   33           NaN   single        NaN      no        1      no   no   

  contact  day_of_week month  duration  campaign  pdays  previous poutcome   y  
0     NaN            5   may       261         1     -1         0      NaN  no  
1     NaN            5   may       151         1     -1         0      NaN  no  
2     NaN            5   may        76         1     -1         0      NaN  no  
3     NaN            5   may        92         1     -1         0      NaN  no  
4     NaN            5   may       198         1     -1         0      NaN  no  

✅ Informations générales :
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 45211 entries, 0 to 45210
Data columns (total 17 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   age          45211 non-null  int64 
 1   job          44923 non-null  object
 2   marital      45211 non-null  object
 3   education    43354 non-null  object
 4   default      45211 non-null  object
 5   balance      45211 non-null  int64 
 6   housing      45211 non-null  object
 7   loan         45211 non-null  object
 8   contact      32191 non-null  object
 9   day_of_week  45211 non-null  int64 
 10  month        45211 non-null  object
 11  duration     45211 non-null  int64 
 12  campaign     45211 non-null  int64 
 13  pdays        45211 non-null  int64 
 14  previous     45211 non-null  int64 
 15  poutcome     8252 non-null   object
 16  y            45211 non-null  object
dtypes: int64(7), object(10)
memory usage: 5.9+ MB
None


## 4.2 Statistiques descriptives
              age        balance   day_of_week      duration      campaign  \
count  45211.000000   45211.000000  45211.000000  45211.000000  45211.000000   
mean      40.936210    1362.272058     15.806419    258.163080      2.763841   
std       10.618762    3044.765829      8.322476    257.527812      3.098021   
min       18.000000   -8019.000000      1.000000      0.000000      1.000000   
25%       33.000000      72.000000      8.000000    103.000000      1.000000   
50%       39.000000     448.000000     16.000000    180.000000      2.000000   
75%       48.000000    1428.000000     21.000000    319.000000      3.000000   
max       95.000000  102127.000000     31.000000   4918.000000     63.000000   

              pdays      previous  
count  45211.000000  45211.000000  
mean      40.197828      0.580323  
std      100.128746      2.303441  
min       -1.000000      0.000000  
25%       -1.000000      0.000000  
50%       -1.000000      0.000000  
75%       -1.000000      0.000000  
max      871.000000    275.000000  

📌
Forte présence de valeurs extrêmes → besoin de prudence en modélisation.

---

# 4.3 Histogrammes – Distributions

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

# 4.4 BOXPLOTS 

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

# 4.5 HEATMAP – Corrélations
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

