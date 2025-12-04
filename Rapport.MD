# 📘 Rapport d'Analyse - DATASET BANK MARKETING
1. INTRODUCTION
1.1 Contexte

Dans le cadre de campagnes marketing bancaires, les institutions financières cherchent à optimiser la probabilité de souscription à un produit (ici, un dépôt à terme) tout en limitant les coûts de prospection. L’essor de la Data Science permet aujourd’hui d’analyser de grandes bases de données clients afin de prédire avec plus de précision le comportement des prospects.

Le jeu de données utilisé provient de la base UCI Machine Learning Repository (Bank Marketing Dataset), et recense les caractéristiques socio-professionnelles et comportementales de milliers de clients contactés par des campagnes téléphoniques.

1.2 Problématique

Peut-on prédire efficacement si un client va souscrire à un dépôt à terme bancaire à partir de ses caractéristiques socio-économiques et historiques de contact ?

Cette problématique répond à un enjeu stratégique majeur :
✅ Grands volumes de clients
✅ Budget marketing limité
✅ Optimisation du ciblage commercial

1.3 Objectifs de l’étude

L’étude poursuit trois objectifs principaux :

Comprendre la structure des données (EDA)

Identifier les variables influençant la souscription

Construire un modèle prédictif fiable

Évaluer ses performances de manière rigoureuse

2. MÉTHODOLOGIE
2.1 Préparation et nettoyage des données

Les étapes de nettoyage ont été motivées par des préoccupations statistiques et pratiques :

Action	Justification
Suppression des variables inutiles	Éviter le bruit et les biais
Encodage des variables catégorielles	Les algorithmes nécessitent des données numériques
Gestion des valeurs aberrantes	Réduire l’influence des extrêmes
Conversion des durées en minutes	Meilleure interprétation

Des variables dérivées ont été ajoutées :

age_group : segmentation clients

total_contacts : pression commerciale cumulée

duration_min : temps d’appel simplifié

2.2 Choix de l’algorithme

Un modèle de classification supervisée a été adopté puisque la variable cible est binaire (souscription : oui / non).

Le modèle retenu :

▶ Logistic Regression (ou Random Forest selon ton modèle utilisé)

Justification :

Critère	Raison
Interprétabilité	Importance des variables
Rapidité	Excellente sur datasets moyens
Robustesse	Peu sensible au bruit
Standard en classification binaire	Référence académique
2.3 Méthode d’évaluation

Le dataset a été scindé :

70% entraînement

30% test

Les métriques retenues :

Accuracy

F1-score

ROC-AUC

Matrice de confusion

RMSE (sur les probabilités)

Ces indicateurs garantissent une évaluation équilibrée même en cas de déséquilibre des classes.

3. RÉSULTATS ET DISCUSSION
3.1 Résultats globaux du modèle (exemple)
Indicateur	Valeur
Accuracy	0.89
F1-Score	0.74
ROC-AUC	0.84
RMSE	0.31

⚠️ Remplace les valeurs selon celles que tu as obtenues en exécutant ton code.

3.2 Analyse de la matrice de confusion
	Prédit NON	Prédit OUI
Réel NON	Haute valeur – bon rejet	
Réel OUI	Erreurs marginales	
Faux positifs	Coût marketing inutile	
Faux négatifs	Manque à gagner	

Interprétation :

Le modèle identifie correctement la majorité des non-souscripteurs.

Certains clients intéressés sont mal classés → opportunités commerciales ratées.

L’équilibre rappel/précision reste acceptable.

3.3 Interprétation des variables clés

Les variables les plus influentes sont :

Variable	Impact
duration	Facteur décisif
previous	Historique client
campaign	Fatigue marketing
age	Ciblage démographique
pdays	Récence du contact

Exemples :

Un appel long augmente fortement la probabilité de souscription

Trop d’appels réduit la probabilité (harcèlement perçu)

Un client déjà contacté récemment est plus favorable

3.4 Discussion critique

✅ Points forts :

Bon niveau de généralisation

Facilement interprétable

Coût computationnel faible

⚠️ Limites :

Déséquilibre des classes

Certains clients atypiques mal prédits

La durée de l’appel n’est connue qu’après l’appel (biais d’utilisation)

4. CONCLUSION
4.1 Bilan

Le modèle développé permet de :

✅ Prédire avec un bon niveau de fiabilité
✅ Optimiser les campagnes commerciales
✅ Réduire les coûts
✅ Améliorer le taux de conversion

4.2 Limites

Dépendance forte à la variable durée

Modélisation statique

Hypothèses linéaires (si régression logistique utilisée)

Données anciennes (non temps réel)

4.3 Pistes d’amélioration

🚀 Perspectives :

Utiliser des modèles non linéaires (XGBoost, Neural Networks)

Intégrer des données externes (revenu, canal digital)

Rebalancer les classes (SMOTE)

Analyse causale

Déploiement en score temps réel

Segmentation client par clustering
