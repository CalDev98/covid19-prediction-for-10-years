# Analyse Prédictive du COVID-19 à Horizon 10 Ans

Pipeline Big Data avec **Apache Spark** et **Spark MLlib**, appliqué à la prédiction du nombre de nouveaux cas de COVID-19 au Maroc, avec une ouverture comparative sur **Prophet**.

> Projet réalisé dans le cadre du cours Big Data / Apache Spark — MLAIM, FSDM Fès.

---

## Sommaire

- [Contexte](#contexte)
- [Objectifs](#objectifs)
- [Données](#données)
- [Architecture du pipeline](#architecture-du-pipeline)
- [Modèles utilisés](#modèles-utilisés)
- [Résultats](#résultats)
- [Structure du dépôt](#structure-du-dépôt)
- [Installation et exécution](#installation-et-exécution)
- [Limites du projet](#limites-du-projet)
- [Pistes d'amélioration](#pistes-damélioration)
- [Équipe](#équipe)

---

## Contexte

Ce projet vise à prédire l'évolution du nombre de nouveaux cas de COVID-19 au Maroc sur un horizon de **10 ans**, à partir des données historiques disponibles (2020-2024), en s'appuyant sur des modèles de régression supervisée entraînés via **Spark MLlib**.

**Objectif pédagogique** : démontrer la maîtrise d'un pipeline Big Data complet (ingestion, nettoyage, feature engineering, modélisation, évaluation) avec Apache Spark — pas produire une prévision épidémiologique fiable à un tel horizon, ce que ne permettent de toute façon pas des modèles de régression classiques. Cette limite fait elle-même partie des résultats analysés dans le projet.

## Objectifs

- Mettre en œuvre un pipeline Big Data avec PySpark sur des données réelles.
- Réaliser un nettoyage et un feature engineering distribué (Spark SQL / DataFrame API).
- Entraîner et comparer plusieurs modèles de régression : `LinearRegression`, `RandomForestRegressor` (Spark MLlib), et **Prophet** en ouverture comparative.
- Évaluer objectivement les modèles (RMSE, R²).
- Produire une projection à 10 ans et en discuter les limites.

## Données

- **Source** : [Our World in Data — COVID-19 dataset](https://covid.ourworldindata.org/data/owid-covid-data.csv)
- **Zone géographique** : Maroc (`location = "Morocco"`)
- **Variable cible** : `new_cases`, agrégée sur base **hebdomadaire**
- **Période couverte** : 5 janvier 2020 → 4 août 2024
- **Volumétrie** : 1 674 lignes journalières brutes → 241 lignes hebdomadaires après agrégation

| Colonne | Description |
|---|---|
| `date` | Date du relevé |
| `location` | Pays (filtré sur `Morocco`) |
| `new_cases` | Nouveaux cas confirmés (cible) |
| `new_deaths` | Nouveaux décès |
| `total_cases` | Cumul des cas |
| `total_deaths` | Cumul des décès |

## Architecture du pipeline

```
Ingestion (CSV OWID)
      │
      ▼
Nettoyage (filtre Maroc, nulls, valeurs négatives, typage date)
      │
      ▼
Agrégation hebdomadaire (journalier → hebdomadaire)
      │
      ▼
Feature engineering (Window, lag, moyenne mobile, taux de croissance)
      │
      ▼
Pipeline Spark ML (VectorAssembler → StandardScaler → Modèle)
      │
      ▼
Évaluation (RMSE, R²) + Projection récursive à 10 ans
```

Développé en **PySpark**, exécuté sur **Google Colab**.

### Point méthodologique clé : rupture du rythme de reporting

À partir de 2022, le Maroc publie un seul chiffre hebdomadaire au lieu d'un chiffre quotidien (tous les autres jours affichent 0). Une feature calculée au niveau journalier devient alors extrêmement bruitée sur cette période. **Solution retenue** : ré-agrégation de toutes les données au niveau hebdomadaire (somme des cas par semaine calendaire).

### Point méthodologique clé : transformation logarithmique

Le dataset présente un changement d'échelle d'environ **1 pour 200** entre les vagues épidémiques (jusqu'à 65 000 cas/semaine) et la phase endémique (quelques dizaines de cas/semaine). Cette rupture déstabilise fortement l'apprentissage, en particulier pour la régression linéaire. **Solution retenue** : transformation `log1p(new_cases)` avant entraînement, reconversion via `expm1` sur les prédictions.

## Modèles utilisés

| Modèle | Bibliothèque | Rôle |
|---|---|---|
| `LinearRegression` (régularisée) | Spark MLlib | Modèle linéaire de référence |
| `RandomForestRegressor` (50 arbres) | Spark MLlib | Modèle d'ensemble non-linéaire |
| `Prophet` | Meta / hors Spark | Modèle spécialisé séries temporelles, ajouté en ouverture comparative |

Split **chronologique** 80/20 (jamais aléatoire, pour respecter la nature temporelle des données). Standardisation (`StandardScaler`) appliquée pour la régression linéaire uniquement.

## Résultats

Évaluation sur l'ensemble de test (~20 % des semaines les plus récentes, sept. 2023 – août 2024) :

| Modèle | RMSE | R² |
|---|---|---|
| Régression Linéaire (MLlib) | 309.23 | -12.45 |
| Random Forest (MLlib) | 70.66 | 0.30 |
| **Prophet** | **61.77** | **0.46** |

**Random Forest** est le meilleur modèle natif Spark MLlib. **Prophet**, testé à titre exploratoire hors périmètre du cours, obtient le meilleur score global et produit une projection à 10 ans plus cohérente (tendance prolongée + intervalle de confiance croissant), là où Random Forest plafonne rapidement — les modèles à base d'arbres ne pouvant pas extrapoler au-delà des valeurs vues à l'entraînement.

## Structure du dépôt

```
.
├── README.md
├── notebooks/
│   └── covid_prediction_spark.ipynb      # Notebook principal (Colab)
├── rapport/
│   └── Rapport_Projet_Covid_Spark.docx   # Rapport complet
├── presentation/
│   └── Presentation_Projet_Covid_Spark.pptx
└── data/
    └── owid-covid-data.csv               # (non versionné, à télécharger — voir Installation)
```

## Installation et exécution

Le projet est conçu pour être exécuté sur **Google Colab**.

1. Ouvrir le notebook `notebooks/covid_prediction_spark.ipynb` dans Google Colab.
2. Installer les dépendances :
   ```python
   !pip install pyspark -q
   !pip install prophet -q   # uniquement pour la section Prophet
   ```
3. Télécharger le dataset :
   ```python
   !wget -q https://covid.ourworldindata.org/data/owid-covid-data.csv -O owid-covid-data.csv
   ```
   *(ou monter un Google Drive contenant déjà le fichier — voir les instructions dans le notebook)*
4. Exécuter les cellules **dans l'ordre**, du début à la fin, dans un runtime fraîchement démarré.

## Limites du projet

- Les modèles à base d'arbres (Random Forest) ne peuvent pas extrapoler au-delà des valeurs numériques vues à l'entraînement — la projection à 10 ans plafonne après quelques mois.
- Aucun modèle testé ne modélise les mécanismes épidémiologiques réels (transmission, immunité) — contrairement aux modèles compartimentaux SIR/SEIR.
- Aucune prise en compte d'événements futurs imprévisibles (nouveaux variants, mesures sanitaires).
- Volume de données limité (4,5 ans, 241 points hebdomadaires) pour une projection à un horizon aussi long.
- Prophet est utilisé hors du périmètre strict de Spark MLlib demandé par le cours ; il est présenté comme une ouverture comparative, pas comme un livrable principal.

## Pistes d'amélioration

- Approfondir Prophet (saisonnalités multiples, régresseurs externes).
- Modèles épidémiologiques compartimentaux (SIR / SEIR).
- Approches d'apprentissage profond séquentiel (LSTM).
- `GeneralizedLinearRegression` avec distribution de Poisson (Spark MLlib), adaptée aux données de comptage.

---

*Source des données : [Our World in Data](https://ourworldindata.org/covid-cases)
