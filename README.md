# 🔍 Fraud Detection — Détection de fraude CB

Modèle de machine learning pour la détection de fraudes sur transactions par carte bancaire, basé sur **LightGBM** avec un pipeline sklearn complet.

---

## 📊 Résultats

| Métrique | Score |
|---|---|
| CV ROC-AUC | **0.966** |
| Recall (fraude) | 0.84 |
| Precision (fraude) | 0.13 |

> Le dataset est fortement déséquilibré (~0.58% de fraudes). Le modèle utilise `class_weight="balanced"` pour compenser.

---

## 🗂️ Structure du projet

```
fraud_detection/
├── data/
│   ├── raw/               ← Données brutes (non versionnées)
│   ├── interim/           ← Données intermédiaires
│   └── processed/         ← Données finales pour le modèle
│
├── fraud_detection/
│   ├── config.py          ← Chemins et variables de configuration
│   ├── dataset.py         ← Chargement et split des données
│   ├── features.py        ← Preprocessing (ColumnTransformer)
│   └── modeling/
│       ├── train.py       ← Entraînement + cross-validation
│       └── predict.py     ← Évaluation + seuil optimal
│
├── models/                ← Modèles sérialisés
├── notebooks/             ← Exploration et expérimentations
├── reports/figures/       ← Graphiques et visualisations
├── main.py                ← Point d'entrée principal
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/Sitylist94/fraud_detection.git
cd fraud_detection

# Créer et activer un environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

Placer les fichiers `train.csv` et `test.csv` dans `data/raw/`, puis :

```bash
python main.py
```

---

## 🛠️ Pipeline

```
CSV brut
  └─► ColumnTransformer
        ├─ StandardScaler      (colonnes numériques)
        └─ OneHotEncoder       (colonnes catégorielles)
              └─► LGBMClassifier (class_weight="balanced")
                    └─► Seuil optimal (courbe Precision-Recall)
                          └─► Prédictions
```

---

## 📦 Stack technique

- **Python 3.11**
- **LightGBM** — modèle de gradient boosting
- **scikit-learn** — pipeline, preprocessing, métriques
- **pandas / numpy** — manipulation des données
- **loguru** — logging

---

## 📁 Données

Les fichiers de données ne sont pas versionnés (voir `.gitignore`). Le dataset contient :

- **Train** : 1 296 675 transactions (dont ~7 500 fraudes)
- **Test** : 555 719 transactions
- **Target** : `is_fraud` (0 = légitime, 1 = fraude)