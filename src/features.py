"""
features.py

This module contains functions and classes related to feature extraction and processing.
"""

import re
import json
import datetime
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

pd.options.display.max_columns = 100
pd.options.display.max_rows = 60
pd.options.display.max_colwidth = 100
pd.options.display.precision = 10
pd.options.display.width = 160


def rename_columns(columns):
    """
    Renaming columns to remove special characters and spaces
    :param columns:
    :return: renamed columns
    """
    # en minuscule
    columns = [col.lower() for col in columns]
    # regex de remplacement
    rgxs = [
        (r"[°|/|']", "_"),
        (r"²", "2"),
        (r"[(|)]", ""),
        (r"é|è", "e"),
        (r"_+", "_"),
    ]
    # on remplace toutes les colonnes une par une
    for rgx in rgxs:
        columns = [re.sub(rgx[0], rgx[1], col) for col in columns]

    return columns


data = pd.read_csv("../data/dpe-v2-tertiaire-2.csv")

data.columns = rename_columns(data.columns)

ID_COL = "n_dpe"
TARGET = "etiquette_dpe"

data.dropna(subset=TARGET, inplace=True)

columns_categorical = [
    "periode_construction",
    "secteur_activite",
    "type_energie_principale_chauffage",
    "type_energie_n_1",
    "type_usage_energie_n_1",
]

for col in columns_categorical:
    data[col] = data[col].fillna("non renseigné")

type_energie_map = {
    "non renseigné": "non renseigné",
    "Électricité": "Électricité",
    "Électricité d'origine renouvelable utilisée dans le bâtiment": "Électricité",
    "Gaz naturel": "Gaz naturel",
    "Butane": "GPL",
    "Propane": "GPL",
    "GPL": "GPL",
    "Fioul domestique": "Fioul domestique",
    "Réseau de Chauffage urbain": "Réseau de Chauffage urbain",
    "Charbon": "Combustible fossile",
    "autre combustible fossile": "Combustible fossile",
    "Bois – Bûches": "Bois",
    "Bois – Plaquettes forestières": "Bois",
    "Bois – Granulés (pellets) ou briquettes": "Bois",
    "Bois – Plaquettes d’industrie": "Bois",
}

for col in [
    "type_energie_principale_chauffage",
    "type_energie_n_1",
]:
    data[col] = data[col].apply(lambda d: type_energie_map[d])

type_usage_map = {
    "non renseigné": "non renseigné",
    "périmètre de l'usage inconnu": "non renseigné",
    "Chauffage": "Chauffage",
    "Eau Chaude sanitaire": "Eau Chaude sanitaire",
    "Eclairage": "Eclairage",
    "Refroidissement": "Refroidissement",
    "Ascenseur(s)": "Ascenseur(s)",
    "auxiliaires et ventilation": "Refroidissement",
    "Autres usages": "Autres usages",
    "Bureautique": "Autres usages",
    "Abonnements": "Autres usages",
    "Production d'électricité à demeure": "Autres usages",
}


data["type_usage_energie_n_1"] = data["type_usage_energie_n_1"].apply(lambda d: type_usage_map[d])

periode_construction_map = {
    "avant 1948": "avant 1948",
    "1948-1974": "1948-1974",
    "1975-1977": "1975-1977",
    "1978-1982": "1978-1982",
    "1983-1988": "1983-1988",
    "1989-2000": "1989-2000",
    "2001-2005": "2001-2005",
    "2006-2012": "2006-2012",
    "2013-2021": "2013-2021",
    "après 2021": "après 2021",
}

encoder = OrdinalEncoder()

data[columns_categorical] = encoder.fit_transform(data[columns_categorical])
for col in columns_categorical:
    data[col] = data[col].astype(int)

mappings = {}
for i, col in enumerate(encoder.feature_names_in_):
    mappings[col] = {int(value): category for value, category in enumerate(encoder.categories_[i])}

with open("../data/categorical_mappings.json", "w", encoding="utf-8") as f:
    json.dump(mappings, f, ensure_ascii=False, indent=4)

columns_int = [
    "version_dpe",
    "surface_utile",
    "conso_kwhep_m2_an",
    "conso_e_finale_energie_n_1",
]

data = data[data["surface_utile"] < 9800]
data = data[data["conso_kwhep_m2_an"] < 2000]

for col in columns_int:
    data[col] = data[col].fillna(-1.0)
    data[col] = data[col].astype(int)

map_target = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9}
data[TARGET] = data[TARGET].apply(lambda d: map_target[d])

train_columns = columns_int + columns_categorical
features = [ID_COL] + train_columns + [TARGET]

data = data[features].copy()
data.reset_index(drop=True, inplace=True)

output_file = f"../data/dpe_processed_{datetime.datetime.now().strftime('%Y%m%d')}.csv"

data.to_csv(output_file, index=False)
print(f"data saved to {output_file}")
