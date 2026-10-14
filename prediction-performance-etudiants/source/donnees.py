"""Génère un jeu de données synthétiques et prépare les ensembles pour les modèles."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .configuration import FEATURES, N_STUDENTS, PASS_THRESHOLD, SEED, TARGET, TEST_SIZE


def generate_data(n: int = N_STUDENTS, seed: int = SEED) -> pd.DataFrame:
    """Génère des observations synthétiques reproductibles à partir d’une formule connue.

    La note finale combine la note précédente (50 %), l’assiduité convertie sur 20 (25 %),
    les heures d’étude (25 %) et un bruit entier compris entre −3 et +3 points.
    """
    rng = np.random.default_rng(seed)

    heures = rng.integers(0, 11, n)
    assiduite = rng.integers(50, 101, n)
    precedente = rng.integers(0, 21, n)
    bruit = rng.integers(-3, 4, n)

    finale = np.rint(
        0.50 * precedente + 0.25 * (assiduite / 5) + 0.25 * (heures * 2) + bruit
    )

    return pd.DataFrame(
        {
            "heures_etude": heures,
            "assiduite": assiduite,
            "note_precedente": precedente,
            "note_finale": np.clip(finale, 0, 20).astype(int),
        }
    )


def split_data(df: pd.DataFrame):
    """Sépare les variables explicatives et les deux types de cibles en ensembles d’entraînement et de test alignés.

    Le même découpage permet de comparer les évaluations de la régression et de la classification.

    """
    X = df[FEATURES]
    y_reg = df[TARGET]
    y_clf = (y_reg >= PASS_THRESHOLD).astype(int)  # 1 = réussite, 0 = échec

    return train_test_split(
        X, y_reg, y_clf, test_size=TEST_SIZE, random_state=SEED, stratify=y_clf
    )
