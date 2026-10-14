"""Tests de base portant sur la validité des données, leur reproductibilité et le comportement des modèles."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.configuration import FEATURES, N_CLUSTERS, TARGET
from source.donnees import generate_data, split_data
from source.modeles import run_classification, run_clustering, run_regression


def test_data_shape_and_ranges():
    df = generate_data(n=200)
    assert len(df) == 200
    assert df[TARGET].between(0, 20).all()
    assert df["heures_etude"].between(0, 10).all()
    assert df["assiduite"].between(50, 100).all()


def test_data_is_reproducible():
    assert generate_data(seed=1).equals(generate_data(seed=1))


def test_models_beat_naive_baselines():
    df = generate_data()
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = split_data(df)

    _, _, reg = run_regression(X_tr, X_te, yr_tr, yr_te)
    assert reg["mae"] < reg["baseline_mae"]

    y_all = (df[TARGET] >= 10).astype(int)
    _, _, clf = run_classification(X_tr, X_te, yc_tr, yc_te, df[FEATURES], y_all)
    assert clf["accuracy"] > clf["baseline_accuracy"]


def test_clustering_returns_expected_groups():
    df_clustered, profile, _ = run_clustering(generate_data())
    assert set(df_clustered["groupe"]) == set(range(1, N_CLUSTERS + 1))
    # Les identifiants des groupes suivent l’ordre croissant de la note finale moyenne.
    assert profile["note_finale"].is_monotonic_increasing
