"""Entraînement et évaluation des modèles de régression, de classification et de regroupement."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .configuration import CV_FOLDS, FEATURES, N_CLUSTERS, PASS_THRESHOLD, SEED, TARGET


# ----------------------------------------------------------------- Régression
def run_regression(X_train, X_test, y_train, y_test):
    """Entraîne une régression linéaire et la compare à un modèle de référence qui prédit la moyenne."""
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, 20)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "baseline_mae": mean_absolute_error(y_test, baseline.predict(X_test)),
    }
    return model, y_pred, metrics


def regression_weights(model) -> pd.Series:
    """Renvoie les coefficients calculés sur les variables standardisées afin de les comparer."""
    coefs = model.named_steps["linearregression"].coef_
    return pd.Series(coefs, index=FEATURES).sort_values()


# ------------------------------------------------------------- Classification
def run_classification(X_train, X_test, y_train, y_test, X_all, y_all):
    """Entraîne une forêt aléatoire et la compare à un modèle de référence prédisant la classe majoritaire."""
    model = RandomForestClassifier(n_estimators=200, random_state=SEED)
    model.fit(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=SEED),
        X_all, y_all, cv=CV_FOLDS, scoring="accuracy",
    )
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, y_pred, metrics


# ----------------------------------------------------------------- Clustering
def run_clustering(df: pd.DataFrame):
    """Regroupe les variables standardisées et ordonne les groupes selon leur note finale moyenne.
    Les numéros de groupe sont ainsi plus faciles à interpréter."""
    X = StandardScaler().fit_transform(df[FEATURES])
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
    raw = kmeans.fit_predict(X)

    order = df.groupby(raw)[TARGET].mean().sort_values().index
    mapping = {old: new + 1 for new, old in enumerate(order)}

    out = df.copy()
    out["groupe"] = pd.Series(raw, index=df.index).map(mapping)

    profile = (
        out.assign(reussite=(out[TARGET] >= PASS_THRESHOLD).astype(int))
        .groupby("groupe")
        .agg(
            effectif=(TARGET, "size"),
            heures_etude=("heures_etude", "mean"),
            assiduite=("assiduite", "mean"),
            note_precedente=("note_precedente", "mean"),
            note_finale=(TARGET, "mean"),
            taux_reussite=("reussite", "mean"),
        )
        .round(2)
    )
    return out, profile, silhouette_score(X, raw)
