"""Point d'entrée : génère les données, entraîne les modèles, produit les résultats.

Usage :
    python execution.py
"""

import os

# Évite un avertissement inoffensif de joblib sous Windows (détection des cœurs).
# Doit être défini avant l'import de scikit-learn.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import json
from pathlib import Path

import pandas as pd

from source.configuration import FEATURES, PASS_THRESHOLD, TARGET
from source.donnees import generate_data, split_data
from source.modeles import (
    regression_weights,
    run_classification,
    run_clustering,
    run_regression,
)
from source.visualisations import plot_clusters, plot_predictions, plot_weights

DOSSIER_RESULTATS = Path(__file__).resolve().parent / "resultats"


def title(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def main() -> None:
    title("PRÉDICTION DE LA PERFORMANCE DES ÉTUDIANTS")

    # 1. Générer les données et créer les ensembles d’entraînement et de test ------------------------------------------------------------
    df = generate_data()
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = split_data(df)
    print(f"{len(df)} étudiants synthétiques | "
          f"entraînement : {len(X_train)} | test : {len(X_test)}")
    print(f"Taux de réussite global (note >= {PASS_THRESHOLD}) : "
          f"{(df[TARGET] >= PASS_THRESHOLD).mean():.0%}")

    # 2. Prédire la note finale par régression ---------------------------------------------------------
    title("1. RÉGRESSION : prédire la note finale")
    reg_model, y_pred, reg = run_regression(X_train, X_test, y_reg_train, y_reg_test)
    print(f"MAE  : {reg['mae']:.2f} points   (modèle de référence : {reg['baseline_mae']:.2f})")
    print(f"RMSE : {reg['rmse']:.2f} points")
    print(f"R²   : {reg['r2']:.2f}")
    weights = regression_weights(reg_model)

    # 3. Prédire la réussite ou l’échec ----------------------------------------------------
    title("2. CLASSIFICATION : réussite ou échec")
    clf_model, _, clf = run_classification(
        X_train, X_test, y_clf_train, y_clf_test, df[FEATURES], (df[TARGET] >= PASS_THRESHOLD).astype(int)
    )
    print(f"Exactitude (test)       : {clf['accuracy']:.1%}   "
          f"(modèle de référence : {clf['baseline_accuracy']:.1%})")
    print(f"F1-score (test)       : {clf['f1']:.2f}")
    print(f"Exactitude (validation croisée, 5 plis)  : {clf['cv_accuracy_mean']:.1%} ± {clf['cv_accuracy_std']:.1%}")

    # 4. Explorer des groupes d’observations synthétiques similaires ---------------------------------------------------------
    title("3. REGROUPEMENT (K-MEANS) : profils d’étudiants")
    df_clustered, profile, silhouette = run_clustering(df)
    print(profile.to_string())
    print(f"\nScore de silhouette : {silhouette:.2f} "
          "(structure de groupes faible : les variables sont générées indépendamment)")

    # 5. Exporter les graphiques, les métriques et les profils des groupes ---------------------------------------------------------
    figures = DOSSIER_RESULTATS / "graphiques"
    plot_clusters(df_clustered, figures)
    plot_predictions(y_reg_test, y_pred, reg["mae"], figures)
    plot_weights(weights, figures)

    # Les noms exportés sont en français, même si les clés internes restent conventionnelles.
    correspondance_metriques = {
        "mae": "erreur_absolue_moyenne",
        "rmse": "racine_erreur_quadratique_moyenne",
        "r2": "coefficient_determination",
        "baseline_mae": "erreur_reference",
        "accuracy": "exactitude",
        "f1": "score_f1",
        "baseline_accuracy": "exactitude_reference",
        "cv_accuracy_mean": "exactitude_cv_moyenne",
        "cv_accuracy_std": "exactitude_cv_ecart_type",
        "confusion_matrix": "matrice_confusion",
    }
    mesures = {
        "regression": {correspondance_metriques.get(k, k): v for k, v in reg.items()},
        "classification": {correspondance_metriques.get(k, k): v for k, v in clf.items()},
        "score_silhouette": silhouette,
    }
    (DOSSIER_RESULTATS / "metriques.json").write_text(
        json.dumps(mesures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    profile.to_csv(DOSSIER_RESULTATS / "profils_groupes.csv")

    title("TERMINÉ")
    print(f"Résultats enregistrés dans : {DOSSIER_RESULTATS}")


if __name__ == "__main__":
    main()
