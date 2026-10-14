"""Crée et enregistre les visualisations du projet au format PNG."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .configuration import LABELS, PASS_THRESHOLD, SEED, TARGET

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_clusters(df_clustered, out_dir: Path) -> Path:
    """Représente les notes finales en fonction des heures d’étude, avec une couleur par groupe."""
    rng = np.random.default_rng(SEED)
    data = df_clustered.copy()
    # Décale légèrement les points entiers pour rendre visibles les observations superposées.
    data["x"] = data["heures_etude"] + rng.uniform(-0.18, 0.18, len(data))
    data["y"] = data[TARGET] + rng.uniform(-0.18, 0.18, len(data))
    data["Groupe"] = "Groupe " + data["groupe"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=data, x="x", y="y", hue="Groupe",
        hue_order=["Groupe 1", "Groupe 2", "Groupe 3"],
        palette="Set2", s=45, alpha=0.7, edgecolor="white", linewidth=0.4, ax=ax,
    )
    ax.axhline(PASS_THRESHOLD, color="crimson", ls="--", lw=1.4,
               label=f"Seuil de réussite ({PASS_THRESHOLD}/20)")
    ax.set(title="Profils d'étudiants (K-Means)", xlabel=LABELS["heures_etude"],
           ylabel=LABELS[TARGET], xlim=(-0.5, 10.5), ylim=(-0.6, 20.6))
    ax.set_xticks(range(0, 11))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.text(0.5, -0.02, "Points légèrement décalés pour la lisibilité (données entières).",
             ha="center", fontsize=8, color="dimgray")
    return _save(fig, out_dir / "profils_etudiants.png")


def plot_predictions(y_true, y_pred, mae: float, out_dir: Path) -> Path:
    """Compare les notes prédites aux notes réelles de l’ensemble de test."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=35, color="#3b7dd8", edgecolor="white", lw=0.4)
    ax.plot([0, 20], [0, 20], color="crimson", ls="--", lw=1.3, label="Prédiction parfaite")
    ax.set(title=f"Régression : prédit vs réel (MAE = {mae:.2f} pt)",
           xlabel="Note réelle (/20)", ylabel="Note prédite (/20)",
           xlim=(0, 20), ylim=(0, 20))
    ax.legend(loc="upper left")
    return _save(fig, out_dir / "notes_predites_reelles.png")


def plot_weights(weights, out_dir: Path) -> Path:
    """Visualise les coefficients de régression associés à chaque variable."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [LABELS[name] for name in weights.index]
    ax.barh(labels, weights.values, color="#3b7dd8")
    ax.set(title="Influence des variables sur la note finale",
           xlabel="Points de note par écart-type de la variable")
    for i, v in enumerate(weights.values):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, weights.max() * 1.15)
    return _save(fig, out_dir / "influence_variables.png")
