"""Configuration centrale pour garantir la reproductibilité des expériences et harmoniser les libellés des variables."""

SEED = 42
N_STUDENTS = 500
TEST_SIZE = 0.20
PASS_THRESHOLD = 10  # Note minimale de réussite sur une échelle de 20 points
N_CLUSTERS = 3
CV_FOLDS = 5

FEATURES = ["heures_etude", "assiduite", "note_precedente"]
TARGET = "note_finale"

# Libellés lisibles utilisés dans les graphiques et dans le terminal
LABELS = {
    "heures_etude": "Heures d'étude",
    "assiduite": "Assiduité (%)",
    "note_precedente": "Note précédente (/20)",
    "note_finale": "Note finale (/20)",
}
