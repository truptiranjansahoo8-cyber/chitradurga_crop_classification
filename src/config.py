"""Central configuration: asset IDs, crop classes, paths and transcribed model statistics."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KEYS_DIR = ROOT / "keys"
AREA_CSV = DATA_DIR / "Chitradurga_2026_Classwise_Area.csv"
RF_STATS_IMG = DATA_DIR / "RandomForest_stats.png"
GB_STATS_IMG = DATA_DIR / "Gredient_boosting_stats.png"

PROJECT = "projects/cedar-channel-511705-h7/assets"
CLASS_ASSET = f"{PROJECT}/crop_classification_2026_Chitradurga"
CONF_ASSET = f"{PROJECT}/crop_classification_confidence_2026_Chitradurga"
AOI_ASSET = f"{PROJECT}/Chitradurga"  # district boundary FeatureCollection
CLASS_BAND = "crop_class"

STUDY_AREA = "Chitradurga District, Karnataka"
SEASON = "Kharif"
YEAR = 2026
MODEL = "Random Forest"

# class value -> (name, colour). Colours validated for colour-vision deficiency separation.
CLASSES = {
    1: ("Arecanut", "#1b5e20"),
    2: ("Coconut", "#2a78d6"),
    3: ("Maize", "#eda100"),
    4: ("Others", "#c4c4c4"),
    5: ("Ragi", "#d55181"),
}
CLASS_NAMES = [name for name, _ in CLASSES.values()]
CLASS_COLORS = {name: color for name, color in CLASSES.values()}

# Uncertainty bands in the confidence asset. With 5 classes the top-class probability is >= 0.2.
CONF_BANDS = {
    "confidence": dict(label="Confidence (top-class probability)", min=0.2, max=1.0, high_is_good=True),
    "margin": dict(label="Margin (top-1 minus top-2 probability)", min=0.0, max=1.0, high_is_good=True),
    "entropy": dict(label="Normalised entropy", min=0.0, max=1.0, high_is_good=False),
}
CONF_PALETTE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]  # single-hue sequential

# Transcribed verbatim from data/*_stats.png. Rows = actual, cols = predicted, order = CLASS_NAMES.
# All derived metrics (accuracy, kappa, P/R/F1) are recomputed from these matrices.
CONFUSION = {
    "Random Forest": [
        [121, 14, 10, 2, 0],
        [8, 48, 5, 0, 8],
        [3, 20, 126, 9, 18],
        [9, 3, 31, 11, 11],
        [3, 6, 21, 0, 121],
    ],
    "Gradient Boosting": [
        [107, 25, 7, 2, 6],
        [9, 45, 8, 0, 7],
        [2, 15, 126, 10, 23],
        [8, 1, 29, 19, 8],
        [3, 7, 21, 0, 120],
    ],
}
TRAINING = {"Fortnights": 21, "Features": 225, "Training pixels": 1322, "Test pixels": 608}
