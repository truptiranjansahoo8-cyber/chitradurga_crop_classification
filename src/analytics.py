"""Pure pandas/numpy data processing (no Streamlit or Earth Engine), so it can be unit tested."""
import numpy as np
import pandas as pd

from src.config import CLASSES

REQUIRED_COLS = {"Class", "Area_Hectares"}


def load_area(path) -> pd.DataFrame:
    """Load class-wise area CSV, validate it and add km² and percentage-share columns."""
    raw = pd.read_csv(path)
    missing = REQUIRED_COLS - set(raw.columns)
    if missing:
        raise ValueError(f"Area CSV is missing columns: {sorted(missing)}")

    df = pd.DataFrame({
        "Class": pd.to_numeric(raw["Class"], errors="coerce"),
        "Area_ha": pd.to_numeric(raw["Area_Hectares"], errors="coerce"),
    }).dropna()
    df["Class"] = df["Class"].astype(int)
    unknown = set(df["Class"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown class values in area CSV: {sorted(unknown)}")
    if (df["Area_ha"] < 0).any():
        raise ValueError("Negative area values in area CSV")

    # 1 km² = 100 ha; cross-check against the CSV's own km² column when present.
    df["Area_km2"] = df["Area_ha"] / 100
    if "Area_SqKm" in raw.columns and not np.allclose(df["Area_km2"], raw.loc[df.index, "Area_SqKm"], rtol=1e-6):
        raise ValueError("Area_Hectares and Area_SqKm columns are inconsistent")

    df["Crop"] = df["Class"].map(lambda c: CLASSES[c][0])
    df["Share_pct"] = 100 * df["Area_ha"] / df["Area_ha"].sum()
    return df[["Class", "Crop", "Area_ha", "Area_km2", "Share_pct"]].sort_values("Class").reset_index(drop=True)


def metrics_from_confusion(cm, labels) -> tuple[dict, pd.DataFrame]:
    """Overall accuracy, kappa, macro/weighted averages and per-class P/R/F1 from a confusion matrix
    (rows = actual, cols = predicted)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    tp = np.diag(cm)
    actual, predicted = cm.sum(axis=1), cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / predicted)
        recall = np.nan_to_num(tp / actual)
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    oa = tp.sum() / n
    pe = (actual * predicted).sum() / n**2
    per_class = pd.DataFrame({"Class": labels, "Precision": precision, "Recall": recall,
                              "F1": f1, "Support": actual.astype(int)})
    w = actual / n
    summary = {
        "Overall accuracy": oa,
        "Kappa": (oa - pe) / (1 - pe),
        "Macro precision": precision.mean(), "Macro recall": recall.mean(), "Macro F1": f1.mean(),
        "Weighted precision": (precision * w).sum(), "Weighted recall": (recall * w).sum(),
        "Weighted F1": (f1 * w).sum(),
        "Test samples": int(n),
    }
    return summary, per_class


def confidence_summary(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-class summary of sampled confidence values (expects columns Crop, confidence)."""
    g = samples.groupby("Crop")["confidence"]
    out = pd.DataFrame({
        "Pixels sampled": g.size(),
        "Mean": g.mean(),
        "Median": g.median(),
        "Share < 0.5 (%)": g.apply(lambda s: 100 * (s < 0.5).mean()),
    })
    return out.reset_index()
