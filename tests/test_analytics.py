"""Run: python -m pytest -q"""
import pandas as pd
import pytest

from src import config as C
from src.analytics import confidence_summary, load_area, metrics_from_confusion


def test_area_csv_units_and_shares():
    df = load_area(C.AREA_CSV)
    assert list(df["Crop"]) == C.CLASS_NAMES
    assert df["Share_pct"].sum() == pytest.approx(100)
    assert df["Area_ha"].sum() == pytest.approx(470382.45, abs=1)
    assert df.loc[df["Crop"] == "Maize", "Area_km2"].item() == pytest.approx(2916.040992)


def test_area_csv_rejects_bad_input(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Class,Foo\n1,2\n")
    with pytest.raises(ValueError):
        load_area(bad)
    bad.write_text("Class,Area_Hectares,Area_SqKm\n1,100,5\n")  # 100 ha != 5 km²
    with pytest.raises(ValueError):
        load_area(bad)


# Values printed in data/RandomForest_stats.png and data/Gredient_boosting_stats.png
REPORTED = {
    "Random Forest": dict(oa=0.7023, kappa=0.6111, macro=(0.657, 0.641, 0.630), weighted=(0.696, 0.702, 0.688),
                          precision=[0.840, 0.527, 0.653, 0.500, 0.766], recall=[0.823, 0.696, 0.716, 0.169, 0.801],
                          f1=[0.832, 0.600, 0.683, 0.253, 0.783], support=[147, 69, 176, 65, 151]),
    "Gradient Boosting": dict(oa=0.6859, kappa=0.5909, macro=(0.664, 0.637, 0.635), weighted=(0.694, 0.686, 0.681),
                              precision=[0.829, 0.484, 0.660, 0.613, 0.732], recall=[0.728, 0.652, 0.716, 0.292, 0.795],
                              f1=[0.775, 0.556, 0.687, 0.396, 0.762], support=[147, 69, 176, 65, 151]),
}


@pytest.mark.parametrize("model", list(REPORTED))
def test_confusion_matrix_reproduces_reported_metrics(model):
    r = REPORTED[model]
    s, pc = metrics_from_confusion(C.CONFUSION[model], C.CLASS_NAMES)
    assert s["Overall accuracy"] == pytest.approx(r["oa"], abs=5e-5)
    assert s["Kappa"] == pytest.approx(r["kappa"], abs=5e-5)
    assert (s["Macro precision"], s["Macro recall"], s["Macro F1"]) == pytest.approx(r["macro"], abs=5e-4)
    assert (s["Weighted precision"], s["Weighted recall"], s["Weighted F1"]) == pytest.approx(r["weighted"], abs=5e-4)
    assert list(pc["Precision"]) == pytest.approx(r["precision"], abs=5e-4)
    assert list(pc["Recall"]) == pytest.approx(r["recall"], abs=5e-4)
    assert list(pc["F1"]) == pytest.approx(r["f1"], abs=5e-4)
    assert list(pc["Support"]) == r["support"]


def test_confidence_summary():
    s = pd.DataFrame({"Crop": ["Maize", "Maize", "Ragi"], "confidence": [0.4, 0.8, 0.6]})
    out = confidence_summary(s).set_index("Crop")
    assert out.loc["Maize", "Mean"] == pytest.approx(0.6)
    assert out.loc["Maize", "Share < 0.5 (%)"] == pytest.approx(50)
    assert out.loc["Ragi", "Pixels sampled"] == 1
