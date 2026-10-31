# Chitradurga Crop Classification — Kharif 2026

Streamlit dashboard for the Random Forest crop map of Chitradurga district (Arecanut, Coconut, Maize, Others, Ragi): interactive Earth Engine map, crop-area analytics, model performance and confidence analysis.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Put a Google Earth Engine **service-account JSON key** in `keys/` (any filename; the first `service_account` key found is used). The service account needs read access to the assets listed in `src/config.py`. `keys/` is git-ignored.

## Run

```bash
streamlit run main.py
```

## Deploy to Streamlit Community Cloud

Earth Engine credentials are read from Streamlit secrets first (`[gee_service_account]`), then from `keys/*.json`.

1. Push the repo to GitHub — `keys/` and `.streamlit/secrets.toml` are git-ignored and must stay out of the repo.
2. On share.streamlit.io create the app with main file `main.py`.
3. In **App settings → Secrets**, paste the contents of your local `.streamlit/secrets.toml`
   (format shown in `.streamlit/secrets.toml.example`).

To regenerate `secrets.toml` from a key file, copy each JSON field into the `[gee_service_account]` table
as a TOML string (`private_key` keeps its `\n` escapes).

## Test

```bash
python -m pytest -q
```

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Streamlit UI |
| `src/config.py` | Asset IDs, class names/colours, transcribed confusion matrices |
| `src/analytics.py` | Area loading/validation and metric computation (unit tested) |
| `src/gee.py` | Earth Engine auth, tile URLs, metadata and pixel sampling |
| `data/` | Class-wise area CSV and model statistics images |

## Notes

- Model metrics are recomputed from the confusion matrices transcribed from `data/*_stats.png`; tests verify they reproduce every printed value.
- Confidence statistics use a seeded random sample of native-resolution pixels, not the full raster. Confidence is the model's top-class probability, not accuracy.
- Areas are pixel-count areas from the classified map and are not bias-adjusted for classification error.
- geemap 0.38.9 has a `geemap.foliumap` import bug; `main.py` includes a small workaround.
