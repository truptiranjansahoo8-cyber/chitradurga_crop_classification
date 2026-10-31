"""Earth Engine helpers. Only lightweight requests: tile URLs, metadata and a random pixel sample."""
import json

import ee
import pandas as pd

from src.config import CLASS_ASSET, CLASS_BAND, CLASSES, CONF_ASSET, KEYS_DIR


def find_service_account_key():
    """Return the first JSON file in keys/ that looks like a service-account key."""
    for path in sorted(KEYS_DIR.glob("*.json")):
        try:
            info = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if info.get("type") == "service_account" and "client_email" in info:
            return path, info
    raise FileNotFoundError(f"No service-account JSON key found in {KEYS_DIR}")


def initialize(secret_info=None):
    """Authenticate with a service-account dict (e.g. from Streamlit secrets), else a key file in keys/."""
    if secret_info:
        info = dict(secret_info)
        creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
    else:
        path, info = find_service_account_key()
        creds = ee.ServiceAccountCredentials(info["client_email"], str(path))
    ee.Initialize(creds, project=info.get("project_id"))
    return info.get("project_id")


def asset_info(asset_id):
    """Band names and asset properties (excluding the footprint geometry)."""
    info = ee.Image(asset_id).getInfo()
    props = {k: v for k, v in info.get("properties", {}).items() if k != "system:footprint"}
    return {"bands": [b["id"] for b in info["bands"]], "properties": props}


def bounds(asset_id):
    """[[south, west], [north, east]] of the asset footprint."""
    coords = ee.Image(asset_id).geometry().bounds().coordinates().get(0).getInfo()
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def tile_url(asset_id, band, vis):
    """XYZ tile URL for a single band rendered with `vis` (no raster download)."""
    return ee.Image(asset_id).select(band).getMapId(vis)["tile_fetcher"].url_format


def outline_tile_url(fc_asset_id, color="#ff0000", width=2):
    """XYZ tile URL for a FeatureCollection drawn as a hollow outline (no fill)."""
    outline = ee.Image().byte().paint(ee.FeatureCollection(fc_asset_id), 1, width)
    return outline.getMapId({"palette": [color]})["tile_fetcher"].url_format


def sample_pixels(n_points=12000, seed=42):
    """Random native-resolution pixel sample of class + uncertainty bands within the crop mask.
    Points falling on masked (non-crop) pixels are dropped by sampleRegions."""
    cls = ee.Image(CLASS_ASSET).select(CLASS_BAND)
    conf = ee.Image(CONF_ASSET).select(["confidence", "margin", "entropy"])
    points = ee.FeatureCollection.randomPoints(cls.geometry(), n_points, seed)
    fc = cls.addBands(conf).sampleRegions(points, scale=cls.projection().nominalScale(), geometries=False)
    rows = [f["properties"] for f in fc.getInfo()["features"]]
    df = pd.DataFrame(rows).dropna()
    df["Crop"] = df[CLASS_BAND].astype(int).map(lambda c: CLASSES.get(c, ("Unknown",))[0])
    return df
