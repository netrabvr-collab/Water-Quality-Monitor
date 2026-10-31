"""
Water-quality pipeline (no Streamlit imports, so it can be tested in a notebook or script).

Phase 1: NDWI water detection (Google Earth Engine, Sentinel-2)
Phase 2: Turbidity prediction (Random Forest)
Phase 3: Risk scoring (real-world thresholds + percentile)
"""

import ee
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

EE_PROJECT_ID = "water-quality-monitor-509313"

CLEAN_MAX = 5       # FNU
MODERATE_MAX = 25   # FNU

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "B8A"]
BAND_NAME_MAP = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B11": "B11", "B12": "B12", "B8A": "B8A",
}


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def init_earth_engine(project_id=EE_PROJECT_ID):
    ee.Initialize(project=project_id)


def load_artifacts(
    model_path=None,
    features_path=None,
    reference_path=None,
):
    """Returns (model, feature_cols, training_reference)."""

    model_path = model_path or SCRIPT_DIR / "turbidity_model_gee.pkl"
    features_path = features_path or SCRIPT_DIR / "turbidity_features_gee.pkl"
    reference_path = reference_path or SCRIPT_DIR / "turbidity_reference.pkl"

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    reference = joblib.load(reference_path)

    return model, feature_cols, reference


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def get_water_mask(lat, lon, start_date, end_date, max_cloud_pct=10):
    """Fetch the clearest Sentinel-2 image in the window and compute an NDWI water mask."""
    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    # FIX: collection.first() never returns None on the client side, so the old
    # `if image is None` check could not fire. Check the collection size instead.
    if collection.size().getInfo() == 0:
        raise ValueError("No Sentinel-2 image under the cloud threshold for this location/date range.")

    image = collection.first()
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    water_mask = ndwi.gt(0).selfMask()

    return {
        "image": image,
        "ndwi": ndwi,
        "water_mask": water_mask,
        "point": point,
        "date": image.date().format("YYYY-MM-dd").getInfo(),
        "cloud_pct": image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo(),
    }


# ---------------------------------------------------------------------------
# Phase 2 (features)
# ---------------------------------------------------------------------------
def extract_reflectance_features(image, water_mask, region, scale=10):
    """
    Mean/std/median reflectance per band over the water-masked region, then the
    feature set the turbidity model expects. 'center' features reuse the
    region mean, as in the original notebook.
    """
    masked_image = image.updateMask(water_mask).select(BANDS)

    combined_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.median(), sharedInputs=True)
    )

    stats = masked_image.reduceRegion(
        reducer=combined_reducer,
        geometry=region,
        scale=scale,
        maxPixels=1e9,
    ).getInfo()

    # FIX: if no water pixels fall inside the buffer, every stat is None. The old
    # code turned those into 0 and silently produced a meaningless prediction.
    # For a multi-station map that would show a confident wrong colour, so fail loudly.
    if any(stats.get(f"{b}_mean") is None for b in ("B3", "B4", "B8")):
        raise ValueError("No water pixels found within 250 m of this point.")

    features = {}
    for ee_band, model_band in BAND_NAME_MAP.items():
        mean_val = stats.get(f"{ee_band}_mean") or 0
        std_val = stats.get(f"{ee_band}_stdDev") or 0
        median_val = stats.get(f"{ee_band}_median") or 0

        features[f"{model_band}_center"] = mean_val
        features[f"{model_band}_buf250_mean"] = mean_val
        features[f"{model_band}_buf250_std"] = std_val
        features[f"{model_band}_buf250_med"] = median_val

    b03 = features["B03_center"]
    b04 = features["B04_center"]
    b08 = features["B08_center"]

    features["ndti"] = (b04 - b03) / (b04 + b03) if (b04 + b03) != 0 else 0
    features["green_red_ratio"] = b03 / b04 if b04 != 0 else 0
    features["nir_red_ratio"] = b08 / b04 if b04 != 0 else 0

    return features


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
def turbidity_risk_score(turbidity_fnu, training_data):
    if turbidity_fnu <= CLEAN_MAX:
        status = "Clean"
    elif turbidity_fnu <= MODERATE_MAX:
        status = "Moderate"
    else:
        status = "Polluted"

    percentile = round(float((training_data < turbidity_fnu).mean() * 100), 1)

    return {
        "turbidity_fnu": float(turbidity_fnu),
        "status": status,
        "risk_score": percentile,
        "percentile": percentile,
        "percentile_description": f"Higher than {percentile}% of readings in the training data",
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def predict_water_quality(lat, lon, start_date, end_date, model, feature_cols, training_data):
    """Coordinates + date window in -> risk assessment out."""
    result = get_water_mask(lat, lon, start_date, end_date)
    region = result["point"].buffer(250)

    features = extract_reflectance_features(result["image"], result["water_mask"], region)
    X = pd.DataFrame([features])[feature_cols]

    pred_turbidity = float(np.expm1(model.predict(X)[0]))
    risk = turbidity_risk_score(pred_turbidity, training_data)

    return {
        "date": result["date"],
        "cloud_pct": result["cloud_pct"],
        "lat": lat,
        "lon": lon,
        **risk,
    }