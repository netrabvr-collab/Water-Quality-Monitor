"""
AI-Powered Water Quality Monitoring — Streamlit App

Combines:
- Phase 1: NDWI water body detection (Google Earth Engine, Sentinel-2)
- Phase 2: Turbidity prediction (Random Forest, trained on USGS matched data)
- Phase 3: Risk scoring (real-world thresholds + data-driven percentile)

Run with: streamlit run app.py
"""

import streamlit as st
import ee
import joblib
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# CONFIG — update this to your own Earth Engine project ID
# ---------------------------------------------------------------------------
EE_PROJECT_ID = "water-quality-monitor-509313"

# A few known stations from the USGS training data, for quick testing.
# Replace/extend with any locations you want.
KNOWN_LOCATIONS = {
    "Willamette River, OR (Jasper)": (43.99818166, -122.9059126),
    "Fox River, IL (McHenry)": (42.3386, -88.2334),
}

STATUS_COLORS = {"Clean": "green", "Moderate": "orange", "Polluted": "red"}


# ---------------------------------------------------------------------------
# CACHED SETUP — these run once per session, not on every interaction
# ---------------------------------------------------------------------------
@st.cache_resource
def init_earth_engine():
    """Initializes Earth Engine once per session."""
    ee.Initialize(project=EE_PROJECT_ID)
    return True


@st.cache_resource
def load_model():
    """Loads the trained turbidity model and its expected feature list."""
    model = joblib.load("turbidity_model.pkl")
    features = joblib.load("turbidity_features.pkl")
    return model, features


@st.cache_data
def load_training_reference():
    """
    Loads real turbidity training values, used only to compute the
    percentile-based risk score. Loading the full USGS CSVs every run
    would be slow, so this expects a pre-saved reference file.

    To create it once from your Phase 2 notebook:
        joblib.dump(df_turbidity["MeasurementValue"], "turbidity_reference.pkl")
    """
    return joblib.load("turbidity_reference.pkl")


# ---------------------------------------------------------------------------
# PIPELINE FUNCTIONS (Phases 1, 2, 3)
# ---------------------------------------------------------------------------
def get_water_mask(lat, lon, start_date, end_date, max_cloud_pct=10):
    """Phase 1: fetch the clearest available Sentinel-2 image and compute an NDWI water mask."""
    point = ee.Geometry.Point([lon, lat])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(str(start_date), str(end_date))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    image = collection.first()
    if image is None:
        raise ValueError("No suitable Sentinel-2 image found for this location/date range.")

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


def extract_reflectance_features(image, water_mask, region, scale=10):
    """
    Extracts mean, std, and median reflectance per Sentinel-2 band over the
    water-masked region, then builds the feature set the turbidity model expects.
    Note: 'center' features use the same water-region mean as 'buf250_mean',
    since we don't have USGS's exact per-station point extraction methodology.
    """
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12", "B8A"]
    masked_image = image.updateMask(water_mask).select(bands)

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

    band_name_map = {
        "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
        "B6": "B06", "B7": "B07", "B8": "B08", "B11": "B11", "B12": "B12", "B8A": "B8A",
    }

    features = {}
    for ee_band, model_band in band_name_map.items():
        mean_val = stats.get(f"{ee_band}_mean", 0) or 0
        std_val = stats.get(f"{ee_band}_stdDev", 0) or 0
        median_val = stats.get(f"{ee_band}_median", 0) or 0

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


def turbidity_risk_score(turbidity_fnu, training_data):
    """
    Phase 3: converts a raw turbidity prediction (FNU) into:
    - status: Clean/Moderate/Polluted, based on real-world turbidity standards
    - risk_score / percentile: this value's percentile rank within real training data
    """
    clean_max = 5
    moderate_max = 25

    if turbidity_fnu <= clean_max:
        status = "Clean"
    elif turbidity_fnu <= moderate_max:
        status = "Moderate"
    else:
        status = "Polluted"

    percentile = round(float((training_data < turbidity_fnu).mean() * 100), 1)

    return {
        "turbidity_fnu": turbidity_fnu,
        "status": status,
        "risk_score": percentile,
        "percentile": percentile,
        "percentile_description": f"Higher than {percentile}% of readings in the training data",
    }


def predict_water_quality(lat, lon, start_date, end_date, model, feature_cols, training_data):
    """Full pipeline: fetch image -> detect water -> predict turbidity -> compute risk score."""
    result = get_water_mask(lat, lon, start_date, end_date)
    region = result["point"].buffer(250)

    features = extract_reflectance_features(result["image"], result["water_mask"], region)
    X = pd.DataFrame([features])[feature_cols]

    pred_log = model.predict(X)[0]
    pred_turbidity = np.expm1(pred_log)

    risk = turbidity_risk_score(pred_turbidity, training_data)

    return {
        "date": result["date"],
        "cloud_pct": result["cloud_pct"],
        "lat": lat,
        "lon": lon,
        **risk,
    }


# ---------------------------------------------------------------------------
# MAP RENDERING
# ---------------------------------------------------------------------------
def build_map(result, name="Selected location"):
    color = STATUS_COLORS.get(result["status"], "gray")

    m = folium.Map(location=[result["lat"], result["lon"]], zoom_start=11)

    popup_text = f"""
    <b>{name}</b><br>
    <b>Status:</b> {result['status']}<br>
    <b>Turbidity:</b> {result['turbidity_fnu']:.1f} FNU<br>
    <b>Risk Score:</b> {result['risk_score']}/100<br>
    <b>Date:</b> {result['date']}<br>
    <b>Cloud %:</b> {result['cloud_pct']:.1f}%
    """

    folium.CircleMarker(
        location=[result["lat"], result["lon"]],
        radius=15,
        popup=folium.Popup(popup_text, max_width=250),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Water Quality Monitor", page_icon="💧", layout="wide")

st.title("💧 AI-Powered Water Quality Monitoring")
st.caption(
    "Detects water bodies from Sentinel-2 satellite imagery and predicts turbidity "
    "using a model trained on real USGS water-quality sensor data."
)

# --- Setup (runs once, cached) ---
try:
    init_earth_engine()
    model, feature_cols = load_model()
    training_data = load_training_reference()
    setup_ok = True
except Exception as e:
    setup_ok = False
    st.error(
        f"Setup failed: {e}\n\n"
        "Make sure turbidity_model.pkl, turbidity_features.pkl, and "
        "turbidity_reference.pkl are in this folder, and that your Earth "
        "Engine project ID is correct."
    )

if setup_ok:
    # --- Input controls ---
    col1, col2 = st.columns([1, 1])

    with col1:
        location_choice = st.selectbox(
            "Choose a location",
            options=list(KNOWN_LOCATIONS.keys()) + ["Custom coordinates"],
        )

        if location_choice == "Custom coordinates":
            lat = st.number_input("Latitude", value=43.99818166, format="%.6f")
            lon = st.number_input("Longitude", value=-122.9059126, format="%.6f")
        else:
            lat, lon = KNOWN_LOCATIONS[location_choice]
            st.write(f"Coordinates: {lat:.4f}, {lon:.4f}")

    with col2:
        default_end = date(2023, 9, 30)
        default_start = date(2023, 6, 1)
        date_range = st.date_input(
            "Date range (finds the clearest image in this window)",
            value=(default_start, default_end),
            max_value=date.today() - timedelta(days=5),
        )

    run_button = st.button("Run water quality assessment", type="primary")

    # --- Run pipeline and show results ---
    if run_button:
        if len(date_range) != 2:
            st.warning("Please select a full date range (start and end date).")
        else:
            start_date, end_date = date_range
            with st.spinner("Fetching satellite imagery and running the model..."):
                try:
                    result = predict_water_quality(
                        lat, lon, start_date, end_date, model, feature_cols, training_data
                    )
                    st.session_state["last_result"] = result
                    st.session_state["last_name"] = (
                        location_choice if location_choice != "Custom coordinates" else "Custom location"
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # --- Display last result, if any ---
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        name = st.session_state["last_name"]

        st.divider()
        st.subheader(f"Results — {name}")

        status = result["status"]
        color_emoji = {"Clean": "🟢", "Moderate": "🟠", "Polluted": "🔴"}.get(status, "⚪")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Status", f"{color_emoji} {status}")
        m2.metric("Turbidity", f"{result['turbidity_fnu']:.1f} FNU")
        m3.metric("Risk Score", f"{result['risk_score']}/100")
        m4.metric("Image Date", result["date"])

        st.caption(result["percentile_description"])
        st.caption(f"Cloud cover on satellite image: {result['cloud_pct']:.1f}%")

        st.subheader("Map")
        risk_map = build_map(result, name=name)
        st_folium(risk_map, width=900, height=500)

        with st.expander("About this prediction"):
            st.markdown(
                """
                - Water body detected via **NDWI** (Normalized Difference Water Index) on Sentinel-2 imagery.
                - Turbidity predicted via a **Random Forest** model (R²≈0.43) trained on 1.7M+ real USGS
                  sensor readings matched to Sentinel-2 reflectance, across 5 US river basins.
                - **Status** is based on real-world turbidity standards (<5 FNU: Clean, 5–25: Moderate, >25: Polluted).
                - **Risk score** is this reading's percentile rank within the real training data — i.e., what
                  percentage of historical readings were lower than this one.
                - Chlorophyll-a and CDOM were also investigated but did not generalize reliably in testing,
                  so this app reports turbidity only. See project documentation for details.
                """
            )
else:
    st.info("Fix the setup issue above, then reload the app.")