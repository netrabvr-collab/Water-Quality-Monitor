import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------- 1. Load and combine ----------
df1 = pd.read_csv("../data/Matched_WQ_S2_1.csv")
df2 = pd.read_csv("../data/Matched_WQ_S2_2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# ---------- 2. Keep only valid (non-cloud-obscured) rows ----------
df_usable = df[df["l2flag_center"] == 1].copy()

# ---------- 3. Isolate turbidity rows ----------
df_turbidity = df_usable[df_usable["parm_nm"].str.contains("Turbidity", na=False)]

# ---------- 4. Station-based train/test split (prevents data leakage) ----------
stations = df_turbidity["station_nm"].unique().tolist()
train_stations, test_stations = train_test_split(stations, test_size=0.2, random_state=42)

train_df = df_turbidity[df_turbidity["station_nm"].isin(train_stations)]
test_df = df_turbidity[df_turbidity["station_nm"].isin(test_stations)]

# ---------- 5. Feature engineering ----------
def add_features(d):
    d = d.copy()
    d["ndti"] = (d["B04_center"] - d["B03_center"]) / (d["B04_center"] + d["B03_center"])
    d["green_red_ratio"] = d["B03_center"] / d["B04_center"].replace(0, np.nan)
    d["nir_red_ratio"] = d["B08_center"] / d["B04_center"].replace(0, np.nan)
    return d

train_df = add_features(train_df)
test_df = add_features(test_df)

band_cols = ["B01_center", "B02_center", "B03_center", "B04_center", "B05_center",
             "B06_center", "B07_center", "B08_center", "B11_center", "B12_center", "B8A_center"]
buf_cols = [c.replace("_center", "_buf250_mean") for c in band_cols]
ratio_cols = ["ndti", "green_red_ratio", "nir_red_ratio"]
feature_cols = band_cols + buf_cols + ratio_cols

X_train = train_df[feature_cols].fillna(0)
y_train = np.log1p(train_df["MeasurementValue"])   # log-transform: turbidity is heavily skewed

X_test = test_df[feature_cols].fillna(0)
y_test = test_df["MeasurementValue"]                # keep original scale for scoring

# ---------- 6. Train ----------
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ---------- 7. Evaluate ----------
y_pred = np.expm1(model.predict(X_test))   # convert predictions back to real FNU scale

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ---------- 8. Feature importance (for your write-up) ----------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances.head(10))

# ---------- 9. Save model for later use in the app ----------
joblib.dump(model, "turbidity_model.pkl")
joblib.dump(feature_cols, "turbidity_features.pkl")