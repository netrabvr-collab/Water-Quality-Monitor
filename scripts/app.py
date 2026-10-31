"""
Water Quality Monitor: Streamlit app

Tabs:
  1. Single location
  2. Station map   (many stations on one map)
  3. Trend over time

Run with: streamlit run app.py
"""

from datetime import date, timedelta

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

import pipeline
from stations import STATIONS

STATUS_COLORS = {"Clean": "green", "Moderate": "orange", "Polluted": "red", "No data": "gray"}
STATUS_EMOJI = {"Clean": "🟢", "Moderate": "🟠", "Polluted": "🔴", "No data": "⚪"}
BADGE_STYLE = {
    "Clean": ("#1e8e3e", "#dcf5e2"),
    "Moderate": ("#b86100", "#fff0d9"),
    "Polluted": ("#c62828", "#fde2e2"),
    "No data": ("#5f6368", "#eceff1"),
}

st.set_page_config(page_title="Water Quality Monitor", page_icon="💧", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp {
    background: linear-gradient(180deg, #86d96f 0px, #bfeea9 320px, #f3faee 720px);
}
.block-container {padding-top: 2rem; max-width: 1200px;}

/* Hero card */
.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 32px;
    background: #ffffff; border-radius: 28px; padding: 44px 56px; margin-bottom: 28px;
    box-shadow: 0 24px 48px rgba(40, 110, 40, 0.18);
}
.eyebrow {color: #4caf50; font-weight: 700; font-size: 13px; letter-spacing: 2px; margin-bottom: 10px;}
.hero h1 {font-size: 54px; line-height: 1.05; font-weight: 800; color: #2b3035; margin: 0 0 14px 0; padding: 0;}
.hero p {color: #6b7280; font-size: 17px; max-width: 520px; margin: 0 0 20px 0;}
.chips span {
    display: inline-block; background: #eaf7e1; color: #2e7d32; font-size: 13px; font-weight: 600;
    padding: 6px 14px; border-radius: 999px; margin: 0 8px 8px 0;
}
.hero-art {
    flex: 0 0 210px; height: 210px; border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, #e3f9b8, #8fd14f 48%, #3f9a3a 100%);
    box-shadow: 0 0 0 14px rgba(134, 217, 111, 0.35), 0 0 0 30px rgba(134, 217, 111, 0.18);
    display: flex; align-items: center; justify-content: center;
}

/* Tabs */
button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.65); border-radius: 999px; padding: 8px 22px; margin-right: 8px;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {background: #ffffff; color: #2e7d32;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}
div[data-baseweb="tab-panel"] {
    background: #ffffff; border-radius: 24px; padding: 28px 32px; margin-top: 14px;
    box-shadow: 0 16px 36px rgba(40, 110, 40, 0.12);
}

/* Buttons */
.stButton > button {
    border-radius: 999px; padding: 10px 30px; font-weight: 700; border: none;
    background: #6cc24a; color: #ffffff;
}
.stButton > button:hover {background: #58b037; color: #ffffff;}

/* Result cards */
.stat {background: #f6fbf2; border: 1px solid #e0f0d6; border-radius: 18px; padding: 16px 20px;}
.stat-label {color: #6b7280; font-size: 13px; font-weight: 600; margin-bottom: 6px;}
.stat-value {color: #2b3035; font-size: 28px; font-weight: 800;}
.stat-value small {font-size: 14px; font-weight: 600; color: #6b7280;}
.badge {font-size: 18px; font-weight: 700; padding: 4px 14px; border-radius: 999px;}

.note {color: #6b7280; font-size: 13px; text-align: center; margin-top: 28px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached setup
# ---------------------------------------------------------------------------
@st.cache_resource
def setup():
    pipeline.init_earth_engine()
    return pipeline.load_artifacts()  # (model, feature_cols, training_reference)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def run_station(lat, lon, start, end):
    """Cached single prediction. Exceptions are not cached, so failures can be retried."""
    model, feature_cols, reference = setup()
    return pipeline.predict_water_quality(lat, lon, start, end, model, feature_cols, reference)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def badge(status):
    fg, bg = BADGE_STYLE.get(status, BADGE_STYLE["No data"])
    return f'<span class="badge" style="color:{fg};background:{bg}">● {status}</span>'


def stat_card(label, value_html):
    return (f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value_html}</div></div>')


def popup_html(name, r):
    if r.get("error"):
        return f"<b>{name}</b><br>No usable result:<br>{r['error']}"
    return (
        f"<b>{name}</b><br>"
        f"<b>Status:</b> {r['status']}<br>"
        f"<b>Turbidity:</b> {r['turbidity_fnu']:.1f} FNU<br>"
        f"<b>Risk score:</b> {r['risk_score']}/100<br>"
        f"<b>Image date:</b> {r['date']}<br>"
        f"<b>Cloud %:</b> {r['cloud_pct']:.1f}"
    )


LEGEND_HTML = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
            padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px; color: #222;">
<b>Turbidity status</b><br>
<span style="color:green">●</span> Clean (≤5 FNU)<br>
<span style="color:orange">●</span> Moderate (5–25)<br>
<span style="color:red">●</span> Polluted (&gt;25)<br>
<span style="color:gray">●</span> No usable data
</div>
"""


def build_map(rows, radius=14):
    """rows: list of dicts with name, lat, lon, status, plus result fields or 'error'."""
    m = folium.Map(location=[rows[0]["lat"], rows[0]["lon"]], zoom_start=9)
    for r in rows:
        color = STATUS_COLORS.get(r["status"], "gray")
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=radius,
            popup=folium.Popup(popup_html(r["name"], r), max_width=260),
            tooltip=r["name"],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(m)
    if len(rows) > 1:
        m.fit_bounds([[r["lat"], r["lon"]] for r in rows], padding=(30, 30))
    else:
        m.location = [rows[0]["lat"], rows[0]["lon"]]
        m.options["zoom"] = 11
    m.get_root().html.add_child(folium.Element(LEGEND_HTML))
    return m


def location_picker(key):
    """Returns (name, lat, lon)."""
    choice = st.selectbox("Location", list(STATIONS) + ["Custom coordinates"], key=f"loc_{key}")
    if choice == "Custom coordinates":
        lat = st.number_input("Latitude", value=43.99818166, format="%.6f", key=f"lat_{key}")
        lon = st.number_input("Longitude", value=-122.9059126, format="%.6f", key=f"lon_{key}")
        return "Custom location", lat, lon
    lat, lon = STATIONS[choice]
    st.caption(f"Coordinates: {lat:.4f}, {lon:.4f}")
    return choice, lat, lon


def period_windows(n_periods, months_per_period):
    """Consecutive (start, end) date windows ending just before today (oldest first)."""
    end = pd.Timestamp(date.today() - timedelta(days=5))
    edges = [end - pd.DateOffset(months=months_per_period * i) for i in range(n_periods, -1, -1)]
    return [(a.date(), b.date()) for a, b in zip(edges[:-1], edges[1:])]


# ---------------------------------------------------------------------------
# Hero + setup
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div>
    <div class="eyebrow">SATELLITE · MACHINE LEARNING</div>
    <h1>Water Quality<br>Monitor</h1>
    <p>Estimate river turbidity from Sentinel-2 satellite imagery, score the risk,
       and see how it changes over time.</p>
    <div class="chips">
      <span>Sentinel-2 imagery</span>
      <span>Google Earth Engine</span>
      <span>Risk scoring</span>
    </div>
  </div>
  <div class="hero-art">
    <svg width="96" height="96" viewBox="0 0 24 24" fill="#ffffff">
      <path d="M12 2.5C12 2.5 5 10.2 5 14.6a7 7 0 0 0 14 0C19 10.2 12 2.5 12 2.5z"/>
    </svg>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    setup()
except Exception as e:
    st.error(
        f"Setup failed: {e}\n\nCheck that turbidity_model_gee.pkl, turbidity_features_gee.pkl and "
        "turbidity_reference.pkl are in this folder and that Earth Engine is authenticated."
    )
    st.stop()

tab_single, tab_map, tab_trend = st.tabs(["Single location", "Station map", "Trend over time"])


# ---------------------------------------------------------------------------
# Tab 1: single location
# ---------------------------------------------------------------------------
with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        name, lat, lon = location_picker("single")
    with c2:
        date_range = st.date_input(
            "Date range (uses the clearest image in this window)",
            value=(date(2023, 6, 1), date(2023, 9, 30)),
            max_value=date.today() - timedelta(days=5),
            key="dr_single",
        )

    if st.button("Run assessment", type="primary", key="run_single"):
        if len(date_range) != 2:
            st.warning("Select both a start and an end date.")
        else:
            with st.spinner("Fetching imagery and running the model..."):
                try:
                    res = run_station(lat, lon, *date_range)
                    st.session_state["single"] = {"name": name, "result": res}
                except Exception as e:
                    st.session_state.pop("single", None)
                    st.error(f"Prediction failed: {e}")

    if "single" in st.session_state:
        name, res = st.session_state["single"].values()
        st.markdown(f"### Results: {name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(stat_card("Status", badge(res["status"])), unsafe_allow_html=True)
        m2.markdown(stat_card("Turbidity", f"{res['turbidity_fnu']:.1f} <small>FNU</small>"),
                    unsafe_allow_html=True)
        m3.markdown(stat_card("Risk score", f"{res['risk_score']}<small>/100</small>"),
                    unsafe_allow_html=True)
        m4.markdown(stat_card("Image date", res["date"]), unsafe_allow_html=True)
        st.caption(res["percentile_description"])
        st_folium(build_map([{"name": name, **res}], radius=15), width=900, height=450,
                  returned_objects=[], key="map_single")


# ---------------------------------------------------------------------------
# Tab 2: station map
# ---------------------------------------------------------------------------
with tab_map:
    st.write("Run the model on several stations and see them together on one map.")
    c1, c2 = st.columns(2)
    with c1:
        chosen = st.multiselect("Stations", list(STATIONS), default=list(STATIONS)[:8], key="net_sel")
    with c2:
        net_range = st.date_input(
            "Date range (clearest image per station)",
            value=(date(2023, 6, 1), date(2023, 9, 30)),
            max_value=date.today() - timedelta(days=5),
            key="dr_net",
        )

    if st.button("Run all stations", type="primary", key="run_net"):
        if not chosen:
            st.warning("Pick at least one station.")
        elif len(net_range) != 2:
            st.warning("Select both a start and an end date.")
        else:
            rows = []
            bar = st.progress(0.0, text="Starting...")
            for i, st_name in enumerate(chosen):
                lat, lon = STATIONS[st_name]
                try:
                    res = run_station(lat, lon, *net_range)
                    rows.append({"name": st_name, "error": None, **res})
                except Exception as e:
                    rows.append({"name": st_name, "lat": lat, "lon": lon,
                                 "status": "No data", "error": str(e)})
                bar.progress((i + 1) / len(chosen), text=f"{st_name} ({i + 1}/{len(chosen)})")
            bar.empty()
            st.session_state["network"] = rows

    if "network" in st.session_state:
        rows = st.session_state["network"]
        ok = [r for r in rows if not r["error"]]
        st.caption(f"{len(ok)} of {len(rows)} stations returned a usable estimate.")
        st_folium(build_map(rows), width=1000, height=520, returned_objects=[], key="map_net")

        table = pd.DataFrame([
            {
                "Station": r["name"],
                "Status": f"{STATUS_EMOJI[r['status']]} {r['status']}",
                "Turbidity (FNU)": None if r["error"] else round(r["turbidity_fnu"], 1),
                "Risk score": None if r["error"] else r["risk_score"],
                "Image date": None if r["error"] else r["date"],
                "Note": r["error"] or "",
            }
            for r in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3: trend
# ---------------------------------------------------------------------------
with tab_trend:
    st.write("Estimate how a location's risk has changed over past periods.")
    c1, c2, c3 = st.columns(3)
    with c1:
        t_name, t_lat, t_lon = location_picker("trend")
    with c2:
        period_label = st.radio("Period length", ["Quarterly", "Monthly"], horizontal=True)
        months = 3 if period_label == "Quarterly" else 1
    with c3:
        n_periods = st.slider("Number of periods", 4, 12 if months == 3 else 24, 8)

    if st.button("Run trend", type="primary", key="run_trend"):
        windows = period_windows(n_periods, months)
        recs = []
        bar = st.progress(0.0, text="Starting...")
        for i, (a, b) in enumerate(windows):
            mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
            try:
                res = run_station(t_lat, t_lon, a, b)
                recs.append({"window": f"{a} → {b}", "x": pd.Timestamp(res["date"]),
                             "turbidity_fnu": res["turbidity_fnu"], "risk_score": res["risk_score"],
                             "status": res["status"], "note": ""})
            except Exception as e:
                recs.append({"window": f"{a} → {b}", "x": mid, "turbidity_fnu": None,
                             "risk_score": None, "status": "No data", "note": str(e)})
            bar.progress((i + 1) / len(windows), text=f"{a} → {b}")
        bar.empty()
        st.session_state["trend"] = {"name": t_name, "df": pd.DataFrame(recs)}

    if "trend" in st.session_state:
        t_name, df = st.session_state["trend"].values()
        metric = st.radio("Plot", ["Risk score (percentile)", "Turbidity (FNU)"], horizontal=True)
        ycol = "risk_score" if metric.startswith("Risk") else "turbidity_fnu"

        df = df.sort_values("x")
        valid = df.dropna(subset=[ycol])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["x"], y=df[ycol], mode="lines", line=dict(color="#9ccc65", width=2),
            connectgaps=False, hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=valid["x"], y=valid[ycol], mode="markers",
            marker=dict(size=12, color=[STATUS_COLORS[s] for s in valid["status"]],
                        line=dict(width=1, color="white")),
            text=valid["window"], showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<br>window: %{text}<extra></extra>",
        ))
        if ycol == "turbidity_fnu":
            fig.add_hline(y=pipeline.CLEAN_MAX, line_dash="dot", line_color="green")
            fig.add_hline(y=pipeline.MODERATE_MAX, line_dash="dot", line_color="red")
        fig.update_layout(
            title=f"{t_name}: {metric}",
            xaxis_title="Satellite image date",
            yaxis_title=metric,
            height=420,
            plot_bgcolor="#f6fbf2",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"{len(valid)} of {len(df)} periods had a usable image. Each point is a single "
            "cloud-filtered scene, so look at the overall direction rather than small differences; "
            "seasonal effects (e.g. spring runoff) can look like trends."
        )
        with st.expander("Data table"):
            st.dataframe(df.rename(columns={"x": "image date"}), hide_index=True,
                         use_container_width=True)


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

st.markdown('<div class="note">Built with Python, Google Earth Engine and Streamlit</div>',
            unsafe_allow_html=True)