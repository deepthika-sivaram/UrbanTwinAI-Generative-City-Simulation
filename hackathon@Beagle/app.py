import streamlit as st
import folium
from streamlit_folium import st_folium
from models import uhi_raw, traffic_delay_raw, pm25_raw
from geo_utils import make_bbox, fetch_osm, grid_bbox, features
from models import apply_scenario, uhi_delta, traffic_delay_pct, pm25_delta
from viz import add_heat_layer
import pandas as pd                    
from ml_models import TrafficML 
from geo_utils import make_bbox, fetch_osm, grid_bbox, features, roads_with_grid_value
from viz import add_heat_layer, add_road_layer
from stgnn_pm import PMSTGNN
from geo_utils import build_rook_adjacency_from_grid

st.set_page_config(page_title="UrbanTwin AI", layout="wide")
st.title("UrbanTwin AI — Generative City Simulation")


@st.cache_resource
def load_models():
    m = {"traffic": None, "pm": None}
    try:
        m["traffic"] = TrafficML("artifacts/traffic_lgbm.pkl")
    except:
        m["traffic"] = None
    # Always create PM model; it will run with/without artifacts
    try:
        m["pm"] = PMSTGNN("artifacts/pm_stgnn.pt", "artifacts/pm_scaler.pkl")
    except:
        m["pm"] = PMSTGNN()  # no files; safe fallback
    return m

MODELS = load_models()
print("[MODEL] Traffic loaded?", MODELS["traffic"] is not None)
print("[MODEL] PM-STGNN loaded?", MODELS["pm"] is not None)

with st.sidebar:
    st.header("Area")
    with st.form("controls"):
        lat = st.number_input("Latitude", value=43.000000, format="%.6f")
        lon = st.number_input("Longitude", value=-78.790000, format="%.6f")
        km  = st.slider("Tile size (km)", 0.5, 2.0, 1.0, 0.5)
        submit = st.form_submit_button("Run Simulation")

    st.header("Scenario")
    add_b = st.slider("% buildings change", -50, 100, 30, 5)
    add_g = st.slider("% green change", -50, 100, 0, 5)

if submit:
    bbox = make_bbox(lat, lon, km)
    bld, roads, green, poly_m = fetch_osm(bbox)

    print(len(bld), len(roads), len(green))
    print(bld.head(3))
    print(roads.head(3))
    print(green.head(3))

    if len(bld) == 0 and len(green) == 0 and len(roads) == 0:
        st.warning("No OSM features in this tile. Try a different location/size.")
        st.stop()

    grid = grid_bbox(poly_m, cell=50)
    base = features(grid, bld, roads, green)
    # st.write(base[["building_cov","green_cov","road_den","impervious"]].describe())  # optional

    st.session_state["sim"] = {"lat": lat, "lon": lon, "base": base, "roads": roads}

# Render from cached base
if "sim" in st.session_state:
    sim  = st.session_state["sim"]
    base = sim["base"]
    lat  = sim["lat"]
    lon  = sim["lon"]
    roads = sim["roads"] 

    scenario = apply_scenario(base, add_b, add_g)

    # Debug prints (optional)
    print("mean_building_cov base/scen:",
          base["building_cov"].mean(), scenario["building_cov"].mean())
    print("mean_green_cov base/scen:",
          base["green_cov"].mean(), scenario["green_cov"].mean())
    print("Δ UHI (mean):",
          (uhi_raw(scenario).mean() - uhi_raw(base).mean()))
    print("Δ Delay (mean):",
          (traffic_delay_raw(scenario).mean() - traffic_delay_raw(base).mean()))
    print("Δ PM2.5 (mean):",
          (pm25_raw(scenario).mean() - pm25_raw(base).mean()))

    # Fields for map (centered for contrast)
    uhi = uhi_delta(scenario)              # keep heuristic for now
    pm  = pm25_delta(scenario)

    # --- Traffic map: per-cell delta ---
    # --- Traffic: ML if available, else heuristic ---
    # --- Traffic: ML if available, else heuristic ---
    if MODELS["traffic"] is not None:
        delay_base_abs = MODELS["traffic"].predict(base)
        delay_scen_abs = MODELS["traffic"].predict(scenario)
    else:
        delay_base_abs = traffic_delay_raw(base)
        delay_scen_abs = traffic_delay_raw(scenario)

    traffic_delta = pd.Series(delay_scen_abs - delay_base_abs, index=scenario.index)
    delay_kpi = float(traffic_delta.mean())
    traffic_level = pd.Series(delay_scen_abs - delay_scen_abs.mean(), index=scenario.index)

    scenario["traffic_level"] = traffic_level
    base["traffic_level"] = traffic_level
    base["uhi_raw"] = 6*base["impervious"] - 4*base["green_cov"]
    scenario["uhi_raw"] = 6*scenario["impervious"] - 4*scenario["green_cov"]

    adj = build_rook_adjacency_from_grid(scenario[["geometry"]])

    # ---------- PM: STGNN if available, else heuristic ----------
    if MODELS["pm"] is not None:
        pm_map, pm_delta_mean = MODELS["pm"].predict_map(
            base, scenario, adj, T=6,
            weather={"temp_c": 22.0, "wind_ms": 2.5, "rh": 55.0},
            traffic=traffic_level.values
        )
        pm_series = pd.Series(pm_map, index=scenario.index, dtype="float64")
        pm_kpi = float(pm_delta_mean)
    else:
        pm_series = pm25_delta(scenario).astype("float64")
        pm_kpi = float(pm25_raw(scenario).mean() - pm25_raw(base).mean())

    # ----- roads paint -----
    vis_level = (
        traffic_level
        * (1.0 + 0.7 * (base["road_den"] / (base["road_den"].quantile(0.95) + 1e-9)))
        * (1.0 + 0.3 * scenario["building_cov"])
    )

    # Paint roads using vis_level (not raw traffic_level)
    roads_colored = roads_with_grid_value(
        roads,
        scenario[["geometry"]],
        vis_level
    )

    # Ensure Folium CRS
    try:
        if getattr(roads_colored, "crs", None) is not None:
            roads_colored = roads_colored.to_crs(epsg=4326)
    except Exception:
        pass

    # Shared color limits for both heat & roads
    VMIN = float(vis_level.min())
    VMAX = float(vis_level.max())

    # per-cell UHI Δ for map
    uhi_map = pd.Series(uhi_raw(scenario) - uhi_raw(base), index=scenario.index, dtype="float64")

    # KPIs (final)
    uhi_kpi = float(uhi_raw(scenario).mean() - uhi_raw(base).mean())
    # pm_kpi already set above; DO NOT recompute here

    c1, c2, c3 = st.columns(3)
    c1.metric("Δ UHI (°C, mean)", f"{uhi_kpi:+.2f}")
    c2.metric("Δ Traffic delay (%)", f"{(delay_kpi*100):+.1f}%") 
    c3.metric("Δ PM2.5 (µg/m³)", f"{pm_kpi:+.2f}")

    layer_choice = st.radio(
        "Map layer",
        ["UHI Δ", "Traffic Δ", "Traffic level", "Traffic (roads)", "PM2.5"],  # <-- add this
        horizontal=True
    )

    m = folium.Map(location=[lat, lon], zoom_start=16, tiles="cartodbpositron",
                width="100%", height="100%")

    # --- Map visualization layers ---
    if layer_choice == "UHI Δ":
        add_heat_layer(
            m,
            scenario[["geometry"]],
            uhi_map,
            "UHI Δ (°C)",
            vmin=-1.0,
            vmax=1.0
        )

    elif layer_choice == "Traffic Δ":
        add_heat_layer(
            m,
            scenario[["geometry"]],
            traffic_delta,
            "Traffic Δ (fraction)",
            vmin=-0.3,
            vmax=0.3
        )

    elif layer_choice == "Traffic level":
        add_heat_layer(
            m,
            scenario[["geometry"]],
            vis_level,               # <--- reuse
            "Traffic level (roads-weighted)",
            vmin=VMIN, vmax=VMAX     # <--- reuse
        )

    elif layer_choice == "PM2.5":
        # 1) clip extremes to boost contrast
        lo, hi = pm_series.quantile(0.05), pm_series.quantile(0.95)
        pm_clip = pm_series.clip(lo, hi)

        # 2) normalize 0..1 for vivid coloring
        eps = 1e-9
        pm_norm = (pm_clip - pm_clip.min()) / (pm_clip.max() - pm_clip.min() + eps)

        # 3) render with tight vmin/vmax (0..1) so the colormap spans the panel
        add_heat_layer(
                m, scenario[["geometry"]], pm_norm,
                "PM2.5 (μg/m³, contrast-stretched)",
                vmin=0.0, vmax=1.0,
                fill_opacity=0.85, line_opacity=0.15
            )

    elif layer_choice == "Traffic (roads)":
        if len(roads_colored) == 0:
            st.info("No roads in this tile to render.")
        else:
            add_road_layer(
                m,
                roads_colored,
                "Traffic along roads",
                vmin=VMIN, vmax=VMAX, # <--- match the heatmap scale
                weight=4
            )

    st_folium(m, width=1200, height=600, key="map")
