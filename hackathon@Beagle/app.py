import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from models import uhi_raw, traffic_delay_raw, pm25_raw
from models import apply_scenario, uhi_delta, pm25_delta
from geo_utils import make_bbox, fetch_osm, grid_bbox, features
from geo_utils import roads_with_grid_value, build_rook_adjacency_from_grid
from viz import add_heat_layer, add_road_layer
from ml_models import TrafficML, UHIML
from stgnn_pm import PMSTGNN
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="UrbanTwin AI", layout="wide")
st.title("UrbanTwin AI — Generative City Simulation")


import json
import os
import textwrap

USE_VERTEX = os.getenv("USE_VERTEX", "1") == "1"
GOOGLE_CLOUD_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")

if USE_VERTEX:
    from vertexai import init as vertex_init
    from vertexai.generative_models import GenerativeModel, Part, SafetySetting

else:
    import google.generativeai as genai

@st.cache_resource
def load_models():
    m = {"traffic": None, "pm": None}

    try:
        m["traffic"] = TrafficML("artifacts/traffic_lgbm.pkl")
    except Exception as e:
        print("[Traffic] load failed:", e)
        m["traffic"] = None

    try:
        pm = PMSTGNN("artifacts/pm_stgnn.pt", "artifacts/pm_scaler.pkl")
    except Exception:
        pm = PMSTGNN()  
    m["pm"] = pm

    try:
        m["uhi"] = UHIML("artifacts/lst_model.pkl")
        print("[UHI] LST model loaded ✓")
    except Exception as e:
        print(f"[UHI] load failed: {e}")
        m["uhi"] = None

    return m

@st.cache_resource
def init_gemini(system_prompt: str):
    if USE_VERTEX:
        try:
            if not GOOGLE_CLOUD_PROJECT:
                raise RuntimeError("GOOGLE_CLOUD_PROJECT not set")
            vertex_init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION)
            return GenerativeModel("gemini-2.5-pro", system_instruction=system_prompt)
        except Exception as e:
            st.warning(f"Vertex init failed ({e}). Falling back to public API…")
           
    if not GEMINI_API_KEY:
        st.warning("No GEMINI_API_KEY and Vertex init failed; Gemini disabled.")
        return None
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-pro", system_instruction=system_prompt)



def summarize_for_llm(lat, lon, km, add_b, add_g, uhi_kpi, delay_kpi, pm_kpi, extras: dict):
    payload = {
        "location": {"lat": lat, "lon": lon, "tile_km": km},
        "scenario_inputs": {"pct_buildings": add_b, "pct_green": add_g},
        "kpis": {
            "delta_uhi_c": uhi_kpi,
            "delta_delay_frac": delay_kpi,
            "delta_pm25_ugm3": pm_kpi
        },
        "notes": {
            "traffic_model": "LGBM or heuristic depending on availability",
            "pm_model": "STGNN" if MODELS["pm"] and MODELS["pm"].is_ready() else "heuristic"
        },
        "stats": extras
    }

    system = textwrap.dedent("""
        You are an urban analytics assistant. Explain the likely impacts of the scenario deltas on heat,
        traffic delay, and PM2.5. Tie the reasoning to urban form:
        - More buildings -> more imperviousness/heat, potential traffic increase.
        - More green -> cooling (lower UHI), modest traffic effect, potential PM2.5 dilution/sink.
        Provide:
        1) A crisp 3-4 line executive summary.
        2) Metric-by-metric reasoning with thresholds (“meaningful” vs “minor”).
        3) 3 actionable nudges (e.g., where to add green, reduce road density impacts).
        Avoid generic fluff; keep it concrete, quantitative, and local to the tile.
    """).strip()

    user = "Analyze this scenario delta and explain what changed. JSON:\n" + json.dumps(payload, indent=2)
    return system, user


DEFAULT_SYSTEM = (
    "You are an urban analytics assistant. Explain the likely impacts..."
)


MODELS = load_models()
print("[MODEL] Traffic loaded?", MODELS["traffic"] is not None)
print("[MODEL] PM-STGNN loaded?", MODELS["pm"].is_ready())
print("[MODEL] UHI loaded?", MODELS.get("uhi") is not None)

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

# ---------------- Data pull & feature grid ----------------
if submit:

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            st.error("Latitude must be between -90 and 90, and longitude between -180 and 180.")
            st.stop()

    bbox = make_bbox(lat, lon, km)
    bld, roads, green, poly_m = fetch_osm(bbox)

    print(len(bld), len(roads), len(green))
    if len(bld) == 0 and len(green) == 0 and len(roads) == 0:
        st.warning("No OSM features in this tile. Try a different location/size.")
        st.stop()

    grid = grid_bbox(poly_m, cell=50)
    base = features(grid, bld, roads, green)

    st.session_state["sim"] = {"lat": lat, "lon": lon, "base": base, "roads": roads}

# ---------------- Simulation ----------------
if "sim" in st.session_state:
    sim   = st.session_state["sim"]
    base  = sim["base"]
    lat   = sim["lat"]
    lon   = sim["lon"]
    roads = sim["roads"]

    scenario = apply_scenario(base, add_b, add_g)

    # ---------- UHI: ML if available, else heuristic ----------
    def _adapt_for_uhi(df):
        """Adapter if your UHIML expects UHI_FEATS = ['building_count','greenery_percent'].
        If those columns are missing, derive them from what we already have.
        """
        use = df.copy()
        if {"building_count", "greenery_percent"}.issubset(use.columns):
            return use

        if {"building_cov", "green_cov"}.issubset(use.columns):
            use["building_count"] = (use["building_cov"] * 100.0).clip(0, 100)  # proxy count
            use["greenery_percent"] = (use["green_cov"] * 100.0).clip(0, 100)   # percent
            return use

        return use

    if MODELS.get("uhi") is not None:
        try:
            base_uhi_abs = MODELS["uhi"].predict(_adapt_for_uhi(base))
            scen_uhi_abs = MODELS["uhi"].predict(_adapt_for_uhi(scenario))
            uhi_delta_series = pd.Series(scen_uhi_abs - base_uhi_abs, index=scenario.index, dtype="float64")
            uhi_backend = "ML✓"
        except Exception as e:
            print(f"[UHI] ML inference failed ({e}); using heuristic.")
            uhi_delta_series = pd.Series(uhi_raw(scenario) - uhi_raw(base), index=scenario.index, dtype="float64")
            uhi_backend = "Heuristic"
    else:
        uhi_delta_series = pd.Series(uhi_raw(scenario) - uhi_raw(base), index=scenario.index, dtype="float64")
        uhi_backend = "Heuristic"

    print("mean_building_cov base/scen:", base["building_cov"].mean(), scenario["building_cov"].mean())
    print("mean_green_cov base/scen:",   base["green_cov"].mean(),   scenario["green_cov"].mean())
    print("Δ UHI (mean):",  (uhi_raw(scenario).mean()  - uhi_raw(base).mean()))
    print("Δ Delay (mean):",(traffic_delay_raw(scenario).mean() - traffic_delay_raw(base).mean()))
    print("Δ PM2.5 (mean):",(pm25_raw(scenario).mean() - pm25_raw(base).mean()))

    # --- Traffic: ML if available, else heuristic ---
    if MODELS["traffic"] is not None:
        delay_base_abs = MODELS["traffic"].predict(base)      
        delay_scen_abs = MODELS["traffic"].predict(scenario)
    else:
        delay_base_abs = traffic_delay_raw(base)
        delay_scen_abs = traffic_delay_raw(scenario)

    # 1) Delta for KPI
    traffic_delta = pd.Series(delay_scen_abs - delay_base_abs, index=scenario.index)
    delay_kpi = float(traffic_delta.mean())

    # 2) Level centered around mean (for visualization)
    traffic_level = pd.Series(delay_scen_abs - delay_scen_abs.mean(), index=scenario.index)

    # Enrich scenario/base for PM model
    scenario["traffic_level"] = traffic_level
    base["traffic_level"]     = traffic_level
    try:
        if MODELS.get("uhi") is not None:
            base["uhi_raw"] = pd.Series(base_uhi_abs, index=base.index, dtype="float64")
            scenario["uhi_raw"] = pd.Series(scen_uhi_abs, index=scenario.index, dtype="float64")
        else:
            raise RuntimeError("UHI ML not loaded")
    except Exception:
        base["uhi_raw"]     = 6 * base["impervious"]     - 4 * base["green_cov"]
        scenario["uhi_raw"] = 6 * scenario["impervious"] - 4 * scenario["green_cov"]

    # Spatial adjacency for STGNN
    adj = build_rook_adjacency_from_grid(scenario[["geometry"]])

    # ---------- PM: STGNN if available, else heuristic ----------
    if MODELS["pm"].is_ready():
        print("[PM] Using STGNN inference…")
        pm_map, pm_delta_mean = MODELS["pm"].predict_map(
            base, scenario, adj, T=6,
            weather={"temp_c": 22.0, "wind_ms": 2.5, "rh": 55.0},
            traffic=traffic_level.values
        )
        pm_series = pd.Series(pm_map, index=scenario.index, dtype="float64")
        pm_kpi = float(pm_delta_mean)
    else:
        print("[PM] Using heuristic fallback…")
        pm_series = pm25_delta(scenario).astype("float64")
        pm_kpi = float(pm25_raw(scenario).mean() - pm25_raw(base).mean())

    vis_level = (
        traffic_level
        * (1.0 + 0.7 * (base["road_den"] / (base["road_den"].quantile(0.95) + 1e-9)))
        * (1.0 + 0.3 * scenario["building_cov"])
    )

    roads_colored = roads_with_grid_value(
        roads,
        scenario[["geometry"]],
        vis_level
    )

    try:
        if getattr(roads_colored, "crs", None) is not None:
            roads_colored = roads_colored.to_crs(epsg=4326)
    except Exception:
        pass

    VMIN = float(vis_level.min())
    VMAX = float(vis_level.max())
    if abs(VMAX - VMIN) < 1e-6:
        VMIN -= 1e-3
        VMAX += 1e-3

    uhi_map = uhi_delta_series
    uhi_kpi = float(uhi_delta_series.mean())
    pm_label = "Δ PM2.5 (µg/m³) — " + ("STGNN✓" if MODELS["pm"].is_ready() else "Heuristic")

    c1, c2, c3 = st.columns(3)
    c1.metric("Δ UHI (°C, mean)", f"{uhi_kpi:+.2f}")
    c2.metric("Δ Traffic delay (%)", f"{(delay_kpi*100):+.1f}%")
    c3.metric(pm_label, f"{pm_kpi:+.2f}")

    # ---------------- Map ----------------
    layer_choice = st.radio(
        "Map layer",
        ["UHI Δ", "Traffic Δ", "Traffic level", "Traffic (roads)", "PM2.5"],
        horizontal=True
    )

    m = folium.Map(
        location=[lat, lon],
        zoom_start=16,
        tiles="cartodbpositron",
        width="100%",
        height="100%"
    )

    if layer_choice == "UHI Δ":
        add_heat_layer(
            m,
            scenario[["geometry"]],
            uhi_delta_series,
            f"UHI Δ (°C) — {uhi_backend}",
            vmin=-1.0, vmax=1.0
        )

    elif layer_choice == "Traffic Δ":
        add_heat_layer(
            m, scenario[["geometry"]],
            traffic_delta, "Traffic Δ (fraction)",
            vmin=-0.3, vmax=0.3
        )

    elif layer_choice == "Traffic level":
        add_heat_layer(
            m, scenario[["geometry"]],
            vis_level, "Traffic level (roads-weighted)",
            vmin=VMIN, vmax=VMAX
        )

    elif layer_choice == "PM2.5":
        pm_delta_series = pm_series

        q05, q95 = pm_delta_series.quantile(0.05), pm_delta_series.quantile(0.95)
        q = float(max(abs(q05), abs(q95), 1e-6))

        pm_norm_signed = (pm_delta_series.clip(-q, q) / q)

        add_heat_layer(
            m,
            scenario[["geometry"]],
            pm_norm_signed,                 
            "Δ PM2.5 (μg/m³) — signed",     
            vmin=-1.0, vmax=1.0,            
            fill_opacity=0.85, line_opacity=0.15,
        )

    elif layer_choice == "Traffic (roads)":
        if len(roads_colored) == 0:
            st.info("No roads in this tile to render.")
        else:
            add_road_layer(
                m, roads_colored, "Traffic along roads",
                vmin=VMIN, vmax=VMAX, 
                weight=4
            )

    st_folium(m, width=1200, height=600, key="map")


with st.expander("Explain changes with Gemini", expanded=False):
    st.caption("Get a concise, data-aware explanation of what changed and why.")
    sim_ready = ("sim" in st.session_state) and ('traffic_delta' in locals()) and ('uhi_map' in locals())

    if not sim_ready:
        st.info("Run a simulation first (choose area and click ‘Run Simulation’).")
    else:
        if st.button("Analyze scenario"):
            extras = {
                "uhi_delta_quantiles": uhi_map.quantile([0.05, 0.5, 0.95]).round(3).to_dict(),
                "traffic_delta_quantiles": traffic_delta.quantile([0.05, 0.5, 0.95]).round(3).to_dict(),
                "pm_quantiles": pm_series.quantile([0.05, 0.5, 0.95]).round(3).to_dict(),
            }

            system, user = summarize_for_llm(
                lat, lon, km, add_b, add_g,
                uhi_kpi, delay_kpi, pm_kpi, extras
            )

            model = init_gemini(system)
            if model is None:
                st.error("Gemini client not initialized. Check credentials.")
            else:
                try:
                    resp = model.generate_content(user)
                    st.markdown(resp.text)
                except Exception as e:
                    st.error(f"Gemini analysis failed: {e}")
