# =============================================================
# NAIP Dashboard — National Agricultural Intelligence Platform
# =============================================================
# Design language: "Earth & Grain"
#   Background : warm paper (#FAF6EE)
#   Ink        : near-black brown (#2B2417)
#   Wheat gold : #C9A227  (primary accent — predictions, highlights)
#   Soil brown : #8B5A2B  (secondary — structure, high yield)
#   Irrigation green : #3D6B5C (positive / healthy yield)
#   Alert terracotta : #A33B2E (drought / flood / risk)
#
# Three modes:
#   1. Historical Results — map + click-through of the locked
#      2018-19 test set predictions (validated against ground truth)
#   2. Current Season Outlook — REAL 2026 weather (NASA POWER,
#      live-fetched) combined with last-known (2019) yield baseline.
#      Honestly labeled: weather is live, yield trend is not.
#   3. Scenario Explorer — pick any historical district/crop/year,
#      apply manual what-if rainfall/temperature sliders.
#
# All three modes share a rule-based Advisory panel (NOT model-
# generated) that translates SHAP/RAI/interval outputs into
# process-level actions an officer already has authority over.
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import pickle
import shap
import os

# ---------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------

st.set_page_config(
    page_title="NAIP — Agricultural Intelligence Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# DESIGN SYSTEM — injected CSS
# ---------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --paper:      #FAF6EE;
    --paper-deep: #F1EADB;
    --ink:        #2B2417;
    --ink-soft:   #564B38;
    --wheat:      #C9A227;
    --wheat-dark: #9C7E1A;
    --wheat-pale: #F4E9C4;
    --soil:       #8B5A2B;
    --soil-light: #D9C4A8;
    --green:      #3D6B5C;
    --green-bg:   #E7EFEA;
    --alert:      #A33B2E;
    --alert-bg:   #F5E4DF;
    --line:       #DDD2BC;
    --shadow:     0 1px 2px rgba(43,36,23,0.04), 0 2px 8px rgba(43,36,23,0.04);
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink);
}
.stApp { background-color: var(--paper); }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }

/* ---- Masthead ---- */
.naip-masthead {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 6px 2px 20px 2px;
    border-bottom: 3px solid var(--ink);
    margin-bottom: 4px;
}
.naip-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.7rem;
    letter-spacing: -0.015em;
    color: var(--ink);
    line-height: 1;
}
.naip-title span { color: var(--wheat-dark); font-weight: 500; }
.naip-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--soil);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 6px;
}
.naip-masthead-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
    text-align: right;
    line-height: 1.5;
    padding-bottom: 4px;
}

/* ---- Eyebrows + section labels ---- */
.naip-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: var(--wheat-dark);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.naip-eyebrow::before {
    content: "";
    display: inline-block;
    width: 14px;
    height: 2px;
    background: var(--wheat-dark);
}

/* ---- Cards / stats ---- */
.naip-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
}
.naip-stat-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.05rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.1;
}
.naip-stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--soil);
    margin-top: 3px;
    font-weight: 500;
}
.naip-stat-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: var(--green);
    margin-top: 7px;
}

/* ---- Status boxes ---- */
.naip-alert, .naip-safe, .naip-scenario {
    padding: 14px 18px;
    border-radius: 4px;
    margin-bottom: 14px;
    font-size: 0.91rem;
    line-height: 1.45;
}
.naip-alert    { background: var(--alert-bg);  border-left: 4px solid var(--alert); }
.naip-safe     { background: var(--green-bg);  border-left: 4px solid var(--green); }
.naip-scenario { background: #FBF3E0;           border-left: 4px solid var(--wheat-dark); }
.naip-alert-title, .naip-safe-title, .naip-scenario-title {
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    font-size: 0.74rem;
    letter-spacing: 0.07em;
    margin-bottom: 5px;
    display: block;
}
.naip-alert-title    { color: var(--alert); }
.naip-safe-title     { color: var(--green); }
.naip-scenario-title { color: var(--wheat-dark); }

/* ---- SHAP bars ---- */
.shap-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}
.shap-label { width: 40%; color: var(--ink); padding-right: 10px; line-height: 1.3; }
.shap-bar-track {
    flex-grow: 1;
    height: 16px;
    background: var(--soil-light);
    border-radius: 3px;
    overflow: hidden;
}
.shap-bar-fill-pos { height: 100%; background: var(--green); border-radius: 3px; }
.shap-bar-fill-neg { height: 100%; background: var(--alert); border-radius: 3px; }
.shap-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    width: 68px;
    text-align: right;
    color: var(--soil);
    font-weight: 600;
}

/* ---- Dividers + headers ---- */
.naip-divider { border-top: 1px solid var(--line); margin: 24px 0 20px 0; }
.naip-panel-district {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.7rem;
    color: var(--ink);
    line-height: 1.15;
}
.naip-panel-state {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--soil);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 600;
}
.naip-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: var(--ink-soft);
    line-height: 1.6;
}

/* ---- Map legend ---- */
.naip-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 10px 14px;
    background: var(--paper-deep);
    border: 1px solid var(--line);
    border-radius: 4px;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--ink-soft);
}
.naip-legend-item { display: flex; align-items: center; gap: 6px; }
.naip-legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; border: 1px solid rgba(43,36,23,0.25); }

/* ---- Empty state ---- */
.naip-empty {
    text-align: center;
    padding: 56px 24px;
    background: #FFFFFF;
    border: 1px dashed var(--line);
    border-radius: 6px;
}
.naip-empty-icon { font-size: 2rem; margin-bottom: 12px; opacity: 0.7; }
.naip-empty-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.92rem;
    color: var(--soil);
    line-height: 1.5;
}

/* ---- Mode intro banner ---- */
.naip-mode-banner {
    font-family: 'Inter', sans-serif;
    font-size: 0.88rem;
    color: var(--ink-soft);
    line-height: 1.55;
    padding: 13px 16px;
    background: var(--paper-deep);
    border-radius: 4px;
    margin-bottom: 16px;
}

/* ---- Streamlit chrome cleanup ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stRadio"] > label { display: none; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# FRIENDLY FEATURE NAMES
# ---------------------------------------------------------------

FRIENDLY_NAMES = {
    'lag_yield_1': "Last year's yield",
    'rolling_yield_3yr': "3-year average yield",
    'time_trend': "Long-term productivity trend",
    'lag_yield_2': "Yield 2 years ago",
    'lag_yield_3': "Yield 3 years ago",
    'nasa_rainfall_rabi': "Winter season rainfall",
    'nasa_rainfall_kharif': "Monsoon season rainfall",
    'nasa_rainfall_annual': "Annual rainfall",
    'nasa_temp_max_rabi': "Winter max temperature",
    'nasa_temp_max_kharif': "Monsoon max temperature",
    'nasa_temp_avg_rabi': "Winter avg temperature",
    'nasa_temp_avg_kharif': "Monsoon avg temperature",
    'nasa_humidity_kharif': "Monsoon humidity",
    'nasa_solar_annual': "Solar radiation",
    'rainfall_anomaly_index': "Rainfall anomaly (RAI)",
    'fertilizer_per_ha': "Fertilizer use intensity",
    'nitrogen_per_ha': "Nitrogen use intensity",
    'irrigation_pct': "Irrigation coverage",
    'log_area': "Area sown",
    'state_encoded': "State-level baseline",
    'crop_encoded': "Crop type",
    'season_encoded': "Season",
    'frost_risk_days': "Frost risk days",
    'heat_stress_days': "Heat stress days",
    'yield_gap_vs_state': "Yield gap vs. state average",
    'yield_trend': "Recent yield trend",
    'drought_flag': "Drought flag",
    'flood_flag': "Flood flag",
}

# ---------------------------------------------------------------
# RULE-BASED ADVISORY LAYER (NOT model-generated)
# ---------------------------------------------------------------

def generate_advisory(row, pred_lower, pred_upper, pred_mid):
    advisories = []
    drought = row.get('drought_flag', 0) == 1
    flood = row.get('flood_flag', 0) == 1
    yield_gap = row.get('yield_gap_vs_state', 0)
    interval_width = pred_upper - pred_lower
    interval_width_pct = (interval_width / pred_mid * 100) if pred_mid > 0 else 0

    if drought:
        advisories.append({'level': 'alert', 'title': 'Drought Relief Review',
            'text': 'Rainfall anomaly indicates a significant deficit this season. Recommend flagging this district for drought relief scheme eligibility review and water-conservation irrigation advisories.'})
    if flood:
        advisories.append({'level': 'alert', 'title': 'Flood Damage Assessment',
            'text': 'Rainfall anomaly indicates significant excess this season. Recommend flood damage assessment and cross-checking crop insurance claim windows.'})
    if interval_width_pct > 28:
        advisories.append({'level': 'caution', 'title': 'High Prediction Uncertainty',
            'text': f'This prediction has a wider-than-typical interval ({interval_width:.0f} kg/ha range). Recommend field-level verification before finalizing procurement or insurance estimates.'})
    if pd.notna(yield_gap) and yield_gap < -400:
        advisories.append({'level': 'caution', 'title': 'Below State Average',
            'text': f'This district is trending {abs(yield_gap):.0f} kg/ha below the state average. May warrant review of input access or extension service coverage.'})
    if not advisories:
        advisories.append({'level': 'normal', 'title': 'No Flags Raised',
            'text': 'No drought, flood, or significant deviation flags for this district-year. Standard monitoring applies.'})
    return advisories


def render_advisory(advisories):
    css_map = {
        'alert':   ('naip-alert', 'naip-alert-title', '⚠'),
        'caution': ('naip-scenario', 'naip-scenario-title', '◐'),
        'normal':  ('naip-safe', 'naip-safe-title', '✓'),
    }
    for adv in advisories:
        box_class, title_class, icon = css_map[adv['level']]
        st.markdown(f"""
            <div class="{box_class}"><span class="{title_class}">{icon} {adv['title']}</span>{adv['text']}</div>
        """, unsafe_allow_html=True)


def render_live_context(state, district, crop, alert_type):
    """
    Renders an expandable 'Live Context' section using the RAG
    module (src/rag_grounding_31.py). Pulls live government
    relief scheme, MSP, and IMD forecast info via Tavily search.
    Degrades silently (shows a caption, never crashes) if the
    module or API key is unavailable.
    """
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    try:
        from rag_grounding_31 import get_live_grounding
    except ImportError:
        st.caption("Live context module not found (src/rag_grounding_31.py missing).")
        return

    with st.expander("🌐 Live Context — government schemes, MSP, weather outlook"):
        try:
            result = get_live_grounding(state, district, crop, alert_type=alert_type)
        except Exception as e:
            st.caption(f"Live grounding failed: {e}")
            return

        if not result.get('available'):
            st.caption(f"Live grounding unavailable: {result.get('reason', 'unknown reason')}")
            st.caption("To enable: set TAVILY_API_KEY or save a key to data/secrets/tavily_key.txt")
            return

        for section in result['sections']:
            st.markdown(f"**{section['category']}**")
            results = section.get('results', [])
            if not results:
                st.caption("No results found for this query.")
                continue
            for r in results:
                st.markdown(f"- [{r['title']}]({r['url']})")
                st.caption(r['content'])


def render_legend(items):
    """items: list of (hex_color, label) tuples"""
    chips = "".join(
        f'<div class="naip-legend-item"><span class="naip-legend-dot" style="background:{color};"></span>{label}</div>'
        for color, label in items
    )
    st.markdown(f'<div class="naip-legend">{chips}</div>', unsafe_allow_html=True)


def render_shap_bars(shap_df, max_abs):
    for _, srow in shap_df.iterrows():
        name = FRIENDLY_NAMES.get(srow['feature'], srow['feature'])
        pct_width = max(8, int(srow['abs_val'] / max_abs * 100)) if max_abs > 0 else 8
        fill_class = "shap-bar-fill-pos" if srow['shap_value'] >= 0 else "shap-bar-fill-neg"
        sign = "+" if srow['shap_value'] >= 0 else ""
        st.markdown(f"""
            <div class="shap-row">
                <div class="shap-label">{name}</div>
                <div class="shap-bar-track"><div class="{fill_class}" style="width:{pct_width}%;"></div></div>
                <div class="shap-value">{sign}{srow['shap_value']:.0f}</div>
            </div>
        """, unsafe_allow_html=True)


def empty_state(icon, text):
    st.markdown(f"""
        <div class="naip-empty">
            <div class="naip-empty-icon">{icon}</div>
            <div class="naip-empty-text">{text}</div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------------------

@st.cache_data
def load_test_predictions():
    return pd.read_csv('outputs/metrics/test_predictions_FINAL.csv')

@st.cache_data
def load_full_history():
    return pd.read_csv('data/processed/model_ready_v2.csv')

@st.cache_data
def load_current_predictions():
    path = 'data/processed/current_predictions_full.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model_and_shap():
    with open('outputs/models/xgb_tuned.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('data/processed/feature_list_v2.txt') as f:
        features = [line.strip() for line in f.readlines()]
    explainer = shap.TreeExplainer(model)
    return model, features, explainer

test_preds = load_test_predictions()
model, FEATURES, explainer = load_model_and_shap()
current_preds = load_current_predictions()

# NOTE: test_predictions_FINAL.csv already contains lat/lon —
# do NOT re-merge coordinates here (caused lat_x/lat_y bug previously).

# ---------------------------------------------------------------
# MASTHEAD
# ---------------------------------------------------------------

st.markdown("""
    <div class="naip-masthead">
        <div>
            <div class="naip-title">NAIP <span>/// Agricultural Intelligence</span></div>
            <div class="naip-subtitle">District Yield Decision Support &middot; Wheat &amp; Rice &middot; Indo-Gangetic Plain</div>
        </div>
        <div class="naip-masthead-tag">UP &middot; PUNJAB &middot; HARYANA<br>118 DISTRICTS MONITORED</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TOP-LEVEL STATS ROW
# ---------------------------------------------------------------

n_districts = test_preds['district'].nunique()
avg_yield = test_preds['pred_yield'].mean()
model_r2 = 0.8644

stat_cols = st.columns(4)
stats = [
    (f"{n_districts}", "Districts Monitored", "118 across UP &middot; Punjab &middot; Haryana"),
    (f"{avg_yield:,.0f}", "Avg. Predicted Yield (kg/ha)", "Wheat &amp; Rice, 2018&ndash;19 test set"),
    (f"{model_r2:.3f}", "Model R&sup2; (Test Set)", "Time-correct validation, 2018&ndash;19"),
    ("86.6%", "Interval Coverage", "Target band: 70&ndash;90%"),
]
for col, (num, label, sub) in zip(stat_cols, stats):
    col.markdown(f"""
        <div class="naip-card">
            <div class="naip-stat-number">{num}</div>
            <div class="naip-stat-label">{label}</div>
            <div class="naip-stat-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# MODE TOGGLE
# ---------------------------------------------------------------

mode = st.radio(
    "Mode",
    ["📊 Historical Results (2018–19 test set)",
     "🌤️ Current Season Outlook (2026, real weather)",
     "🔮 Scenario Explorer (any year, manual what-if)"],
    horizontal=True,
    label_visibility="collapsed",
)

st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

# =================================================================
# MODE 1 — HISTORICAL RESULTS
# =================================================================

if mode.startswith("📊"):

    st.markdown("""
        <div class="naip-mode-banner">
            Validated predictions against actual recorded yields for 2018&ndash;19 &mdash; the held-out test period
            never seen during model training. This is the basis for the model's reported accuracy (R&sup2; = 0.864).
        </div>
    """, unsafe_allow_html=True)

    map_col, panel_col = st.columns([1.6, 1])

    with map_col:
        st.markdown('<div class="naip-eyebrow">District Yield Map</div>', unsafe_allow_html=True)
        crop_filter = st.radio("Crop", ["Wheat", "Rice"], horizontal=True, label_visibility="collapsed", key="hist_crop")

        render_legend([
            ("#A33B2E", "Drought flagged"),
            ("#3D6B5C", "Flood flagged"),
            ("#E8C468", "Lower yield"),
            ("#C9A227", "Mid yield"),
            ("#8B5A2B", "Higher yield"),
        ])

        map_data = test_preds[test_preds['crop'] == crop_filter].dropna(subset=['lat', 'lon'])

        m = folium.Map(location=[28.5, 78.0], zoom_start=6, tiles='CartoDB positron')

        def get_color(row):
            if row.get('drought_flag', 0) == 1: return '#A33B2E'
            if row.get('flood_flag', 0) == 1: return '#3D6B5C'
            y = row['pred_yield']
            if y < 2500: return '#E8C468'
            elif y < 3500: return '#C9A227'
            else: return '#8B5A2B'

        for _, row in map_data.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']], radius=7,
                popup=f"<b>{row['district']}</b><br>Predicted: {row['pred_yield']:.0f} kg/ha<br>Range: {row['pred_lower']:.0f}–{row['pred_upper']:.0f}",
                tooltip=row['district'], color='#2B2417', weight=1, fill=True,
                fill_color=get_color(row), fill_opacity=0.85,
            ).add_to(m)

        map_output = st_folium(m, width=None, height=480, key="hist_map", returned_objects=["last_object_clicked_tooltip"])
        st.caption("Click any marker to view its prediction, explanation, and recommended actions.")

    with panel_col:
        selected_district = None
        if map_output and map_output.get("last_object_clicked_tooltip"):
            selected_district = map_output["last_object_clicked_tooltip"]

        if selected_district:
            row_match = test_preds[(test_preds['district'] == selected_district) & (test_preds['crop'] == crop_filter)]
            if len(row_match) > 0:
                row = row_match.iloc[0]

                st.markdown(f"""
                    <div class="naip-panel-state">{row['state']}</div>
                    <div class="naip-panel-district">{row['district']}</div>
                """, unsafe_allow_html=True)
                st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

                if row.get('drought_flag', 0) == 1:
                    st.markdown('<div class="naip-alert"><span class="naip-alert-title">⚠ Drought Risk Flagged</span>Rainfall anomaly index indicates significantly below-normal rainfall this season.</div>', unsafe_allow_html=True)
                elif row.get('flood_flag', 0) == 1:
                    st.markdown('<div class="naip-alert"><span class="naip-alert-title">⚠ Flood Risk Flagged</span>Rainfall anomaly index indicates significantly above-normal rainfall this season.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="naip-safe"><span class="naip-safe-title">✓ Normal Rainfall Pattern</span>No drought or flood anomaly detected this season.</div>', unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="naip-card">
                        <div class="naip-stat-number">{row['pred_yield']:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                        <div class="naip-stat-label">Predicted Yield — {crop_filter}, {int(row['year'])}</div>
                        <div class="naip-stat-sub">80% interval: {row['pred_lower']:,.0f} – {row['pred_upper']:,.0f} kg/ha</div>
                        <div class="naip-stat-sub" style="color: var(--soil);">Actual recorded: {row['actual_yield']:,.0f} kg/ha</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="naip-eyebrow">Why this prediction</div>', unsafe_allow_html=True)
                row_features = row[FEATURES].to_frame().T.astype(float)
                shap_vals_row = explainer.shap_values(row_features)[0]
                shap_df = pd.DataFrame({'feature': FEATURES, 'shap_value': shap_vals_row})
                shap_df['abs_val'] = shap_df['shap_value'].abs()
                shap_df = shap_df.sort_values('abs_val', ascending=False).head(6)
                render_shap_bars(shap_df, shap_df['abs_val'].max())

                st.markdown('<div class="naip-eyebrow" style="margin-top:18px;">Recommended Actions</div>', unsafe_allow_html=True)
                render_advisory(generate_advisory(row, row['pred_lower'], row['pred_upper'], row['pred_yield']))

                alert_type_1 = 'drought' if row.get('drought_flag', 0) == 1 else 'flood' if row.get('flood_flag', 0) == 1 else 'general'
                render_live_context(row['state'], row['district'], crop_filter, alert_type_1)
            else:
                empty_state("🌾", f"No {crop_filter} data available for {selected_district} in the test period.")
        else:
            empty_state("🌾", "Click any district marker on the map<br>to see its yield prediction and explanation.")

# =================================================================
# MODE 2 — CURRENT SEASON OUTLOOK (real 2026 weather)
# =================================================================

elif mode.startswith("🌤️"):

    if current_preds is None:
        empty_state("🌤️", "Current-season predictions have not been generated yet.<br>Run <code>src/27_generate_current_predictions.py</code> first.")
    else:
        st.markdown(f"""
            <div class="naip-mode-banner">
                Uses REAL current weather (NASA POWER, through {current_preds['weather_data_as_of'].iloc[0]}) for the
                just-completed Rabi 2025&ndash;26 season, combined with each district's most recently published yield
                data ({int(current_preds['baseline_yield_year'].iloc[0])} &mdash; government yield statistics publish
                1&ndash;2 years behind real time). <b>Weather signal is live; the yield-trend baseline is the latest
                published figure, not live.</b> Rice/Kharif 2026 outlook will be available once the monsoon season
                completes (around October 2026).
            </div>
        """, unsafe_allow_html=True)

        n_drought = current_preds['drought_flag'].sum()
        n_flood = current_preds['flood_flag'].sum()
        mean_change = current_preds['change_vs_2019'].mean()

        cstat_cols = st.columns(4)
        cstats = [
            (f"{len(current_preds)}", "Districts Assessed", "Wheat, current season"),
            (f"{current_preds['current_pred_yield'].mean():,.0f}", "Mean Predicted Yield (kg/ha)", "2026 outlook"),
            (f"{n_drought}", "Drought-Flagged Districts", "Below-normal Rabi rainfall"),
            (f"{mean_change:+.0f}", "Avg. Change vs 2019 (kg/ha)", "Directional trend signal"),
        ]
        for col, (num, label, sub) in zip(cstat_cols, cstats):
            col.markdown(f"""
                <div class="naip-card">
                    <div class="naip-stat-number">{num}</div>
                    <div class="naip-stat-label">{label}</div>
                    <div class="naip-stat-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

        cmap_col, cpanel_col = st.columns([1.6, 1])

        with cmap_col:
            st.markdown('<div class="naip-eyebrow">2026 Outlook Map</div>', unsafe_allow_html=True)

            render_legend([
                ("#A33B2E", "Drought flagged"),
                ("#3D6B5C", "Flood flagged"),
                ("#8B5A2B", "Trending up vs. 2019"),
                ("#E8C468", "Trending down vs. 2019"),
            ])

            cm = folium.Map(location=[28.5, 78.0], zoom_start=6, tiles='CartoDB positron')

            def get_color_current(r):
                if r.get('drought_flag', 0) == 1: return '#A33B2E'
                if r.get('flood_flag', 0) == 1: return '#3D6B5C'
                return '#8B5A2B' if r['change_vs_2019'] > 0 else '#E8C468'

            for _, r in current_preds.dropna(subset=['lat', 'lon']).iterrows():
                folium.CircleMarker(
                    location=[r['lat'], r['lon']], radius=7, tooltip=r['district'],
                    popup=f"<b>{r['district']}</b><br>2026 outlook: {r['current_pred_yield']:.0f} kg/ha<br>vs 2019: {r['change_vs_2019']:+.0f}",
                    color='#2B2417', weight=1, fill=True, fill_color=get_color_current(r), fill_opacity=0.85,
                ).add_to(cm)

            current_map_output = st_folium(cm, width=None, height=480, key="current_map", returned_objects=["last_object_clicked_tooltip"])
            st.caption("Click any marker to view its 2026 outlook, explanation, and recommended actions.")

        with cpanel_col:
            selected_current_district = None
            if current_map_output and current_map_output.get("last_object_clicked_tooltip"):
                selected_current_district = current_map_output["last_object_clicked_tooltip"]

            if selected_current_district:
                crow_match = current_preds[current_preds['district'] == selected_current_district]
                if len(crow_match) > 0:
                    crow = crow_match.iloc[0]

                    st.markdown(f"""
                        <div class="naip-panel-state">{crow['state']}</div>
                        <div class="naip-panel-district">{crow['district']}</div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

                    render_advisory(generate_advisory(
                        crow, crow['current_pred_yield'] * 0.85, crow['current_pred_yield'] * 1.15, crow['current_pred_yield']
                    ))

                    alert_type_2 = 'drought' if crow.get('drought_flag', 0) == 1 else 'flood' if crow.get('flood_flag', 0) == 1 else 'general'
                    render_live_context(crow['state'], crow['district'], 'Wheat', alert_type_2)

                    st.markdown(f"""
                        <div class="naip-card">
                            <div class="naip-stat-number">{crow['current_pred_yield']:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                            <div class="naip-stat-label">2026 Wheat Outlook</div>
                            <div class="naip-stat-sub">vs. 2019 actual: {crow['change_vs_2019']:+,.0f} kg/ha</div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="naip-eyebrow">Why this outlook</div>', unsafe_allow_html=True)
                    shap_cols = [c for c in crow.index if c.startswith('shap_')]
                    shap_df_c = pd.DataFrame({
                        'feature': [c.replace('shap_', '') for c in shap_cols],
                        'shap_value': crow[shap_cols].astype(float).values
                    })
                    shap_df_c['abs_val'] = shap_df_c['shap_value'].abs()
                    shap_df_c = shap_df_c.sort_values('abs_val', ascending=False).head(6)
                    render_shap_bars(shap_df_c, shap_df_c['abs_val'].max())
                else:
                    empty_state("🌾", "No data available for this district.")
            else:
                empty_state("🌤️", "Click any district marker to see its<br>2026 wheat outlook and explanation.")

# =================================================================
# MODE 3 — SCENARIO EXPLORER (manual what-if on historical years)
# =================================================================

else:
    st.markdown("""
        <div class="naip-mode-banner">
            Explore any district, crop, and historical year, and apply manual rainfall or temperature adjustments
            to see how the prediction responds. Uses recorded historical weather as the baseline — this is a
            what-if tool, not a forecast.
        </div>
    """, unsafe_allow_html=True)

    history = load_full_history()

    form_col1, form_col2, form_col3 = st.columns(3)
    with form_col1:
        live_state = st.selectbox("State", sorted(history['state'].unique()))
    with form_col2:
        districts_in_state = sorted(history[history['state'] == live_state]['district'].unique())
        live_district = st.selectbox("District", districts_in_state)
    with form_col3:
        live_crop = st.selectbox("Crop", ["Wheat", "Rice"])

    district_history = history[
        (history['state'] == live_state) & (history['district'] == live_district) & (history['crop'] == live_crop)
    ].sort_values('year')

    if len(district_history) == 0:
        empty_state("🔍", f"No historical data available for {live_district} — {live_crop}.<br>Try another combination.")
    else:
        available_years = district_history['year'].tolist()
        live_year = st.selectbox("Year to predict", available_years, index=len(available_years) - 1)

        row_data_match = district_history[district_history['year'] == live_year]

        if len(row_data_match) == 0:
            empty_state("🔍", "No data row found for this exact year.")
        else:
            row_data = row_data_match.iloc[0]

            st.markdown('<div class="naip-eyebrow" style="margin-top:18px;">Adjust Weather Scenario</div>', unsafe_allow_html=True)
            st.caption("Defaults are the actual recorded weather for this district-year. Adjust to simulate a what-if scenario.")

            wcol1, wcol2 = st.columns(2)
            with wcol1:
                rainfall_adj = st.slider("Seasonal rainfall adjustment (%)", -50, 50, 0,
                    help="Simulate a wetter or drier season than what actually occurred")
            with wcol2:
                temp_adj = st.slider("Temperature adjustment (°C)", -3.0, 3.0, 0.0, step=0.5,
                    help="Simulate a hotter or cooler season")

            live_features = row_data[FEATURES].copy().astype(float)
            rain_cols = [c for c in FEATURES if 'rainfall' in c]
            temp_cols = [c for c in FEATURES if 'temp' in c]
            for c in rain_cols: live_features[c] = live_features[c] * (1 + rainfall_adj / 100)
            for c in temp_cols: live_features[c] = live_features[c] + temp_adj

            X_live = live_features.to_frame().T
            live_pred = model.predict(X_live)[0]
            live_shap = explainer.shap_values(X_live)[0]

            st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
            result_col, explain_col = st.columns([1, 1.2])

            with result_col:
                actual_val = row_data.get('yield_kg_ha', None)
                st.markdown(f"""
                    <div class="naip-card">
                        <div class="naip-stat-number">{live_pred:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                        <div class="naip-stat-label">Predicted Yield — {live_crop}, {live_district}, {int(live_year)}</div>
                        {f'<div class="naip-stat-sub">Actual recorded yield: {actual_val:,.0f} kg/ha</div>' if pd.notna(actual_val) else ''}
                    </div>
                """, unsafe_allow_html=True)

                if rainfall_adj != 0 or temp_adj != 0:
                    st.markdown(f'<div class="naip-scenario"><span class="naip-scenario-title">Scenario Active</span>Rainfall {rainfall_adj:+d}%, Temperature {temp_adj:+.1f}&deg;C vs. actual recorded conditions.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="naip-safe"><span class="naip-safe-title">Baseline Scenario</span>Using actual recorded weather for this district-year.</div>', unsafe_allow_html=True)

            with explain_col:
                st.markdown('<div class="naip-eyebrow">Why this prediction</div>', unsafe_allow_html=True)
                shap_df_live = pd.DataFrame({'feature': FEATURES, 'shap_value': live_shap})
                shap_df_live['abs_val'] = shap_df_live['shap_value'].abs()
                shap_df_live = shap_df_live.sort_values('abs_val', ascending=False).head(6)
                render_shap_bars(shap_df_live, shap_df_live['abs_val'].max())

            st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="naip-eyebrow">Recommended Actions</div>', unsafe_allow_html=True)

            advisory_row = row_data.copy()
            advisory_row['drought_flag'] = 1 if rainfall_adj <= -25 else row_data.get('drought_flag', 0)
            advisory_row['flood_flag'] = 1 if rainfall_adj >= 25 else row_data.get('flood_flag', 0)
            render_advisory(generate_advisory(advisory_row, live_pred * 0.85, live_pred * 1.15, live_pred))

            alert_type_3 = 'drought' if advisory_row.get('drought_flag', 0) == 1 else 'flood' if advisory_row.get('flood_flag', 0) == 1 else 'general'
            render_live_context(live_state, live_district, live_crop, alert_type_3)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------

st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="naip-meta">
        NAIP v1.0 — Model: XGBoost (tuned) &middot; Test R&sup2;: 0.864 &middot; Validation: chronological split (train &le;2015, test 2018&ndash;19)<br>
        Data: NASA POWER &middot; ICRISAT/Mendeley &middot; India Agriculture Crop Production &middot; Geocoding: Nominatim/OpenStreetMap
    </div>
""", unsafe_allow_html=True)