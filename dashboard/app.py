# =============================================================
# ANNA Dashboard — Agricultural Nowcasting and Nearly-live Analytics
# अन्न (Anna) — "Food is Divine" (Taittiriya Upanishad: Annam Brahma)
# =============================================================
# Design language: "Earth & Grain"
#   Background : warm paper (#FAF6EE)
#   Ink        : near-black brown (#2B2417)
#   Wheat gold : #C9A227  (primary accent — predictions, highlights)
#   Soil brown : #8B5A2B  (secondary — structure, high yield)
#   Irrigation green : #3D6B5C (positive / healthy yield)
#   Alert terracotta : #A33B2E (drought / flood / risk)
#
# v2 — rebuilt as a real prediction tool, not a simulation:
#   1. Districts Requiring Attention (District Watch) — HOMEPAGE,
#      zero clicks: every district's outlook is already computed
#      and ranked by genuine change since the last check.
#   2. Kharif 2026 Early Warning — NEW capability, honestly scoped
#      as a risk CLASSIFICATION (not a yield prediction), using
#      real monsoon-so-far data vs. each district's own history.
#   3. Current Season Outlook — REAL 2026 weather (NASA POWER),
#      now with a REAL calibrated 80% interval (quantile models),
#      replacing point-only estimates and manual sliders as the
#      default uncertainty view.
#   4. Historical Results — locked 2018-19 test-set predictions,
#      validated against ground truth. Kept as proof the model works.
#   5. Scenario Explorer — DEMOTED to Advanced/Testing. Manual
#      what-if sliders on historical years; explicitly labeled as
#      not a forecast.
#
# All tabs share a rule-based Advisory panel (NOT model-generated)
# that translates SHAP/RAI/interval outputs into process-level
# actions an officer already has authority over.
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
    page_title="ANNA — Agricultural Nowcasting and Nearly-live Analytics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# DESIGN SYSTEM — injected CSS
# ---------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700;9..144,800&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ================================================================
   ANNA — Design System v3.0 "Earth & Grain, Elevated"
   Award-winning aesthetic: bold editorial typography, layered depth,
   micro-animations, glassmorphism accents, modern data visualization
   ================================================================ */

:root {
    /* Core palette */
    --paper:       #F8F4EC;
    --paper-deep:  #EFE8D8;
    --paper-glass: rgba(248,244,236,0.72);
    --ink:         #1C1610;
    --ink-soft:    #4A3F2F;
    --ink-faint:   #8C7B64;

    /* Brand */
    --wheat:       #C9A227;
    --wheat-dark:  #9C7E1A;
    --wheat-pale:  #F6EDCB;
    --wheat-glow:  rgba(201,162,39,0.18);
    --soil:        #7A4E2A;
    --soil-light:  #D4B896;
    --green:       #2E5E4E;
    --green-mid:   #3D6B5C;
    --green-bg:    #E4EDE9;
    --green-glow:  rgba(46,94,78,0.15);
    --alert:       #9B2C1E;
    --alert-mid:   #C0392B;
    --alert-bg:    #F7E8E5;
    --alert-glow:  rgba(155,44,30,0.15);
    --line:        #DDD3BD;
    --line-faint:  #EDE6D6;

    /* Gradients */
    --grad-hero:   linear-gradient(135deg, #1C1610 0%, #3D2B1A 40%, #5C3D1E 70%, #9C7E1A 100%);
    --grad-wheat:  linear-gradient(120deg, var(--wheat-pale), var(--paper));
    --grad-green:  linear-gradient(120deg, var(--green-bg), var(--paper));
    --grad-alert:  linear-gradient(120deg, var(--alert-bg), var(--paper));

    /* Shadows (layered depth) */
    --shadow-xs:   0 1px 2px rgba(28,22,16,0.06);
    --shadow-sm:   0 2px 8px rgba(28,22,16,0.08), 0 1px 2px rgba(28,22,16,0.04);
    --shadow-md:   0 8px 24px rgba(28,22,16,0.10), 0 2px 6px rgba(28,22,16,0.06);
    --shadow-lg:   0 20px 48px rgba(28,22,16,0.14), 0 4px 12px rgba(28,22,16,0.08);
    --shadow-glow: 0 0 0 3px var(--wheat-glow), 0 8px 24px rgba(201,162,39,0.12);

    /* Radii */
    --r-sm:  6px;
    --r-md:  12px;
    --r-lg:  20px;
    --r-xl:  28px;

    /* Transitions */
    --t-fast:   0.15s cubic-bezier(0.4,0,0.2,1);
    --t-smooth: 0.28s cubic-bezier(0.4,0,0.2,1);
    --t-spring: 0.4s cubic-bezier(0.34,1.56,0.64,1);
}

/* ── KEYFRAME ANIMATIONS ── */

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideRight {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 var(--wheat-glow); }
    50%       { box-shadow: 0 0 0 8px transparent; }
}
@keyframes barFill {
    from { width: 0%; opacity: 0.4; }
    to   { opacity: 1; }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@keyframes breathe {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.92); }
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes counterUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--paper) !important;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
.stApp { background-color: var(--paper); }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* ── HERO MASTHEAD ── */
.naip-masthead {
    background: var(--grad-hero);
    background-size: 200% 200%;
    animation: gradientShift 12s ease infinite;
    padding: 52px 56px 44px;
    margin: -1rem -1rem 0 -1rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.naip-masthead::before {
    content: "अन्न";
    position: absolute;
    right: 56px;
    top: 50%;
    transform: translateY(-60%);
    font-family: 'Fraunces', serif;
    font-size: 11rem;
    font-weight: 800;
    color: rgba(255,255,255,0.04);
    letter-spacing: -0.04em;
    line-height: 1;
    pointer-events: none;
    user-select: none;
}
.naip-masthead::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--wheat) 0%, rgba(201,162,39,0.3) 60%, transparent 100%);
}

.naip-title {
    font-family: 'Fraunces', serif;
    font-weight: 800;
    font-size: 3.8rem;
    letter-spacing: -0.03em;
    color: #FFFFFF;
    line-height: 0.95;
    animation: fadeUp 0.7s var(--t-smooth) both;
}
.naip-title span {
    color: var(--wheat);
    font-weight: 400;
    font-size: 1.9rem;
    display: block;
    margin-top: 6px;
    letter-spacing: -0.01em;
    animation: fadeUp 0.7s 0.1s var(--t-smooth) both;
}
.naip-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
    margin-top: 14px;
    animation: fadeUp 0.7s 0.2s var(--t-smooth) both;
}
.naip-masthead-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    text-align: right;
    line-height: 1.8;
    padding-bottom: 4px;
    animation: fadeIn 1s 0.4s both;
}

/* ── LIVE PULSE INDICATOR ── */
.anna-live-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.75);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 16px;
    backdrop-filter: blur(8px);
    animation: fadeIn 1s 0.5s both;
}
.anna-live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #4ADE80;
    animation: breathe 2s ease-in-out infinite;
    flex-shrink: 0;
}

/* ── STAT CARDS ── */
.naip-card {
    background: #FFFFFF;
    border: 1px solid var(--line-faint);
    border-radius: var(--r-lg);
    padding: 24px 26px 22px;
    margin-bottom: 16px;
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: hidden;
    transition: transform var(--t-smooth), box-shadow var(--t-smooth);
    animation: fadeUp 0.5s var(--t-smooth) both;
}
.naip-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--wheat), var(--wheat-dark));
    border-radius: var(--r-lg) var(--r-lg) 0 0;
}
.naip-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
}
.naip-stat-number {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
    animation: counterUp 0.6s var(--t-smooth) both;
}
.naip-stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: var(--ink-faint);
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.01em;
}
.naip-stat-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: var(--green-mid);
    margin-top: 10px;
    line-height: 1.4;
}

/* ── EYEBROWS ── */
.naip-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--wheat-dark);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.naip-eyebrow::before {
    content: "";
    display: inline-block;
    width: 20px;
    height: 2px;
    background: linear-gradient(90deg, var(--wheat-dark), var(--wheat));
    border-radius: 2px;
    animation: slideRight 0.4s var(--t-smooth) both;
    transform-origin: left;
}

/* ── STATUS BOXES ── */
.naip-alert {
    background: var(--grad-alert);
    border: 1px solid rgba(155,44,30,0.18);
    border-left: 4px solid var(--alert-mid);
    border-radius: var(--r-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px var(--alert-glow);
    animation: fadeUp 0.35s var(--t-smooth) both;
    transition: transform var(--t-fast), box-shadow var(--t-fast);
}
.naip-alert:hover {
    transform: translateX(3px);
    box-shadow: 0 4px 20px var(--alert-glow);
}
.naip-safe {
    background: var(--grad-green);
    border: 1px solid rgba(46,94,78,0.15);
    border-left: 4px solid var(--green-mid);
    border-radius: var(--r-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px var(--green-glow);
    animation: fadeUp 0.35s var(--t-smooth) both;
    transition: transform var(--t-fast);
}
.naip-safe:hover { transform: translateX(3px); }
.naip-scenario {
    background: var(--grad-wheat);
    border: 1px solid rgba(201,162,39,0.2);
    border-left: 4px solid var(--wheat-dark);
    border-radius: var(--r-md);
    padding: 16px 20px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px var(--wheat-glow);
    animation: fadeUp 0.35s var(--t-smooth) both;
    transition: transform var(--t-fast);
}
.naip-scenario:hover { transform: translateX(3px); }
.naip-alert-title, .naip-safe-title, .naip-scenario-title {
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
    display: block;
}
.naip-alert-title    { color: var(--alert-mid); }
.naip-safe-title     { color: var(--green); }
.naip-scenario-title { color: var(--wheat-dark); }

/* ── SHAP BARS ── */
.shap-row {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
    font-family: 'Inter', sans-serif;
    font-size: 0.83rem;
}
.shap-label {
    width: 42%;
    color: var(--ink-soft);
    padding-right: 12px;
    line-height: 1.3;
    font-size: 0.8rem;
}
.shap-bar-track {
    flex-grow: 1;
    height: 10px;
    background: var(--line-faint);
    border-radius: 100px;
    overflow: hidden;
}
.shap-bar-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, var(--green), var(--green-mid));
    border-radius: 100px;
    animation: barFill 0.8s var(--t-smooth) both;
    transform-origin: left;
}
.shap-bar-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, var(--alert), var(--alert-mid));
    border-radius: 100px;
    animation: barFill 0.8s var(--t-smooth) both;
    transform-origin: left;
}
.shap-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    width: 64px;
    text-align: right;
    color: var(--ink-faint);
    font-weight: 600;
    padding-left: 8px;
}

/* ── DIVIDERS ── */
.naip-divider {
    border: none;
    border-top: 1px solid var(--line-faint);
    margin: 28px 0 24px;
    position: relative;
}

/* ── PANEL HEADERS ── */
.naip-panel-district {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--ink);
    line-height: 1.1;
    animation: fadeUp 0.4s var(--t-smooth) both;
}
.naip-panel-state {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--soil);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
    animation: fadeIn 0.4s var(--t-smooth) both;
}

/* ── META / FOOTER ── */
.naip-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--ink-faint);
    line-height: 1.7;
    padding: 20px 0 8px;
    border-top: 1px solid var(--line-faint);
}

/* ── MAP LEGEND ── */
.naip-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 12px 16px;
    background: #FFFFFF;
    border: 1px solid var(--line-faint);
    border-radius: var(--r-md);
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
    font-size: 0.76rem;
    color: var(--ink-soft);
    box-shadow: var(--shadow-xs);
}
.naip-legend-item { display: flex; align-items: center; gap: 7px; }
.naip-legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
    border: 2px solid rgba(28,22,16,0.2);
    box-shadow: 0 0 0 2px rgba(255,255,255,0.8);
}

/* ── EMPTY STATE ── */
.naip-empty {
    text-align: center;
    padding: 72px 24px;
    background: #FFFFFF;
    border: 1px dashed var(--line);
    border-radius: var(--r-xl);
    animation: fadeIn 0.4s both;
}
.naip-empty-icon { font-size: 2.4rem; margin-bottom: 14px; opacity: 0.5; }
.naip-empty-text {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    color: var(--ink-faint);
    line-height: 1.6;
}

/* ── MODE BANNERS ── */
.naip-mode-banner {
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    color: var(--ink-soft);
    line-height: 1.6;
    padding: 16px 20px;
    background: #FFFFFF;
    border: 1px solid var(--line-faint);
    border-radius: var(--r-md);
    margin-bottom: 20px;
    box-shadow: var(--shadow-xs);
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.4s var(--t-smooth) both;
}
.naip-mode-banner::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--wheat), var(--wheat-dark));
    border-radius: var(--r-md) 0 0 var(--r-md);
}

/* ── TABS STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: var(--r-md) !important;
    padding: 6px !important;
    gap: 4px !important;
    border: 1px solid var(--line-faint) !important;
    box-shadow: var(--shadow-xs) !important;
    margin-bottom: 20px !important;
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--r-sm) !important;
    border: none !important;
    color: var(--ink-faint) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: all var(--t-fast) !important;
    white-space: nowrap;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--wheat-pale) !important;
    color: var(--wheat-dark) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--soil) 0%, var(--ink) 100%) !important;
    color: #FFFFFF !important;
    box-shadow: var(--shadow-sm) !important;
}
/* Force every nested element inside the active tab (Streamlit wraps the
   label text in its own <p>/<div>) to white — otherwise the global
   ".stApp *" ink-color override below wins on those children and the
   label text goes dark-on-dark and disappears, leaving only the emoji
   visible against a near-black bar. */
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] * {
    color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 4px !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}

/* ── STREAMLIT CHROME CLEANUP ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
div[data-testid="stRadio"] > label { display: none; }

/* ── CONTRAST OVERRIDES (must come last) ── */
.stApp, .stApp * { color: var(--ink) !important; }
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--ink) !important; }

div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p {
    color: var(--ink) !important; font-weight: 600 !important;
}
div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: var(--r-sm) !important;
}
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
    border-radius: var(--r-sm) !important;
}
div[data-baseweb="select"] span { color: var(--ink) !important; }
ul[role="listbox"] { background-color: #FFFFFF !important; }
ul[role="listbox"] li { color: var(--ink) !important; background-color: #FFFFFF !important; }
ul[role="listbox"] li:hover { background-color: var(--wheat-pale) !important; }

div[data-testid="stSlider"] label {
    color: var(--ink-soft) !important; font-weight: 600 !important; font-size: 0.85rem !important;
}
div[data-testid="stThumbValue"] {
    color: #FFFFFF !important; background-color: var(--ink) !important;
    border-radius: 6px !important;
}
.stCaption, [data-testid="stCaptionContainer"], small {
    color: var(--ink-faint) !important; opacity: 1 !important;
}
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid var(--line-faint) !important;
    border-radius: var(--r-md) !important;
    box-shadow: var(--shadow-xs) !important;
}
div[data-testid="stExpander"] summary {
    color: var(--ink) !important; font-weight: 600 !important;
    background-color: #FFFFFF !important;
}
div[data-testid="stExpander"] summary:hover { color: var(--wheat-dark) !important; }
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] li { color: var(--ink) !important; }
div[data-testid="stExpander"] a { color: var(--green) !important; text-decoration: underline !important; }
.stMarkdown a { color: var(--green) !important; text-decoration: underline !important; }
div[data-testid="stAlert"] { background-color: var(--paper-deep) !important; border-radius: var(--r-md) !important; }
div[data-testid="stAlert"] p { color: var(--ink) !important; }
code {
    color: var(--alert) !important;
    background-color: var(--wheat-pale) !important;
    padding: 2px 6px !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Re-assert custom component colors after global override */
.naip-stat-number   { color: var(--ink) !important; font-family: 'Fraunces', serif !important; }
.naip-stat-label    { color: var(--ink-faint) !important; }
.naip-stat-sub      { color: var(--green-mid) !important; }
.naip-alert-title   { color: var(--alert-mid) !important; }
.naip-safe-title    { color: var(--green) !important; }
.naip-scenario-title{ color: var(--wheat-dark) !important; }
.naip-alert, .naip-alert *  { color: var(--ink) !important; }
.naip-safe, .naip-safe *    { color: var(--ink) !important; }
.naip-scenario, .naip-scenario * { color: var(--ink) !important; }
.naip-title         { color: #FFFFFF !important; font-family: 'Fraunces', serif !important; }
.naip-title span    { color: var(--wheat) !important; }
.naip-subtitle      { color: rgba(255,255,255,0.55) !important; }
.naip-masthead-tag  { color: rgba(255,255,255,0.4) !important; }
.naip-eyebrow       { color: var(--wheat-dark) !important; }
.naip-panel-district{ color: var(--ink) !important; font-family: 'Fraunces', serif !important; }
.naip-panel-state   { color: var(--soil) !important; }
.naip-meta          { color: var(--ink-faint) !important; }
.naip-mode-banner   { color: var(--ink-soft) !important; }
.naip-mode-banner * { color: var(--ink-soft) !important; }
.naip-legend        { color: var(--ink-soft) !important; }
.naip-empty-text    { color: var(--ink-faint) !important; }
.shap-label         { color: var(--ink-soft) !important; }
.shap-value         { color: var(--ink-faint) !important; }
.anna-live-badge, .anna-live-badge * { color: rgba(255,255,255,0.75) !important; }
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] * { color: var(--ink-faint) !important; }
.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover * { color: var(--wheat-dark) !important; }
/* Must come after the plain-tab rule above so the active tab's white
   text wins over the ink-faint default on the same descendants. */
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] * { color: #FFFFFF !important; }
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# FRIENDLY FEATURE NAMES
# ---------------------------------------------------------------

FRIENDLY_NAMES = {
    # Autoregressive features - explicitly labeled with data vintage
    # so the "Why this outlook" panel is honest about what it's using
    'lag_yield_1':        "Last published yield (2019)",
    'rolling_yield_3yr':  "3-year average yield (up to 2019)",
    'time_trend':         "Long-run productivity trend",
    'lag_yield_2':        "Yield 2 seasons ago (2018)",
    'lag_yield_3':        "Yield 3 seasons ago (2017)",
    'yield_trend':        "Recent yield direction",
    'yield_gap_vs_state': "Gap vs. state average",
    # Live weather features - these ARE current-season data
    'nasa_rainfall_rabi':     "This season's winter rainfall (live)",
    'nasa_rainfall_kharif':   "This season's monsoon rainfall (live)",
    'nasa_rainfall_annual':   "Annual rainfall (live)",
    'nasa_temp_max_rabi':     "Peak winter temperature (live)",
    'nasa_temp_max_kharif':   "Peak monsoon temperature (live)",
    'nasa_temp_avg_rabi':     "Average winter temperature (live)",
    'nasa_temp_avg_kharif':   "Average monsoon temperature (live)",
    'nasa_humidity_kharif':   "Monsoon humidity (live)",
    'nasa_solar_annual':      "Solar radiation",
    'rainfall_anomaly_index': "Rainfall vs. district's own history",
    'heat_stress_days':       "Days above 35°C",
    'frost_risk_days':        "Frost-risk days",
    'drought_flag':           "Drought flag",
    'flood_flag':             "Flood flag",
    # Agronomic features (new, computed from daily weather)
    'gdd_kharif':             "Monsoon growing degree days (live)",
    'gdd_rabi':               "Winter growing degree days (live)",
    'max_dry_streak_kharif':  "Longest dry spell in monsoon (live)",
    'max_dry_streak_rabi':    "Longest dry spell in winter (live)",
    'rainfall_cv_kharif':     "Monsoon rainfall consistency (live)",
    'rainfall_cv_rabi':       "Winter rainfall consistency (live)",
    'water_balance_kharif':   "Net water balance, monsoon (live)",
    'water_balance_rabi':     "Net water balance, winter (live)",
    # NDVI satellite vegetation
    'ndvi_kharif':            "Monsoon crop health — satellite (live)",
    'ndvi_rabi':              "Winter crop health — satellite (live)",
    # Management / structural
    'fertilizer_per_ha': "Fertiliser use",
    'nitrogen_per_ha':   "Nitrogen use",
    'irrigation_pct':    "Irrigation coverage",
    'log_area':          "Area sown",
    'state_encoded':     "State-level baseline",
    'crop_encoded':      "Crop type",
    'season_encoded':    "Season",
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
    crop = row.get('crop', 'crop')
    district = row.get('district', 'this district')
    state = row.get('state', 'the state')

    if drought:
        advisories.append({'level': 'alert', 'title': 'Drought Conditions Detected',
            'text': (
                f"{district} is showing a significant rainfall deficit this season compared to its own historical average. "
                f"Recommended actions: (1) Check eligibility for central drought relief funds under NDRF/SDRF norms. "
                f"(2) Issue water-conservation advisories to farmers &mdash; prioritise micro-irrigation for standing crops. "
                f"(3) Coordinate with the state agriculture department on contingency crop plans if deficit continues."
            )})

    if flood:
        advisories.append({'level': 'alert', 'title': 'Excess Rainfall / Flood Risk',
            'text': (
                f"{district} is recording significantly above-normal rainfall this season. "
                f"Recommended actions: (1) Conduct rapid crop damage surveys in low-lying areas. "
                f"(2) Ensure farmers are aware of the crop insurance claim window &mdash; most policies require "
                f"notification within 72 hours of damage. (3) Coordinate drainage relief with the irrigation department."
            )})

    if interval_width_pct > 28:
        advisories.append({'level': 'caution', 'title': 'Verify Before Acting on This Number',
            'text': (
                f"The model is less certain than usual about {district}'s yield this season &mdash; the range of likely "
                f"outcomes is wider than normal. This typically happens when current weather conditions are unusual "
                f"compared to the historical pattern this district is benchmarked against. "
                f"Before using this figure for procurement targets, insurance estimates, or PDS planning, "
                f"cross-check with a field-level crop-cutting experiment (CCE) or block-level officer report."
            )})

    if pd.notna(yield_gap) and yield_gap < -400:
        gap_pct = abs(yield_gap) / (pred_mid - yield_gap) * 100 if pred_mid > 0 else 0
        advisories.append({'level': 'caution', 'title': 'Consistently Below State Average',
            'text': (
                f"{district} is predicted to yield roughly {abs(yield_gap):,.0f} kg/ha below the {state} average "
                f"({gap_pct:.0f}% lower). This gap has persisted across multiple seasons, suggesting a structural "
                f"issue rather than a one-off weather event. Recommended review areas: fertiliser and seed access "
                f"at the farm level, irrigation infrastructure coverage, and whether extension service visits are "
                f"reaching the most affected blocks."
            )})

    if not advisories:
        advisories.append({'level': 'normal', 'title': 'No Flags This Season',
            'text': (
                f"{district} does not show any drought, flood, or significant yield-gap flags for this season. "
                f"Standard monitoring applies. The yield outlook is within normal historical range for this district."
            )})
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
    import sys
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)

    try:
        from rag_grounding_31 import get_live_grounding
    except ImportError:
        return  # silently skip - module not present

    with st.expander("🌐 Live Context — government schemes, MSP, weather outlook"):
        try:
            result = get_live_grounding(state, district, crop, alert_type=alert_type)
        except Exception:
            st.markdown("""
                <div class="naip-scenario">
                    <span class="naip-scenario-title">Live context not available</span>
                    The live government scheme and MSP lookup couldn't connect right now.
                    This doesn't affect the yield prediction or alerts above &mdash; those use
                    locally stored model data. Try again after a few minutes, or check your
                    internet connection.
                </div>
            """, unsafe_allow_html=True)
            return

        if not result.get('available'):
            st.markdown(f"""
                <div class="naip-scenario">
                    <span class="naip-scenario-title">Live context not configured</span>
                    This section pulls real-time information on government relief schemes, Minimum
                    Support Prices (MSP), and IMD weather forecasts specific to this district and
                    crop. To enable it, a Tavily API key needs to be configured by the system
                    administrator &mdash; contact your ANNA support team. The yield predictions and
                    district alerts above are unaffected.
                </div>
            """, unsafe_allow_html=True)
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

@st.cache_data
def load_district_watch():
    path = 'outputs/metrics/district_watch_feed.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_kharif_risk():
    path = 'data/processed/kharif_2026_risk_flags.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_quantile_models():
    """
    Loads the REAL trained quantile-regression models (10th / 90th
    percentile) that produce the dashboard's published 80% interval
    coverage (82.1%). Used to give the Current Season Outlook a real
    calibrated range instead of a single point number - this is what
    replaces the old manual what-if sliders as the default uncertainty
    view, per the honest-prediction-model rework.
    """
    with open('outputs/models/xgb_lower.pkl', 'rb') as f:
        lower = pickle.load(f)
    with open('outputs/models/xgb_upper.pkl', 'rb') as f:
        upper = pickle.load(f)
    return lower, upper

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
district_watch = load_district_watch()
kharif_risk = load_kharif_risk()
xgb_lower_model, xgb_upper_model = load_quantile_models()

# Real calibrated 80% interval for the current 2026 outlook (replaces
# point-only estimates with an actual probabilistic range, using the
# SAME quantile models that produced the dashboard's published 82.1%
# coverage figure - not an arbitrary slider-driven number).
if current_preds is not None:
    X_current_all = current_preds[FEATURES].astype(float)
    current_preds['pred_lower_2026'] = xgb_lower_model.predict(X_current_all)
    current_preds['pred_upper_2026'] = xgb_upper_model.predict(X_current_all)

# Official IMD 2026 Southwest Monsoon outlook - verified live via RAG
# (script 31), reported as context wherever Kharif risk is discussed.
# A NATIONAL figure (district-level IMD sub-forecasts aren't available
# via the RAG layer for all 118 districts) - stated explicitly.
IMD_2026_LPA_PCT = 90
IMD_DEFICIENT_PROB_PCT = 60
IMD_VERIFIED_DATE = "2026-05-29"

# NOTE: test_predictions_FINAL.csv already contains lat/lon —
# do NOT re-merge coordinates here (caused lat_x/lat_y bug previously).

# ---------------------------------------------------------------
# MASTHEAD
# ---------------------------------------------------------------

st.markdown("""
    <div class="naip-masthead">
        <div>
            <div class="naip-title">ANNA <span>/// Agricultural Nowcasting &amp; Nearly-live Analytics</span></div>
            <div class="naip-subtitle">अन्न &middot; District Yield Decision Support &middot; Wheat &amp; Rice &middot; Indo-Gangetic Plain</div>
            <div class="anna-live-badge">
                <span class="anna-live-dot"></span>
                Live &mdash; NASA POWER &middot; MODIS NDVI &middot; JRC Surface Water
            </div>
        </div>
        <div class="naip-masthead-tag">UP &middot; PUNJAB &middot; HARYANA<br>118 DISTRICTS<br>40 FEATURES<br>XGBoost v2.0</div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# TOP-LEVEL STATS ROW
# ---------------------------------------------------------------

n_districts = current_preds['district'].nunique() if current_preds is not None else 118
avg_yield_current = current_preds['current_pred_yield'].mean() if current_preds is not None else 0
model_r2 = 0.8592  # single chronological test-split (2018-19), NOT walk-forward

stat_cols = st.columns(4)
stats = [
    (f"{n_districts}", "Districts Monitored", "118 across UP &middot; Punjab &middot; Haryana"),
    (f"{avg_yield_current:,.0f}", "Avg. 2026 Wheat Outlook (kg/ha)", "Live model, real 2026 NASA weather"),
    (f"{model_r2:.3f}", "Model R&sup2; (2018&ndash;19 test set)", "Walk-forward CV range: 0.19&ndash;0.93"),
    ("82.1%", "Interval Coverage", "80% prediction band &middot; target 70&ndash;90%"),
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

tab_watch, tab_kharif, tab_current, tab_hist, tab_scenario, tab_about = st.tabs([
    "🚨 Districts Requiring Attention",
    "🌱 Kharif 2026 Early Warning",
    "🌤️ Current Season Outlook",
    "📊 Historical Results",
    "🔮 Scenario Explorer",
    "ℹ️ About",
])

# =================================================================
# TAB 1 — DISTRICTS REQUIRING ATTENTION (District Watch) — HOMEPAGE
# =================================================================
# This is the first tab an officer sees, with zero clicks required.
# Every district's latest 2026 outlook is already computed and
# ranked by genuine change since the last check - not a map to
# explore, a prioritized list you're handed.
# =================================================================

with tab_watch:
    if district_watch is None or len(district_watch) == 0:
        empty_state("🚨", "District Watch has not been run yet.<br>Run <code>src/28_district_watch.py</code> first.")
    else:
        st.markdown(f"""
            <div class="naip-mode-banner">
                This is the default view because it needs no input: every district's latest 2026 outlook is
                automatically compared against its previous check. <b>Districts are ranked by how much has
                genuinely changed</b> &mdash; not by raw severity alone &mdash; so a long-standing drought that
                hasn't worsened ranks below a district that just started deteriorating.<br><br>
                <b>Note on crops shown:</b> Currently displaying <b>Wheat (Rabi) only</b>. This is because Wheat's
                2025&ndash;26 season has completed and real weather data exists for the full growing period. Rice
                (Kharif) 2026 predictions will appear here automatically once the monsoon season completes
                (~October 2026) and the model has a full season of weather to work from &mdash; predicting Rice
                yields mid-season would be overclaiming.
            </div>
        """, unsafe_allow_html=True)

        level_counts = district_watch['alert_level'].value_counts()
        n_urgent = int(level_counts.get('urgent', 0))
        n_watch = int(level_counts.get('watch', 0))
        n_new = int(level_counts.get('new', 0))
        n_minor = int(level_counts.get('minor_change', 0))
        n_stable = int(level_counts.get('stable', 0) + level_counts.get('elevated_stable', 0) + level_counts.get('baseline', 0))

        wstat_cols = st.columns(5)
        wstats = [
            (f"{n_urgent}", "Urgent", "Large shift since last check &mdash; review first"),
            (f"{n_watch}", "Watch", "Meaningful change, not yet urgent"),
            (f"{n_minor}", "Minor change", "Real but small shift, below alert threshold"),
            (f"{n_new}", "New", "No prior snapshot to compare against"),
            (f"{n_stable}", "Stable", "Genuinely no change since last check"),
        ]
        for col, (num, label, sub) in zip(wstat_cols, wstats):
            col.markdown(f"""
                <div class="naip-card">
                    <div class="naip-stat-number">{num}</div>
                    <div class="naip-stat-label">{label}</div>
                    <div class="naip-stat-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

        level_filter = st.multiselect(
            "Show alert levels",
            ["urgent", "watch", "minor_change", "new", "elevated_stable", "stable", "baseline"],
            default=["urgent", "watch", "new"],
            key="watch_level_filter",
        )

        feed = district_watch[district_watch['alert_level'].isin(level_filter)].copy()
        alert_order = {'urgent': 0, 'watch': 1, 'new': 2, 'elevated_stable': 3, 'minor_change': 4, 'stable': 5, 'baseline': 6}
        feed['_sort'] = feed['alert_level'].map(alert_order)
        feed = feed.sort_values(['_sort', 'rai_severity_relative'], ascending=[True, False])

        WATCH_BADGE = {
            'urgent':           ('naip-alert',    'naip-alert-title',    '⚠ Urgent'),
            'watch':            ('naip-scenario', 'naip-scenario-title', '◐ Watch'),
            'new':              ('naip-scenario', 'naip-scenario-title', '● New'),
            'elevated_stable':  ('naip-scenario', 'naip-scenario-title', '◐ Elevated, stable'),
            'minor_change':     ('naip-safe',     'naip-safe-title',     '· Minor change'),
            'stable':           ('naip-safe',     'naip-safe-title',     '✓ Stable'),
            'baseline':         ('naip-safe',     'naip-safe-title',     '✓ Baseline'),
        }

        list_col, detail_col = st.columns([1.3, 1])

        with list_col:
            st.markdown('<div class="naip-eyebrow">Ranked Alert Feed</div>', unsafe_allow_html=True)
            if len(feed) == 0:
                empty_state("✓", "No districts match the selected filter.")
            else:
                for _, r in feed.iterrows():
                    box_class, title_class, icon_title = WATCH_BADGE.get(r['alert_level'], WATCH_BADGE['stable'])
                    st.markdown(f"""
                        <div class="{box_class}" style="margin-bottom:8px;">
                            <span class="{title_class}">{icon_title} &mdash; {r['district']} ({r['state']}), {r['crop']}</span>
                            {r['change_note']}
                        </div>
                    """, unsafe_allow_html=True)

        with detail_col:
            st.markdown('<div class="naip-eyebrow">Inspect a District</div>', unsafe_allow_html=True)
            if len(feed) == 0:
                empty_state("🚨", "No districts in the current filter to inspect.")
            else:
                pick_options = [f"{r['district']} — {r['crop']}" for _, r in feed.iterrows()]
                picked = st.selectbox("District", pick_options, key="watch_detail_pick", label_visibility="collapsed")
                picked_district, picked_crop = picked.rsplit(" — ", 1)
                wrow_match = feed[(feed['district'] == picked_district) & (feed['crop'] == picked_crop)]

                if len(wrow_match) > 0:
                    wrow = wrow_match.iloc[0]

                    st.markdown(f"""
                        <div class="naip-panel-state">{wrow['state']}</div>
                        <div class="naip-panel-district">{wrow['district']}</div>
                    """, unsafe_allow_html=True)
                    st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

                    box_class, title_class, icon_title = WATCH_BADGE.get(wrow['alert_level'], WATCH_BADGE['stable'])
                    st.markdown(f"""
                        <div class="{box_class}"><span class="{title_class}">{icon_title}</span>{wrow['change_note']}</div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class="naip-card">
                            <div class="naip-stat-number">{wrow['current_pred_yield']:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                            <div class="naip-stat-label">Latest 2026 Outlook &mdash; {wrow['crop']}</div>
                            <div class="naip-stat-sub">Rainfall anomaly severity (vs. this district's own history): {wrow['rai_severity_relative']:.2f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.caption("In plain terms: this score measures how unusual this season's rainfall anomaly is "
                               "compared to what's normal for THIS district specifically — not a fixed national scale.")

                    shap_cols_w = [c for c in wrow.index if c.startswith('shap_')]
                    if shap_cols_w:
                        st.markdown('<div class="naip-eyebrow">Why this outlook</div>', unsafe_allow_html=True)
                        shap_df_w = pd.DataFrame({
                            'feature': [c.replace('shap_', '') for c in shap_cols_w],
                            'shap_value': wrow[shap_cols_w].astype(float).values
                        })
                        shap_df_w['abs_val'] = shap_df_w['shap_value'].abs()
                        shap_df_w = shap_df_w.sort_values('abs_val', ascending=False).head(6)
                        render_shap_bars(shap_df_w, shap_df_w['abs_val'].max())

                    st.markdown('<div class="naip-eyebrow" style="margin-top:18px;">Recommended Actions</div>', unsafe_allow_html=True)
                    # Uses this district's REAL calibrated 80% interval (already
                    # computed for current_preds via the quantile models) instead
                    # of a fabricated +/-15% band, so the "wide interval, verify
                    # before acting" flag reflects genuine model uncertainty.
                    watch_lower = wrow['pred_lower_2026'] if 'pred_lower_2026' in wrow.index and pd.notna(wrow['pred_lower_2026']) else wrow['current_pred_yield'] * 0.85
                    watch_upper = wrow['pred_upper_2026'] if 'pred_upper_2026' in wrow.index and pd.notna(wrow['pred_upper_2026']) else wrow['current_pred_yield'] * 1.15
                    render_advisory(generate_advisory(
                        wrow, watch_lower, watch_upper, wrow['current_pred_yield']
                    ))

                    alert_type_w = 'drought' if wrow.get('drought_flag', 0) == 1 else 'flood' if wrow.get('flood_flag', 0) == 1 else 'general'
                    render_live_context(wrow['state'], wrow['district'], wrow['crop'], alert_type_w)
                else:
                    empty_state("🚨", "No data found for that selection.")

# =================================================================
# TAB 2 — KHARIF 2026 EARLY WARNING (new capability)
# =================================================================
# HONEST SCOPE: this is a RISK CLASSIFICATION, not a yield
# prediction. The Kharif/Rice season is still in progress and a
# yield number now would be overclaiming. What this CAN honestly
# say: based on real rainfall recorded so far compared to each
# district's own history, plus the official IMD national outlook
# (reported separately as context, not blended into the number),
# here is which districts are most exposed to risk right now.
# =================================================================

with tab_kharif:
    from datetime import date
    current_month = date.today().month
    # Kharif (monsoon) season: June-October → focus on Kharif risk
    # Rabi (winter) season: November-March → focus on Rabi outlook
    # April-May: inter-season, prompt user to check District Watch for Rabi results
    if current_month in range(6, 11):
        season_label = "Kharif (Monsoon) 2026"
        season_note = "The <b>Kharif (monsoon/rice) season is currently in progress</b>. The risk classification below is based on real rainfall recorded so far this season compared to each district's own history. Full yield prediction will be available once the season completes (~October)."
    elif current_month in [11, 12] or current_month in range(1, 4):
        season_label = "Rabi (Winter) Season Active"
        season_note = "The <b>Rabi (winter/wheat) season is currently in progress</b>. Check the <b>Current Season Outlook</b> tab for live district-level wheat predictions based on real NASA weather data. Kharif 2026 results will appear in <b>Historical Results</b> once the government publishes them (~2027)."
    else:
        season_label = "Inter-Season Period"
        season_note = "This is the inter-season period (April–May). Rabi (wheat) harvest results are being compiled; Kharif (rice) sowing begins in June. Check <b>Districts Requiring Attention</b> for the latest Wheat 2026 outlook."

    if kharif_risk is None or len(kharif_risk) == 0:
        empty_state("🌱", "Kharif early-warning data not found.<br>Run <code>src/32_kharif_early_warning.py</code> first.")
    else:
        st.markdown(f"""
            <div class="naip-mode-banner">
                <b>{season_label}</b> &mdash; {season_note}<br><br>
                <b>What this tab shows:</b> Each district's actual rainfall recorded so far this June compared
                to that SAME calendar window in 2020&ndash;2025 &mdash; an honest like-for-like comparison,
                not a linear extrapolation. For context (not blended into the number): IMD's official national
                outlook is {IMD_2026_LPA_PCT}% of the Long Period Average, with a {IMD_DEFICIENT_PROB_PCT}%
                chance of a deficient season nationally (IMD, verified {IMD_VERIFIED_DATE}).
                <b>This is a risk flag, not a yield number.</b>
            </div>
        """, unsafe_allow_html=True)

        krisk_counts = kharif_risk['kharif_risk_level'].value_counts()
        kstat_cols = st.columns(4)
        kstats = [
            (f"{int(krisk_counts.get('high', 0))}", "High Risk", "Tracking well below normal for this point in season"),
            (f"{int(krisk_counts.get('moderate', 0))}", "Moderate Risk", "Somewhat below normal so far"),
            (f"{int(krisk_counts.get('low', 0))}", "Low Risk", "Tracking near or above normal"),
            (f"{int(krisk_counts.get('insufficient_data', 0))}", "Insufficient Data", "Fewer than 3 years of history available"),
        ]
        for col, (num, label, sub) in zip(kstat_cols, kstats):
            col.markdown(f"""
                <div class="naip-card">
                    <div class="naip-stat-number">{num}</div>
                    <div class="naip-stat-label">{label}</div>
                    <div class="naip-stat-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)

        krisk_filter = st.multiselect(
            "Show risk levels", ["high", "moderate", "low", "insufficient_data"],
            default=["high", "moderate"], key="kharif_level_filter",
        )
        kfeed = kharif_risk[kharif_risk['kharif_risk_level'].isin(krisk_filter)].sort_values('kharif_risk_zscore')

        KHARIF_BADGE = {
            'high':              ('naip-alert',    'naip-alert-title',    '⚠ High Risk'),
            'moderate':          ('naip-scenario', 'naip-scenario-title', '◐ Moderate Risk'),
            'low':               ('naip-safe',     'naip-safe-title',     '✓ Low Risk'),
            'insufficient_data': ('naip-scenario', 'naip-scenario-title', '? Insufficient Data'),
        }

        if len(kfeed) == 0:
            empty_state("🌱", "No districts match the selected filter.")
        else:
            for _, kr in kfeed.iterrows():
                box_class, title_class, icon_title = KHARIF_BADGE.get(kr['kharif_risk_level'], KHARIF_BADGE['moderate'])
                st.markdown(f"""
                    <div class="{box_class}" style="margin-bottom:8px;">
                        <span class="{title_class}">{icon_title} &mdash; {kr['district']} ({kr['state']})</span>
                        {kr['risk_explanation']}
                    </div>
                """, unsafe_allow_html=True)

# =================================================================
# TAB 4 — HISTORICAL RESULTS (kept as proof the model works)
# =================================================================

with tab_hist:

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
# TAB 3 — CURRENT SEASON OUTLOOK (real 2026 weather, real interval)
# =================================================================

with tab_current:

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
                    popup=f"<b>{r['district']}</b><br>2026 outlook: {r['current_pred_yield']:.0f} kg/ha "
                          f"({r['pred_lower_2026']:.0f}&ndash;{r['pred_upper_2026']:.0f})<br>vs 2019: {r['change_vs_2019']:+.0f}",
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

                    # Uses the district's REAL calibrated 80% interval instead
                    # of a fabricated +/-15% band, so the interval-width-based
                    # "verify before acting" flag reflects genuine uncertainty.
                    render_advisory(generate_advisory(
                        crow, crow['pred_lower_2026'], crow['pred_upper_2026'], crow['current_pred_yield']
                    ))

                    alert_type_2 = 'drought' if crow.get('drought_flag', 0) == 1 else 'flood' if crow.get('flood_flag', 0) == 1 else 'general'
                    render_live_context(crow['state'], crow['district'], 'Wheat', alert_type_2)

                    st.markdown(f"""
                        <div class="naip-card">
                            <div class="naip-stat-number">{crow['current_pred_yield']:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                            <div class="naip-stat-label">2026 Wheat Outlook</div>
                            <div class="naip-stat-sub">80% prediction interval: {crow['pred_lower_2026']:,.0f} &ndash; {crow['pred_upper_2026']:,.0f} kg/ha</div>
                            <div class="naip-stat-sub" style="color: var(--soil);">vs. 2019 actual: {crow['change_vs_2019']:+,.0f} kg/ha</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"In plain terms: based on the model's track record, this district's actual 2026 wheat "
                               f"yield is more likely than not to fall between {crow['pred_lower_2026']:,.0f} and "
                               f"{crow['pred_upper_2026']:,.0f} kg/ha &mdash; a real calibrated range, not a single guess.")

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
# TAB 5 — SCENARIO EXPLORER (Advanced / Testing — demoted)
# =================================================================
# Kept for development/demo flexibility, but explicitly demoted:
# this is a manual what-if tool on HISTORICAL years, not a forecast.
# For real forward-looking estimates use the Current Season Outlook
# (real 2026 weather) or Kharif 2026 Early Warning (real
# monsoon-so-far data) tabs instead.
# =================================================================

with tab_scenario:
    st.markdown("""
        <div class="naip-mode-banner">
            <b>Current-Season What-If Explorer.</b> This tool starts from the real, live conditions for 2026
            (actual NASA weather data, actual satellite NDVI, actual published yield trend) and lets you
            adjust rainfall and temperature to see how the model's prediction responds. This is genuinely useful
            for testing questions like "what if the monsoon undershoots further?" or "what happens to this
            district if we get a late cold spell?" &mdash; using real data as the baseline, not an arbitrary
            historical year.
        </div>
    """, unsafe_allow_html=True)

    if current_preds is None or len(current_preds) == 0:
        empty_state("🔮", "Current-season predictions not yet generated.<br>Run <code>src/27_generate_current_predictions.py</code> first.")
    else:
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            live_state = st.selectbox("State", sorted(current_preds['state'].unique()), key="scen_state")
        with form_col2:
            districts_in_state = sorted(current_preds[current_preds['state'] == live_state]['district'].unique())
            live_district = st.selectbox("District", districts_in_state, key="scen_district")

        crow_match = current_preds[current_preds['district'] == live_district]
        if len(crow_match) == 0:
            empty_state("🔍", f"No 2026 data available for {live_district}.")
        else:
            crow = crow_match.iloc[0]

            st.markdown('<div class="naip-eyebrow" style="margin-top:18px;">Adjust from real 2026 conditions</div>', unsafe_allow_html=True)
            st.caption(f"Baseline: actual NASA weather through {crow.get('data_as_of','2026-06-23')} · real NDVI · last published yield 2019. Sliders adjust forward from this real starting point.")

            wcol1, wcol2 = st.columns(2)
            with wcol1:
                rainfall_adj = st.slider("Rainfall change from current season (%)", -50, 50, 0,
                    help="Simulate if the rest of the season is wetter or drier than current conditions suggest")
            with wcol2:
                temp_adj = st.slider("Temperature adjustment (°C)", -3.0, 3.0, 0.0, step=0.5,
                    help="Simulate a hotter or cooler finish to the season than current readings")

            live_features = crow[FEATURES].copy().astype(float)
            rain_cols = [c for c in FEATURES if 'rainfall' in c or 'water_balance' in c]
            # Includes heat_stress_days / frost_risk_days, which are
            # temperature-derived features but don't contain 'temp' or
            # 'gdd' in their name - without this they'd stay frozen at
            # baseline while the temperature slider moves, making the
            # SHAP explanation inconsistent with the scenario applied.
            temp_cols = [c for c in FEATURES if 'temp' in c or 'gdd' in c or c in ('heat_stress_days', 'frost_risk_days')]
            for c in rain_cols:
                if c in live_features.index:
                    live_features[c] = live_features[c] * (1 + rainfall_adj / 100)
            for c in temp_cols:
                if c in live_features.index:
                    live_features[c] = live_features[c] + temp_adj

            X_live = live_features.to_frame().T
            live_pred = model.predict(X_live)[0]
            live_lower = xgb_lower_model.predict(X_live)[0]
            live_upper = xgb_upper_model.predict(X_live)[0]
            live_shap = explainer.shap_values(X_live)[0]

            baseline_pred = crow['current_pred_yield']
            delta = live_pred - baseline_pred

            st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
            result_col, explain_col = st.columns([1, 1.2])

            with result_col:
                st.markdown(f"""
                    <div class="naip-card">
                        <div class="naip-stat-number">{live_pred:,.0f} <span style="font-size:1rem; color:var(--soil); font-weight:500;">kg/ha</span></div>
                        <div class="naip-stat-label">Scenario yield &mdash; {live_district}, Wheat 2026</div>
                        <div class="naip-stat-sub">80% range: {live_lower:,.0f} &ndash; {live_upper:,.0f} kg/ha</div>
                        <div class="naip-stat-sub" style="color:var(--soil);">vs. current baseline: {delta:+,.0f} kg/ha</div>
                    </div>
                """, unsafe_allow_html=True)
                st.caption(f"In plain terms: if rainfall runs {rainfall_adj:+d}% from current levels and temperature shifts {temp_adj:+.1f}°C, this district's likely wheat yield range shifts to {live_lower:,.0f}–{live_upper:,.0f} kg/ha.")

                if rainfall_adj != 0 or temp_adj != 0:
                    st.markdown(f'<div class="naip-scenario"><span class="naip-scenario-title">Scenario Active</span>Rainfall {rainfall_adj:+d}%, Temperature {temp_adj:+.1f}&deg;C from real 2026 baseline.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="naip-safe"><span class="naip-safe-title">Baseline &mdash; Real 2026 Conditions</span>No adjustments applied. Showing the model\'s current-season prediction unchanged.</div>', unsafe_allow_html=True)

            with explain_col:
                st.markdown('<div class="naip-eyebrow">Why this prediction</div>', unsafe_allow_html=True)
                shap_df_live = pd.DataFrame({'feature': FEATURES, 'shap_value': live_shap})
                shap_df_live['abs_val'] = shap_df_live['shap_value'].abs()
                shap_df_live = shap_df_live.sort_values('abs_val', ascending=False).head(6)
                render_shap_bars(shap_df_live, shap_df_live['abs_val'].max())

            st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="naip-eyebrow">Recommended Actions for This Scenario</div>', unsafe_allow_html=True)

            advisory_row = crow.copy()
            advisory_row['drought_flag'] = 1 if rainfall_adj <= -25 else crow.get('drought_flag', 0)
            advisory_row['flood_flag'] = 1 if rainfall_adj >= 25 else crow.get('flood_flag', 0)
            render_advisory(generate_advisory(advisory_row, live_lower, live_upper, live_pred))

            alert_type_scen = 'drought' if advisory_row.get('drought_flag', 0) == 1 else 'flood' if advisory_row.get('flood_flag', 0) == 1 else 'general'
            render_live_context(live_state, live_district, 'Wheat', alert_type_scen)

# =================================================================
# TAB 6 — ABOUT
# =================================================================

with tab_about:
    st.markdown("""
        <div class="naip-mode-banner">
            <b>ANNA — Agricultural Nowcasting and Nearly-live Analytics</b><br>
            <i>अन्न (Anna)</i> — Sanskrit for "food" or "grain". The Taittiriya Upanishad declares
            <i>Annam Brahma</i> — "food is divine". A system that predicts food availability at the
            district level, named after the oldest Sanskrit word for crop itself.
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="naip-eyebrow">What ANNA does</div>', unsafe_allow_html=True)
    st.markdown("""
ANNA uses a machine learning model (tuned XGBoost, R²=0.8592 on a held-out chronological test set) to predict
wheat and rice yields at the district level, one season ahead. It combines four live and historical data sources:

- **NASA POWER satellite weather** — daily rainfall, temperature, humidity, and solar radiation, updated within ~3 days of real time. Engineered into agronomic features: GDD, dry-spell duration, water balance, rainfall concentration.
- **MODIS satellite NDVI** — a direct measure of vegetation health from space (250m, 16-day), via Google Earth Engine
- **JRC Global Surface Water** — surface water body and reservoir availability per district (30m, annual), via Google Earth Engine
- **Government crop yield records** — 22 years (1997–2019) of district-level published data used to train the model
    """)

    st.markdown('<div class="naip-eyebrow">How to use each tab</div>', unsafe_allow_html=True)
    st.markdown("""
- **Districts Requiring Attention** — your starting point. Shows which districts have meaningfully changed since the last check, ranked by severity. No input needed.
- **Kharif 2026 Early Warning** — an honest risk flag (not a yield number) for the current monsoon season, based on real rainfall-so-far vs. each district's own history.
- **Current Season Outlook** — ANNA's live Wheat 2026 prediction for every district, with a calibrated 80% uncertainty range.
- **Historical Results** — model predictions vs. actual government yields for 2018–19. This is the proof ANNA works, not a current forecast.
- **Scenario Explorer** — adjust rainfall and temperature from the real 2026 baseline to test what-if questions before making procurement or insurance decisions.
    """)

    st.markdown('<div class="naip-eyebrow">Honest limitations</div>', unsafe_allow_html=True)
    st.markdown("""
- **Yield trend data lags by ~5 years.** Government district-level statistics are published with a 1–2 year delay. ANNA's trend signal refers to 2019, not 2025. This is a universal constraint shared by all published systems in India.
- **Model accuracy varies by year.** Walk-forward validation shows R² ranging from 0.19 to 0.93 across years — the headline R²=0.8592 is from a chronological 2018–19 test set. Some years are harder to predict than others.
- **Interval calibration degrades at high predicted yields.** For the top 20% of predictions, the 80% interval captures only ~63% of actual outcomes. Verify high-yield predictions with field data before acting on them.
- **Rice yield prediction is unavailable mid-season.** A yield number for rice before harvest would be overclaiming. The Kharif tab provides an honest risk classification instead.
- **Surface water uses 2021 as current baseline.** JRC dataset has no 2022+ coverage yet.
    """)

    st.markdown('<div class="naip-eyebrow">Data & credits</div>', unsafe_allow_html=True)
    st.markdown("""
- Weather: NASA POWER (power.larc.nasa.gov)
- Satellite NDVI: NASA MODIS MOD13Q1 via Google Earth Engine
- Surface water: JRC Global Surface Water (Pekel et al. 2016, Nature) via GEE
- Groundwater: Kuruva et al. (2025), Nature Scientific Data, DOI: 10.1038/s41597-025-05899-5
- Crop yield training data: Government of India, Ministry of Agriculture (1997–2019)
- Built with: Python · XGBoost · SHAP · Streamlit · FastAPI · Folium
    """)

# ---------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------

st.markdown('<div class="naip-divider"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="naip-meta">
        ANNA v2.0 &middot; Agricultural Nowcasting and Nearly-live Analytics &middot;
        Model: XGBoost (tuned, 40 features) &middot; R&sup2;: 0.8592 (2018&ndash;19 test set) &middot;
        Walk-forward CV: 0.19&ndash;0.93 &middot; Interval coverage: 82.1%<br>
        Data: NASA POWER &middot; MODIS NDVI (GEE) &middot; JRC Surface Water (GEE) &middot;
        India Agriculture Crop Production &middot; Kuruva et al. 2025 (Nature Scientific Data)
    </div>
""", unsafe_allow_html=True)