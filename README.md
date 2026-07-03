# NAIP — National Agricultural Intelligence Platform

> District-level crop yield prediction and decision-support system for agricultural officers across **Haryana**, **Punjab**, and **Uttar Pradesh**, India.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![XGBoost 2.1.1](https://img.shields.io/badge/model-XGBoost%202.1.1-orange.svg)](https://xgboost.readthedocs.io/)
[![NASA POWER](https://img.shields.io/badge/weather-NASA%20POWER-green.svg)](https://power.larc.nasa.gov/)
[![Google Earth Engine](https://img.shields.io/badge/satellite-GEE%20MODIS%20%2B%20JRC-lightgreen.svg)](https://earthengine.google.com/)
[![FastAPI](https://img.shields.io/badge/api-FastAPI-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## What This Is

NAIP is a **production-grade agricultural intelligence system** that predicts Wheat and Rice yields at the district level, one season ahead, and delivers those predictions to district agricultural officers through:

- A **live 6-tab Streamlit dashboard** with season-aware early warning, SHAP explanations in officer language, and a real-time District Watch alert feed
- A **REST API** (FastAPI, 5 endpoints, auto-generated Swagger docs) consumable by any MIS/ERP system
- An **automated daily refresh pipeline** (8-step orchestrator, Windows Task Scheduler ready)
- A **validated sanity gate** that caught 5 real bugs during development and prevents regressions in production

It is not a research prototype — it is a running system with real live data, a documented audit trail of bugs found and fixed, and a production deployment pipeline.

**Live outputs (July 2026):**
- 118 districts monitored across 3 states
- Mean Wheat 2026 outlook: **4,304 kg/ha** (+244 kg/ha vs. 2019)
- Kharif 2026 risk: **28 districts high risk**, 88 moderate (real June 2026 monsoon data, validated against IMD reporting)
- Last refresh: daily via Windows Task Scheduler

---

## Project Goals

This system was built to serve four goals simultaneously:

| Goal | Status |
|---|---|
| 4th-year B.Tech major project | ✅ Complete — working end-to-end system, public GitHub |
| Research paper | ⏳ In progress — all methodology documented and validated |
| Patent filing | ⏳ Draft started — time-sensitive given public repo; provisional filing recommended |
| Portfolio / demo | ✅ Complete — live dashboard, FastAPI, demoable in minutes |

---

## Architecture

```
DATA SOURCES                    PIPELINE (scripts 00–40)              OUTPUTS
────────────                    ────────────────────────              ───────
NASA POWER API (daily)  ──►  22 fetch current weather            Streamlit Dashboard (6 tabs)
                             26 fix fill values (-999 bug)        FastAPI REST API (5 endpoints)
MODIS NDVI (GEE)        ──►  34 fetch NDVI historical            District Watch alerts
                             35 fetch NDVI current (live)         Kharif 2026 risk flags
JRC Surface Water (GEE) ──►  36 fetch surface water              Walk-forward CV (12 folds)
                                                                   SHAP importance rankings
Govt Yield Records      ──►  15 aggregate weather                 Per-decile interval calibration
(1997–2019, 118 dist.)       16 merge full weather
                             17 feature engineering v2
Kuruva et al. 2025      ──►  37 load groundwater ─── ►  38 residual analysis (finding, not feature)
(Groundwater, Nature         
Scientific Data)         

Model: 19 tune xgboost → 20 finalize (SHAP + quantile intervals)

Automated:  00_run_full_refresh.py
            runs 22→26→25→35→27→28→32→33 daily
            logs to logs/refresh_YYYYMMDD_HHMMSS.log
```

---

## Key Results

| Metric | Value | Notes |
|---|---|---|
| Model | Tuned XGBoost | 40 features, `learning_rate=0.05`, `max_depth=8`, 600 trees |
| Test R² (2018–19) | **0.8592** | Chronological split — train ≤2015, valid 2016–17, test 2018–19 |
| Test RMSE | **343.9 kg/ha** | |
| Test MAPE | **8.2%** | Wheat: 6.7%, Rice: 8.7% |
| Walk-forward R² | **0.19 – 0.93** | 12 folds (2008–2019), mean 0.77 ± 0.21 |
| Interval coverage | **82.1%** | Target: 70–90%; uses quantile_alpha=0.02/0.98 |
| Wheat MAPE | 6.7% | |
| Rice MAPE | 8.7% | |

> **Walk-forward range is reported explicitly** (0.19–0.93, not just the headline 0.86). The 2014–2016 folds are consistently weaker and are documented in the dashboard's About tab and methodology section — not hidden behind the single-split number. The chronological validation directly contrasts with the random 80/20 splits common in published Indian crop yield papers, which can inflate reported accuracy via data leakage.

---

## Feature Set (40 features)

| Category | Features | Source | Completeness |
|---|---|---|---|
| Autoregressive | lag_yield_1/2/3, rolling_yield_3yr, time_trend, yield_trend, yield_gap_vs_state | Govt records | 90–100% |
| Live weather (seasonal) | rainfall, temp, humidity, solar (Kharif + Rabi + Annual) | NASA POWER | 100% |
| Agronomic (engineered) | GDD_kharif/rabi, max_dry_streak_kharif/rabi, rainfall_cv_kharif/rabi, water_balance_kharif/rabi | Computed from NASA POWER daily | 100% |
| Drought/flood derived | drought_flag, flood_flag, rainfall_anomaly_index, heat_stress_days, frost_risk_days | Derived | 100% |
| NDVI (satellite) | ndvi_kharif, ndvi_rabi | MODIS MOD13Q1 via GEE | 91.4% (pre-2000: NaN expected) |
| Surface water | water_pct_kharif, water_pct_rabi | JRC Global Surface Water via GEE | 100% (1997–2021) |
| Management | fertilizer_per_ha, nitrogen_per_ha, irrigation_pct, log_area | ICRISAT/Mendeley | 63–65% (documented) |
| Structural | state_encoded, crop_encoded, season_encoded | Derived | 100% |

**Why groundwater is NOT a model feature:** Kuruva et al. (2025, Nature Scientific Data) provides quality-controlled groundwater depth for 28/118 districts (17% coverage). At this density, adding it as a feature risks selection bias — the 28 districts with data are not a random sample, they are the better-monitored ones. We instead ran a residual-correlation analysis (`src/38_groundwater_analysis.py`): Pearson r=0.223, p=0.294 — groundwater decline rate is not independently significant beyond what the existing weather/NDVI features already capture. This is itself a publishable finding: the multi-source feature design adequately proxies water availability without direct water-table data.

---

## Real Bugs Found and Fixed

This project caught **5 real bugs** through manual diagnosis and an automated sanity gate. All 5 are now covered by the regression suite:

| # | Bug | Symptom | Root Cause | Fix |
|---|---|---|---|---|
| 1 | **Rabi season-window misalignment** | Weather SHAP share only 7.3% for a wheat-belt model | Jan–Mar assigned to the *wrong* crop-year (same calendar year rather than Nov(Y)–Mar(Y+1)) — harvest-period weather was missing entirely from the season being predicted | Switched to crop-year-correct aggregation; SHAP share roughly doubled |
| 2 | **Stale quantile models** | Dashboard silently serving incorrect prediction intervals | `xgb_lower.pkl`/`xgb_upper.pkl` were pre-v2 leftovers, feature-schema-mismatched; script 20 trained fresh models but never saved them | Added the two missing `pickle.dump()` calls in script 20 |
| 3 | **Kharif pacing bug** | 117/118 districts "high risk" simultaneously | Linear pacing of 21 days of early-June rainfall across a 153-day season, then comparing to a full-season average — structurally biased for any early-June window in any year | Switched to same-calendar-window comparison (June 1–21 actual vs. June 1–21 in 2020–2025); result independently verified against real news (Haryana -16%, Punjab -25%, June 2026) |
| 4 | **District Watch silent gap** | 98% "stable" classification with std=51.5 real variance | A 5–150 kg/ha change range had no category — fell through silently to "stable" | Added `minor_change` category; automated gate caught this via the classification-diversity check |
| 5 | **Windows pipe encoding crash** | Orchestrator stopped at step 2 with `cp1252` error | Python falls back to Windows codepage `cp1252` when output is captured via subprocess pipe; Unicode checkmarks (✓) crash it | Set `PYTHONIOENCODING=utf-8` in the orchestrator for all subprocess calls |

---

## Automated Validation

### Sanity Gate (`src/33_pipeline_sanity_checks.py`)
Runs after every pipeline refresh. 4 checks:
1. **Model/feature schema consistency** — all saved `.pkl` files must match `feature_list_v2.txt` exactly (caught bug #2)
2. **Classification diversity** — flags any risk/alert output where >85% of rows collapse to one category, with smart disambiguation of benign cases (first-run baseline, model-version-change) from real bugs (caught bug #3 and #4)
3. **Live feature plausibility** — flags if >50% of districts are simultaneously >3σ from historical range
4. **Required file presence** — all pipeline outputs must exist and be non-empty

### Regression Suite (`tests/test_pipeline.py`)
25 pytest tests across 5 test classes. Covers all 5 bugs by name:
```bash
pytest tests/test_pipeline.py -v   # All 25 should pass
```

---

## Project Structure

```
NAIP/
├── src/                              # 42 numbered pipeline scripts (00–40)
│   ├── 00_run_full_refresh.py        # Master orchestrator (daily via Task Scheduler)
│   ├── 14_fetch_weather_all_districts.py
│   ├── 15_aggregate_weather_all.py   # Season-window fix + GDD/dry-streak/water-balance
│   ├── 16_merge_full_weather_retrain.py
│   ├── 17_feature_engineering_v2.py
│   ├── 19_tune_xgboost.py
│   ├── 20_finalize_tuned_model.py    # SHAP + calibrated quantile intervals
│   ├── 22_fetch_current_weather.py   # NASA POWER live pull
│   ├── 25_current_conditions.py      # Live Rabi 2025-26 feature assembly
│   ├── 27_generate_current_predictions.py
│   ├── 28_district_watch.py          # Snapshot/diff alert engine
│   ├── 32_kharif_early_warning.py    # Same-window risk classification
│   ├── 33_pipeline_sanity_checks.py  # Automated validation gate
│   ├── 34_fetch_ndvi_historical.py   # MODIS NDVI backfill (GEE)
│   ├── 35_fetch_ndvi_current.py      # Live NDVI for current season
│   ├── 36_fetch_surface_water.py     # JRC surface water (GEE)
│   ├── 37_load_groundwater.py        # Kuruva et al. 2025 loader
│   ├── 38_groundwater_analysis.py    # Residual-correlation finding
│   ├── 40_model_validation.py        # Walk-forward CV + residuals + calibration
│   └── setup_tavily.py               # One-time Tavily API key setup
├── dashboard/
│   └── app.py                        # Streamlit app (1,428 lines, 6 tabs)
├── api/
│   └── naip_api.py                   # FastAPI REST API (5 endpoints)
├── tests/
│   └── test_pipeline.py              # 25 pytest regression tests
├── data/
│   ├── raw/                          # NASA POWER, groundwater exports
│   ├── processed/                    # Merged, engineered datasets
│   ├── snapshots/                    # District Watch time-series
│   └── secrets/                      # API keys (gitignored, never committed)
├── outputs/
│   ├── models/                       # xgb_tuned.pkl, xgb_lower.pkl, xgb_upper.pkl
│   ├── metrics/                      # SHAP rankings, walk-forward CV, residuals
│   └── plots/
├── logs/                             # Refresh logs (timestamped)
├── run_refresh.bat                   # Windows Task Scheduler entry point
└── requirements.txt
```

---

## Setup

### Prerequisites
- Anaconda / conda
- Google Earth Engine account (free, [signup here](https://earthengine.google.com/)) — for NDVI + surface water
- Tavily API key (optional, free 1000/month, [signup here](https://app.tavily.com/home)) — for live government scheme / MSP context

### Installation

```bash
# 1. Create and activate conda environment
conda create -n naip python=3.10
conda activate naip
pip install -r requirements.txt

# 2. One-time Earth Engine authentication (opens browser)
python src/34_fetch_ndvi_historical.py

# 3. Optional: set up Tavily for live policy context
python src/setup_tavily.py
```

### Historical pipeline (one-time, ~2–3 hours total)

```bash
python src/14_fetch_weather_all_districts.py   # ~4 min (118 districts × 1.5s)
python src/15_aggregate_weather_all.py          # ~2 min
python src/16_merge_full_weather_retrain.py
python src/17_feature_engineering_v2.py
python src/34_fetch_ndvi_historical.py          # ~5 min (GEE)
python src/36_fetch_surface_water.py            # ~15 min (GEE, 1997–2021)
python src/16_merge_full_weather_retrain.py     # re-run with satellite features
python src/17_feature_engineering_v2.py         # re-run
python src/19_tune_xgboost.py                   # ~10 min (RandomizedSearchCV)
python src/20_finalize_tuned_model.py
python src/40_model_validation.py               # Walk-forward CV (~15 min)
```

### Optional: groundwater data (adds residual-analysis finding)

```
1. Download CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv from:
   https://doi.org/10.6084/m9.figshare.29293877.v3
   (Kuruva et al. 2025, Nature Scientific Data — free, no login)
2. Save to: data/raw/groundwater/
3. python src/37_load_groundwater.py
4. python src/38_groundwater_analysis.py
```

### Daily live refresh

```bash
# Manual
python src/00_run_full_refresh.py

# Automated (Windows Task Scheduler → run_refresh.bat)
# Runs: 22 → 26 → 25 → 35 → 27 → 28 → 32 → 33
# Logs to: logs/refresh_YYYYMMDD_HHMMSS.log
```

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

### Launch API

```bash
uvicorn api.naip_api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Run tests

```bash
pytest tests/test_pipeline.py -v
# Expected: 25 passed
```

---

## Dashboard Tabs

| Tab | What it shows |
|---|---|
| 🚨 Districts Requiring Attention | District Watch homepage — zero-click, ranked alert feed, drill-down per district |
| 🌱 Kharif 2026 Early Warning | Risk classification by monsoon-so-far rainfall, season-aware, IMD context displayed separately |
| 🌤️ Current Season Outlook | Live Wheat 2026 predictions with calibrated 80% interval, live NDVI and surface water features |
| 📊 Historical Results | Validated 2018–19 predictions vs. actual yields — proof the model works |
| 🔮 Scenario Explorer | Real 2026 baseline what-if (rainfall/temperature sliders from live starting point) |
| ℹ️ About | System description, tab guide, honest limitations, data citations |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | System status, 40-feature schema check, data freshness |
| `GET /predict/{district}` | Full prediction: yield, interval, top SHAP factors, drought/flood flags |
| `GET /predictions` | All 118 districts, filterable by state / drought-only / sort order |
| `GET /watch` | District Watch alert feed, filterable by alert level |
| `GET /kharif-risk` | Kharif 2026 early-warning risk classification |
| `GET /districts` | All monitored districts |

Interactive Swagger UI auto-generated at `/docs`.

---

## RAG / Live Grounding Layer

When a district shows an alert, the system queries the live web (Tavily) in a prioritized order:
1. Government relief schemes (state-specific, alert-type-specific)
2. MSP (Minimum Support Price) for the crop
3. IMD weather outlook for the state

This layer is **enhancement-only** — its absence never breaks the prediction system. If no API key is configured, the dashboard shows a plain-language card (not a raw error). The layer is designed specifically to complement the model's backward-looking anomaly signal (RAI) with forward-looking official forecasts (IMD), a distinction documented as an architectural contribution.

**Validated real finding:** During development, this layer surfaced IMD's official 2026 Southwest Monsoon outlook (90% of LPA, 60% chance deficient, revised 29 May 2026) — independently verified via a second search and cross-referenced against news reports showing Haryana at −16%, Punjab at −25% deficit in June 2026, consistent with the Kharif early-warning system's district-level output.

---

## Honest Limitations

- **Yield trend data lags by ~5 years.** Government district-level statistics are published with a 1–2 year lag. The most recent published yield data is 2019. The model's trend signal ("last published yield") refers to 2019 even for 2026 predictions. This is a universal constraint of Indian agricultural statistics, shared by all published systems — stated explicitly, not hidden.
- **Walk-forward accuracy varies significantly by year** (R² 0.19 in 2014 vs. 0.93 in 2013). The headline 0.86 is from a relatively favorable test window. Both numbers are reported.
- **Prediction interval calibration degrades in the highest yield quintile** (63% coverage vs. 82% average). High-yield predictions should be cross-checked with field data before acting on them.
- **Rice (Kharif) yield is not predicted mid-season.** The Kharif tab is a risk classification, not a yield forecast. Predicting yield before harvest would be overclaiming.
- **Surface water uses 2021 as current baseline.** JRC Global Surface Water dataset covers only to 2021; 2022–2026 is approximated using the most recent available year.
- **Groundwater covers 28/118 districts only** (Kuruva et al. 2025). Excluded from model features to avoid selection bias; used for a residual-correlation finding instead.

---

## Data Sources & Citations

| Source | What it provides | Citation / URL |
|---|---|---|
| NASA POWER | Daily weather 1997–present | [power.larc.nasa.gov](https://power.larc.nasa.gov/) |
| MODIS MOD13Q1 | NDVI 2000–present | NASA/USGS via Google Earth Engine |
| JRC Global Surface Water | Surface water extent 1984–2021 | Pekel et al. (2016), *Nature* 491, 418–422. Via GEE. |
| India Agriculture Crop Production | Yield, area, production 1997–2019 | Government of India, Ministry of Agriculture |
| ICRISAT/Mendeley | Fertilizer, irrigation, evapotranspiration | ICRISAT Village Dynamics in South Asia |
| Groundwater levels | Depth-to-water-table, 2000–2022 | Kuruva, S.K. et al. (2025). *Scientific Data* 12, 1609. DOI: [10.1038/s41597-025-05899-5](https://doi.org/10.1038/s41597-025-05899-5) |
| Tavily | Live government/IMD retrieval | [app.tavily.com](https://app.tavily.com) |
| Nominatim/OSM | District geocoding | OpenStreetMap contributors |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built as a 4th-year B.Tech major project by Kunal. Full audit trail of bugs found and fixed documented in `NAIP_Full_Project_Report.docx` and inline in each script's docstring.*