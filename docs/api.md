# NAIP API Design

**Status date:** 2026-10-04
**Stack:** FastAPI (Python), Pydantic v2, containerised; reads PostgreSQL (see `database.md`).
**Base path:** `/api/v1`

Legend: ✅ Built · 🔧 Built, needs refactor · 📐 Designed, not built
Phase: **MVP** / **P2** / **P3** (see roadmap in `systemdesign.md`)

---

## 1. Current state (`api/naip_api.py`, ~230 lines) 🔧

| Endpoint | Notes / issues |
|---|---|
| `GET /health` | Schema check (40 features) + per-file freshness. Good base; extend with run/gate info. |
| `GET /districts` | Returns the district table. |
| `GET /predictions` | Filter by state, `drought_only`, `sort_by`. `sort_by` accepts any column (exposes internals, 500-prone). |
| `GET /predict/{district}` | Matches by district name only (names can repeat across states), returns the first row, hardcodes `change_vs_2019`. |
| `GET /watch` | Filter by alert level. |
| `GET /kharif-risk` | Filter by risk level. |

Other gaps: CORS is `*` and GET-only, no response schemas, no pagination, no versioning, no crop dimension (live data is wheat only), no history/SHAP/environment/pipeline endpoints, no scenario endpoint.

---

## 2. Conventions

### 2.1 Versioning and shape
- All routes under `/api/v1`. Breaking changes go to `/api/v2`; additive fields do not bump the version.
- District identity is the **state + district slug** (`/districts/uttar-pradesh/kheri`), not the bare name.
- Every success response uses one envelope:

```json
{
  "data": { },
  "meta": {
    "run_id": "b1c2…",
    "weather_data_as_of": "2026-09-29",
    "model_version": "xgb_tuned_2026-07",
    "prediction_basis": "current_2026_weather_with_2019_yield_baseline",
    "baseline_yield_year": 2019,
    "gate_passed": true,
    "generated_at": "2026-10-04T09:30:00Z"
  }
}
```

`meta` is on **every** response that serves run-derived data. The 2019 yield-lag baseline and the data age are part of the result, not footnotes.

- Lists add `"page": {"limit": 50, "offset": 0, "total": 118}`. Default limit 50, max 200 (the full 118 districts fit in one request for map views via `limit=200`).
- Sorting uses an **allow-list** per endpoint (`sort=pred_yield`, `order=desc`), never raw column names.

### 2.2 Errors
Consistent body, correct status codes:

```json
{ "error": { "code": "DISTRICT_NOT_FOUND", "message": "…", "details": {} } }
```

`400` validation · `404` not found · `409` no promoted run yet · `422` schema violation · `429` rate limited · `503` database unavailable.

### 2.3 Caching
- Read endpoints send `ETag` derived from `run_id` + query, and `Cache-Control: public, s-maxage=300, stale-while-revalidate=3600`. Data only changes when a new run is promoted, so caching is safe and keeps the free-tier API quiet.
- Scenario endpoints are `POST` and uncached.

### 2.4 Security and limits
- Read-only API: the API database role has SELECT only. There are **no write endpoints**; writes happen in the refresh job.
- CORS: allow only the production and preview frontend origins.
- Rate limit (per IP): reads generous; scenario endpoints strict (initial: 30/min predict, 6/min for grid/map). In-memory limiter is sufficient for a single instance.
- No auth for v1 (public research demo). Revisit only if abuse appears.

### 2.5 Docs
OpenAPI is generated from Pydantic models and served at `/docs`; the frontend's TypeScript types are generated from the same schema in CI so the two cannot drift.

---

## 3. Endpoint catalogue (target) 📐

### System
| Method | Path | Purpose | Phase |
|---|---|---|---|
| GET | `/health` | Liveness, schema check, DB reachability, freshness, last gate result | MVP |
| GET | `/meta` | Latest run, model version, feature schema, known baselines/limitations | MVP |

### Reference
| GET | `/districts` | All districts (state, name, slug, lat/lon) | MVP |
|---|---|---|---|
| GET | `/districts/{state}/{district}` | One district + which crops/seasons have data | MVP |

### Predictions and district detail
| Method | Path | Purpose | Phase |
|---|---|---|---|
| GET | `/predictions?crop=&state=&sort=&order=&limit=&offset=` | Latest predictions for the map and ranking | MVP |
| GET | `/districts/{state}/{district}/prediction?crop=` | Yield, interval, flags, baseline info | MVP |
| GET | `/districts/{state}/{district}/shap?crop=` | Full SHAP vector for the waterfall chart | MVP |
| GET | `/districts/{state}/{district}/history?crop=` | Actual yield vs model predictions by year, with interval | MVP |
| GET | `/districts/{state}/{district}/environment?metric=&season=` | Rainfall, temperature, NDVI, surface water, groundwater series | MVP |

### Alerts
| Method | Path | Purpose | Phase |
|---|---|---|---|
| GET | `/watch?alert_level=&crop=` | District Watch feed | MVP |
| GET | `/districts/{state}/{district}/watch` | What moved since the last run (feature-level deltas) | P2 |
| GET | `/kharif-risk?risk_level=` | Kharif early-warning classification | MVP |
| GET | `/districts/{state}/{district}/kharif` | Season-to-date rainfall vs historical window | MVP |

### Compare
| GET | `/compare?districts=a/b,c/d&crop=` | Side-by-side summary for 2–4 districts | P2 |
|---|---|---|---|

### Pipeline health and methodology
| Method | Path | Purpose | Phase |
|---|---|---|---|
| GET | `/pipeline/runs?limit=` | Run history with status and gate result | P2 |
| GET | `/pipeline/runs/{run_id}` | Sanity-check detail for one run | P2 |
| GET | `/pipeline/drift?run_id=` | Per-feature drift status | P2 |
| GET | `/pipeline/incidents` | Documented incidents and failure taxonomy | P2 |
| GET | `/model/metrics` | Test, walk-forward CV, calibration summary | P3 |
| GET | `/model/residuals?crop=` | Held-out predictions for residual/calibration charts | P3 |

### Export
| GET | `/export/predictions?format=csv|json` | Latest predictions with `meta` as header rows/fields | P3 |
|---|---|---|---|

### Scenario engine
| Method | Path | Purpose | Phase |
|---|---|---|---|
| POST | `/scenarios/predict` | Baseline vs scenario for one district | P2 |
| POST | `/scenarios/sweep` | Response curve along one driver | P2 |
| POST | `/scenarios/grid` | 2D grid (e.g. rainfall × temperature) for a heatmap | P2 |
| POST | `/scenarios/sensitivity` | One-at-a-time sensitivity (tornado) | P2 |
| POST | `/scenarios/map` | One scenario applied to all districts | P2 |

---

## 4. Selected read contracts

### `GET /predictions?crop=Wheat&limit=200`
```json
{
  "data": [
    {
      "state": "Uttar Pradesh", "district": "Kheri", "slug": "uttar-pradesh/kheri",
      "crop": "Wheat", "season": "Rabi",
      "pred_yield_kg_ha": 3965.7, "pred_lower": 3412.0, "pred_upper": 4480.0,
      "drought_flag": false, "flood_flag": false,
      "rainfall_anomaly_index": 4.41,
      "change_vs_baseline_kg_ha": -266.3
    }
  ],
  "page": { "limit": 200, "offset": 0, "total": 118 },
  "meta": { "…": "…" }
}
```
(Illustrative values. `change_vs_baseline` uses `baseline_yield_year` from `meta` instead of a hardcoded "2019".)

Today only **Wheat (Rabi)** has live predictions; `crop=Rice` returns an empty list with a `meta.notice` until a live Kharif step exists.

### `GET /districts/{state}/{district}/shap?crop=Wheat`
Returns every feature's SHAP value and its input value, so the UI can draw a waterfall and show the feature's actual value next to its contribution.

### `GET /districts/{state}/{district}/kharif`
Returns season-to-date rainfall, days elapsed, historical window mean/std/n_years, z-score, risk level, explanation. A full daily cumulative-vs-envelope series is **not available today** (only window totals are stored); the chart is either a window-level comparison (MVP) or backed by a stored daily cumulative series (see Open items).

---

## 5. Scenario engine

### 5.1 What it does, and what it does not
It answers: *"Holding everything else at this district's baseline, how does the model's prediction change if the season's weather or inputs were different?"*

It is **model sensitivity, not a causal forecast.** The model learned associations from 1998–2019 yield data. The API returns this framing in every scenario response (`meta.interpretation`), and the UI must show it.

### 5.2 Design: perturb daily weather, recompute with pipeline code
Your derived features (`gdd_*`, `max_dry_streak_*`, `rainfall_cv_*`, `water_balance_*`, `heat_stress_days`, `frost_risk_days`, RAI and the drought/flood flags) come from **daily** weather. Editing the seasonal aggregates directly produces feature combinations that cannot occur and breaks consistency (this is the failure of the current explorer).

So the engine:
1. Loads the baseline daily series for the district/season window (`daily_weather_season`, see `database.md`).
2. Applies **perturbation operators** to the daily series.
3. Recomputes every dependent feature using the **same functions as the pipeline** (extracted to a shared module, `pipeline/features/weather.py`, imported by both the pipeline and the API).
4. Builds the 40-feature vector, scores it with `xgb_tuned` (point) and `xgb_lower`/`xgb_upper` (interval), and computes SHAP for baseline and scenario.

One shared function guarantees that "no change" reproduces the stored baseline.

### 5.3 Models
`xgb_tuned` for **both crops**, consistent with the live predictions (crop is a model feature). Interval from `xgb_lower`/`xgb_upper` (quantiles 0.02 / 0.98). Report coverage using the validated empirical figure from the Methodology page, not a nominal "95%".

### 5.4 Baselines
| Mode | Meaning | Availability |
|---|---|---|
| `current` | The latest promoted run's input row (current weather + yield-lag baseline) | Wheat now; Rice once a live Kharif step exists |
| `replay` + `year` | A district's actual historical features for that year | Wheat and Rice, 1998–2019 |

`replay` is the most defensible mode: every feature combination was observed. **Rice scenarios use `replay` in v1.**

### 5.5 Perturbation operators (applied in this order)
| Operator | Parameter | Effect on daily series | Initial bounds |
|---|---|---|---|
| `rain_scale_pct` | percent | Multiplies daily rainfall | −60 … +100 |
| `dry_spell` | `days`, `start_offset_days` | Zeroes rainfall for a run of days | days 0 … 60 |
| `temp_shift_c` | °C | Adds to `tmax`, `tmin`, `tavg` | −3 … +5 |
| `heatwave` | `days`, `delta_c` | Adds `delta_c` to `tmax`/`tavg` on N days | days 0 … 30, Δ 0 … 8 |
| `irrigation_pct_delta` | points | Direct change to `irrigation_pct` (clamped 0–100) | −30 … +30 |
| `nitrogen_pct_delta` | percent | Direct change to `nitrogen_per_ha`/`fertilizer_per_ha` | −50 … +50 |

Bounds are initial and tunable. `rain_scale_pct` alone leaves dry streak and CV unchanged (uniform scaling), which is why `dry_spell` exists as a separate operator.

### 5.6 Feature recompute rules
| Features | Rule | Source |
|---|---|---|
| `nasa_rainfall_{season}` | Sum of daily rainfall in the window | Recomputed |
| `nasa_rainfall_annual` | Annual sum with the season window replaced | Recomputed |
| `nasa_temp_avg_*`, `nasa_temp_max_*` | Per the pipeline definition | Recomputed |
| `gdd_{season}` | Σ max(`tavg` − 10, 0) | Recomputed |
| `max_dry_streak_{season}` | Longest run of days with rain < 1 mm | Recomputed |
| `rainfall_cv_{season}` | std / mean of daily rainfall | Recomputed |
| `water_balance_{season}` | Σ rain − Σ ET0 (simplified Hargreaves, uses solar, tmax, tmin, tavg) | Recomputed |
| `heat_stress_days` | Days with `tmax` > 35 °C | Recomputed |
| `frost_risk_days` | Per the pipeline definition | Recomputed |
| `rainfall_anomaly_index` | Season rainfall vs the district's historical mean/std | Recomputed |
| `drought_flag` / `flood_flag` | RAI < −1 / RAI > +1 | Recomputed |
| `irrigation_pct`, `nitrogen_per_ha`, `fertilizer_per_ha` | Direct adjustment | Operator |
| `nasa_humidity_kharif`, `nasa_solar_annual` | Held at baseline (not perturbed in v1) | Held |
| `lag_yield_*`, `rolling_yield_3yr`, `yield_trend`, `yield_gap_vs_state`, `time_trend`, `log_area`, encodings | Held at baseline | Held |
| `ndvi_*`, `water_pct_*` | Held at baseline by default | Held |

> The exact season window and the `nasa_temp_*`/`frost_risk_days` definitions must be taken from `src/15_aggregate_weather_all.py` and `src/25_current_conditions.py` when extracting the shared module; this table is the contract, not a substitute for reading that code.

### 5.7 Known limitations the API must surface
- **Satellite features are held fixed.** NDVI and surface water respond to weather in reality; holding them at baseline can understate a scenario's effect. Every response includes `held_fixed: [...]`. An option to exclude them is an open question (below).
- **Lag features carry much of the signal.** In the one live row inspected (Kheri), `lag_yield_1` has a far larger SHAP value than any weather feature, so weather scenarios may move predictions modestly. The UI should show this rather than hide it.
- **2019 yield-lag baseline** applies to every `current`-mode scenario.
- **Extrapolation.** Tree models do not extrapolate; see `warnings` below.

### 5.8 `POST /scenarios/predict`

Request:
```json
{
  "district": "uttar-pradesh/kheri",
  "crop": "Wheat",
  "baseline": { "mode": "current" },
  "adjustments": {
    "rain_scale_pct": -20,
    "temp_shift_c": 1.5,
    "heatwave": { "days": 5, "delta_c": 4 },
    "dry_spell": { "days": 14, "start_offset_days": 40 },
    "irrigation_pct_delta": 0,
    "nitrogen_pct_delta": 0
  }
}
```

Response (shape; values omitted):
```json
{
  "data": {
    "baseline": { "pred": 0, "lower": 0, "upper": 0 },
    "scenario": { "pred": 0, "lower": 0, "upper": 0 },
    "delta": { "abs_kg_ha": 0, "pct": 0 },
    "features_changed": [ { "feature": "gdd_rabi", "baseline": 0, "scenario": 0 } ],
    "shap_delta": [ { "feature": "heat_stress_days", "delta": 0 } ],
    "held_fixed": ["ndvi_rabi", "water_pct_rabi", "lag_yield_1"],
    "warnings": [
      { "code": "OUT_OF_RANGE", "features": ["heat_stress_days"],
        "message": "Scenario value exceeds this district's observed range (1998–2019); prediction is an extrapolation." }
    ]
  },
  "meta": { "interpretation": "model sensitivity, not a causal forecast", "…": "…" }
}
```

### 5.9 Other scenario endpoints
- `POST /scenarios/sweep` — `{district, crop, baseline, adjustments (base), vary: {param, from, to, steps}}` → points of `{x, pred, lower, upper}` for the response curve. `steps` ≤ 41.
- `POST /scenarios/grid` — two varied parameters, ≤ 15×15 → matrix for the heatmap.
- `POST /scenarios/sensitivity` — runs each operator at ± a standard step, returns ranked deltas for the tornado chart.
- `POST /scenarios/map` — one adjustment set across all districts for a crop → per-district delta for the map. Returns the same `warnings` per district.

All are vectorised: one feature-recompute pass builds a batch, one `predict` call scores it.

### 5.10 Validity warnings
Returned, never silently applied:
- `OUT_OF_RANGE` — a recomputed feature is outside that district-crop's observed history.
- `OPERATOR_CLAMPED` — an input was clamped to its bound.
- `BASELINE_STALE` — `meta.weather_data_as_of` is older than a threshold.
- `NO_LIVE_BASELINE` — `mode=current` requested for a crop without live inputs (returns 409 instead of guessing).

---

## 6. Validation of the scenario engine (tests)

These double as research evidence for the paper:

1. **Identity.** With no adjustments and `replay` baseline, the recomputed feature vector equals the stored `feature_history.features` for the same year (within float tolerance). This proves the shared code matches training.
2. **Determinism.** Same request → identical response.
3. **Direction checks.** More `heat_stress_days` / `heatwave` should not increase predicted yield on average; a longer `dry_spell` should not. Report where the model disagrees rather than forcing agreement.
4. **Replay accuracy.** Replaying held-out years reproduces the stored held-out predictions.
5. **Bounds.** Out-of-bound inputs return 422 or `OPERATOR_CLAMPED`, never a 500.
6. **Performance.** `grid` (225 cells) and `map` (118 districts) complete within a target latency measured on the chosen free host (to be set after measurement).

---

## 7. Open items

1. **Kharif cumulative series.** For the early-warning chart, decide between window-level comparison only or storing a daily cumulative rainfall series per district (small).
2. **`daily_weather_season` table.** The scenario engine needs daily weather (~1M daily rows across 118 districts, 1997 onward). Store as per-district-season arrays to keep it compact; added to `database.md`.
3. **Live Kharif/Rice step.** Needed for `mode=current` on Rice; until then Rice uses `replay`.
4. **Satellite-hold toggle.** Whether to offer a mode that excludes NDVI/water features from the scenario.
5. **Interval label.** Confirm the validated empirical coverage of the 0.02–0.98 quantile models before labelling the band in the UI.
6. **Shared feature module extraction** from `src/15` and `src/25` (first refactor task for the scenario work).