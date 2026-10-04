# NAIP System Design

**Status date:** 2026-10-04
**Companion to:** `system-architecture.md` (what the parts are) · this document covers *how we get there*: repo restructure, pipeline design, scenario internals, the live rice step, testing and the phased roadmap.

Legend: ✅ Built · 🔧 Built, needs refactor · 📐 Designed, not built

---

## 1. Design goals

1. **One feature code path.** Training, live prediction and the scenario engine all compute features with the same functions.
2. **Safe refreshes.** A bad run can never reach users (promotion gate).
3. **Organised repo.** Anyone can find the stage a script belongs to and run it from one entry point.
4. **Honest outputs.** Baselines, data age and model limits travel with every result.
5. **Zero cost.** Free tiers only.

---

## 2. Repo restructure

### 2.1 Target layout

```
naip/
├─ pipeline/
│  ├─ ingest/        weather (NASA POWER), ndvi (Earth Engine), water, groundwater, yield
│  ├─ features/      weather.py (shared), yield_lags.py, build_live.py
│  ├─ train/         train.py, tune.py, quantile.py, evaluate.py
│  ├─ predict/       predict.py, shap_explain.py
│  ├─ alerts/        watch.py, kharif_risk.py
│  ├─ validate/      sanity_gate.py, drift.py, model_validation.py
│  ├─ db/            migrations/, seed_reference.py, write_run.py
│  └─ run_refresh.py   (orchestrator; replaces src/00)
├─ api/              app/ (routers, schemas, services, scenario/)
├─ web/              Next.js frontend
├─ models/           serving models only (xgb_tuned, xgb_lower, xgb_upper)
├─ data/reference/   small committed reference data (see database.md)
├─ config/           pipeline_overrides.json, settings (paths, windows, thresholds)
├─ tests/
├─ docs/
├─ notebooks/        figures + exploratory work
├─ archive/          retired scripts and data versions (not imported by anything)
└─ .github/workflows/
```

### 2.2 Proposed mapping of existing scripts 🔧

This is a **proposal based on filenames and the orchestrator**, not a dependency analysis. Before moving any file, run `grep -rn "<name>" src api dashboard tests` to confirm nothing live imports or calls it.

| Proposed home | Existing scripts |
|---|---|
| `ingest/` | `01_fetch_weather`, `02_fetch_yield`, `07_get_district_list`, `08_fix_and_geocode`, `09_fix_failed_geocodes`, `14_fetch_weather_all_districts`, `22_fetch_current_weather`, `34_fetch_ndvi_historical`, `35_fetch_ndvi_current`, `36_fetch_surface_water`, `37_load_groundwater` |
| `features/` | `04_clean_data`, `10`/`11` (season duplicates), `12_merge_coordinates`, `13_final_clean_master`, `15_aggregate_weather_all`, `16_merge_full_weather_retrain`, `17_feature_engineering_v2`, `25_current_conditions`, `26_fix_fill_values` |
| `train/` | `18_train_model_v2`, `19_tune_xgboost`, `20_finalize_tuned_model`; `21_crop_specific_models` only if kept |
| `predict/` | `27_generate_current_predictions`, `rag_grounding_31` (advisory grounding) |
| `alerts/` | `28_district_watch`, `32_kharif_early_warning` |
| `validate/` | `33_pipeline_sanity_checks`, `23`/`24` (drift), `40_model_validation`, `41_baseline_comparison`, `42_lagged_flood_feature_test`, `38_groundwater_analysis` |
| `archive/` | `03_explore_data`, `05_feature_engineering`, `06_train_model`, `29_test_second_snapshot`, `30_debug_watch_merge`, `debug_district_coords_merge`, `33_verify_kharif_risk`*, `setup_tavily`, root `diagnose_kharif.py`, `check_days.py` |

\* Two scripts are numbered 33 (`33_pipeline_sanity_checks`, `33_verify_kharif_risk`); only the first is in the orchestrator.

### 2.3 How to refactor safely (VS Code workflow)
- One branch per batch (`refactor/ingest`, `refactor/features`, …); small commits using `git mv` so history follows the files.
- After each batch: `pytest` and one full refresh run. The sanity gate is the regression check.
- Keep `src/` entry points working until the orchestrator imports from `pipeline/`; then delete `src/` in one commit.
- Replace the numeric prefixes with stage names. The execution order lives in `run_refresh.py`, not in filenames.
- Add `ruff` (lint + format) and `pre-commit` once the layout settles.

### 2.4 Hygiene fixes found in the repo
| Issue | Fix |
|---|---|
| `run_refresh.bat` hardcodes `C:\Users\Kunal\...` and conda | Superseded by GitHub Actions; keep a local `make refresh` / script for development |
| `src/35` calls interactive `ee.Authenticate()` and hardcodes a GCP project id | Service-account init from an env var; project id in config |
| `data/secrets/tavily_key.txt` stored in the project folder | Env var / GitHub Secret; delete the folder |
| Many dataset versions (`master_dataset` v1–v5, old splits) | Archive after the grep check |
| `outputs/models` holds 8 pickles; 3 are used live | Keep 3 in `models/`, archive the rest |
| `.gitignore` excludes all models | Whitelist the 3 serving models |
| Committed `__pycache__` in `src/`, `api/`, `tests/` | Already gitignored; `git rm --cached` if tracked |

---

## 3. Pipeline design

### 3.1 Stages and ordering (kept from `src/00`)
```
fetch weather → fix fill values → build current-season features → fetch NDVI →
predict (+SHAP) → district watch diff → Kharif early warning → sanity gate → persist/promote
```
Fail-stop behaviour stays: if a stage fails, later stages do not run. The sanity gate stays last and informational for the orchestrator, but now decides **promotion** (see `database.md` §8).

### 3.2 Shared feature module
`pipeline/features/weather.py` is extracted from `src/15_aggregate_weather_all.py` and `src/25_current_conditions.py`, which today contain duplicated logic (GDD, dry streak, rainfall CV, water balance, RAI, flags). It exposes one function that takes a daily-weather block for a season window and returns the seasonal feature set.

It is imported by:
1. the training feature build,
2. the live feature build,
3. the scenario engine.

This removes the duplication that has already caused one train/serve mismatch (noted in the script 35 header) and guarantees the scenario "identity" test (`api.md` §6).

### 3.3 Run record
Each run records `run_id`, git SHA, model version, `weather_data_as_of`, per-check results and gate outcome (`database.md`). Logs are attached per run instead of accumulating `logs/refresh_*.log` files.

---

## 4. Live rice (Kharif) prediction — Phase 2 task 📐

**Goal:** give rice a real `current` baseline, like wheat, so Rice scenarios can run in `current` mode and the map/ranking can show rice.

**Current state:** `src/27` predicts only Wheat (Rabi). `src/25` handles Kharif only as partial-season rainfall (informational). `src/35` fetches NDVI for the Rabi window only and authenticates interactively. Kharif early warning (`src/32`) already exists and is a risk flag, not a yield prediction.

**Design:**
1. **Kharif window features.** Add a Kharif block to the live feature build, using the same window convention as training (Kharif is Jun–Oct per `src/25`) and the shared feature function. Produces `gdd_kharif`, `max_dry_streak_kharif`, `rainfall_cv_kharif`, `water_balance_kharif`, `heat_stress_days`, RAI and flags for the current year.
2. **Kharif NDVI.** Extend the live NDVI fetch to the Kharif window (and fix its auth, §2.4). Needed for `ndvi_kharif`.
3. **Surface water.** Use the Kharif live `water_pct` baseline (a current-baseline file exists; verify its coverage for the Kharif window).
4. **Generalise prediction.** Remove the Wheat-only filter in the predict stage; run both crops through `xgb_tuned`, with SHAP.
5. **Completeness gate.** Kharif is only predictable once the season window is complete. The step checks `weather_data_as_of` against the season end (allowing for the NASA POWER lag of a few days). While the season is incomplete, it writes **no** rice prediction and `/predictions?crop=Rice` returns a `meta.notice` ("season in progress"). The existing Kharif early-warning remains the in-season signal.
6. **Yield-lag baseline.** Rice lags come from the last known historical year, exactly like wheat (so the 2019 baseline caveat applies and is shown).
7. **Sanity checks for rice.** Extend the gate: prediction count (118), value range vs history, no NaN features, drift.
8. **Tests.** Identity test on a replayed historical Kharif year; a regression test that wheat output is unchanged.

**Dependencies:** shared feature module (§3.2), headless Earth Engine auth (§2.4), `database.md` run tables.

**Done when:** a scheduled run writes 118 rice and 118 wheat predictions in the same run, the gate passes, and the UI shows both.

---

## 5. Scenario engine internals (see `api.md` §5 for the contract)

```
request → validate (Pydantic, bounds)
        → load baseline (current row | replay year) + daily weather arrays
        → apply operators in fixed order (rain_scale → dry_spell → temp_shift → heatwave)
        → recompute features with pipeline/features/weather.py
        → apply direct operators (irrigation, nitrogen)
        → assemble 40-feature vector, check schema order against model_versions.feature_list
        → xgb_tuned / xgb_lower / xgb_upper predict (batched)
        → TreeExplainer SHAP for baseline and scenario → shap_delta
        → range check vs feature_history → warnings
```

Design points:
- **Stateless.** No scenario storage; the frontend keeps state in the URL (shareable).
- **Batched.** `sweep`, `grid`, `sensitivity` and `map` build all rows first, then call `predict` once.
- **Loaded once.** Models and the SHAP explainer load at API startup; reference ranges are cached per district-crop.
- **Schema guard.** Feature order is checked against the stored `feature_list` at startup; a mismatch fails fast (extends the existing 40-feature `/health` check).
- **Free-host constraints.** Memory and cold-start time on the chosen host decide whether SHAP explainers are built at startup or lazily. Measure before choosing.

---

## 6. Frontend and API interplay

- The frontend only calls `/api/v1`; TypeScript types are generated from the OpenAPI schema so contract changes break the build, not production.
- Map, ranking and district pages are fetched server-side or with ISR-style caching keyed to `meta.run_id`; scenario calls are client-side.
- Cold start: the API host may sleep when idle, so the UI needs loading and retry states on first request.

Detail in `ui-ux.md`.

---

## 7. Testing strategy

| Layer | What | Where |
|---|---|---|
| Unit | feature functions (GDD, dry streak, CV, water balance, RAI) on small hand-built daily series | `tests/features/` |
| Identity | recompute from daily weather reproduces stored features for a replayed year | `tests/scenario/` |
| Pipeline | sanity-gate checks fire on injected bad data (NaN, wrong count, out-of-range) | `tests/pipeline/` |
| API | schema, pagination, error codes, caching headers, bounds on scenario inputs | `tests/api/` |
| Contract | generated OpenAPI matches frontend types | CI |
| Regression | wheat predictions unchanged by the refactor | one golden file |
| Scenario research checks | direction/sign checks, replay accuracy | `tests/scenario/` + a report for the paper |

The existing `tests/test_pipeline.py` (363 lines) is split into these folders rather than rewritten.

---

## 8. Roadmap

Effort: S (hours) · M (a few days) · L (a week or more). Order matters; each phase has an exit criterion.

### Phase 0 — Foundation (cleanup)
- Secrets out of the project; rotate the Tavily key if it was ever committed (S)
- Archive pass using the grep check; new folder layout (M)
- Extract `features/weather.py` and make train + live use it (M)
- Whitelist serving models and `data/reference/` (S)
**Exit:** `pytest` green and a full refresh reproduces the current wheat predictions.

### Phase 1 — MVP (first public deploy)
- Postgres schema, migrations, seed, run writer + promotion (M)
- API v1 read endpoints with envelope, errors, caching (M)
- Next.js app: Overview map, District detail with charts, District Watch, Kharif page (L)
- CI on PRs; deploy web to Vercel and API to the container host (M)
**Exit:** public URL; every page shows `run_id`/data age; CI blocks failing PRs.

### Phase 2 — Research depth
- **Live Kharif/rice prediction step** (§4) (L)
- Headless Earth Engine auth and scheduled refresh in Actions (M)
- Scenario Explorer v2 (`api.md` §5): predict → sweep/grid → sensitivity → map (L)
- Compare page, Pipeline Health page (M)
**Exit:** scheduled refresh promotes wheat + rice; scenario identity test passes; Pipeline Health shows run history and drift.

### Phase 3 — Polish and publication support
- Methodology page (interactive figures) (M)
- CSV/JSON export (S)
- Hindi toggle (M, optional)
- Documentation and paper cross-references (S)

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Earth Engine headless auth fails or needs account setup | NDVI refresh stays manual; blocks scheduled rice step | Test early in Phase 2; fall back to manual NDVI step with a clear freshness label |
| Free API host sleeps / limited memory | Slow first load; SHAP may not fit | Loading states; lazy SHAP; measure before deciding |
| Free Postgres limits | Writes fail | Retention policy; verify limits at setup |
| Refactor changes behaviour silently | Wrong predictions | Golden-file regression + identity test + gate |
| Scenario outputs over-interpreted as forecasts | Misleading claims | Fixed framing text, warnings, `held_fixed` list in every response |
| Daily weather table larger than estimated | Storage pressure | Measure at seed time; fall back to fewer years or a repo-hosted compressed file |
| District boundary mismatch for polygons | Map errors | Bubble map first; polygons later |

---

## 10. Open items

1. Confirm which scripts are truly dead via the grep check before archiving.
2. Decide whether `21_crop_specific_models` is retired or kept as a research comparison (the live and scenario paths use `xgb_tuned`).
3. Confirm the Kharif window definition and which Kharif water baseline file the live step should use.
4. Measure memory and cold start on the chosen API host before fixing the SHAP loading strategy.