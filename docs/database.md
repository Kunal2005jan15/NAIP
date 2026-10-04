# NAIP Database Design

**Status date:** 2026-10-04
**Engine:** PostgreSQL on a free tier (Neon or Supabase). Verify current storage limits and idle/scale-to-zero behaviour before committing; the design below is sized to stay far under a typical free quota.

Legend: ✅ Built · 🔧 Built, needs refactor · 📐 Designed, not built

---

## 1. Current state

Today there is no database. All data are CSV/JSON/PKL files under `data/` and `outputs/`, read directly by the pipeline, the API (mtime-cached) and the Streamlit dashboard. 🔧

Consequences this design removes: different readers can see different file versions, a failed refresh can overwrite good outputs, history is only available as accumulating snapshot files, and the serving path depends on files that are not in version control.

---

## 2. Principles

1. **One serving store.** The API reads Postgres only.
2. **Runs are atomic.** Each refresh writes under one `run_id`; a run is served only after the sanity gate passes.
3. **Reproducible by construction.** Every prediction stores the exact input feature vector, model version and run, so any number can be re-derived.
4. **Small by design.** 118 districts × 2 crops is tiny; the only growth risk is per-run SHAP detail, handled by retention (section 6).
5. **Reference data in git, run data in the database.** See section 5.

---

## 3. Data classes

| Class | Examples | Lives in | Changes |
|---|---|---|---|
| Reference (slow) | district list, coordinates, yield history, seasonal weather/NDVI/water/groundwater history, test predictions | Postgres, seeded from `data/reference/` | Rarely; reseed |
| Run output (fast) | predictions, watch events, Kharif risk, drift, sanity checks | Postgres, written by the refresh | Every run |
| Model artifacts | `xgb_tuned`, `xgb_lower`, `xgb_upper` | Git (`models/`), loaded by the API | On retrain |
| Overrides | `pipeline_overrides.json` | Git (stays a file) | By hand |

`pipeline_overrides.json` deliberately stays a version-controlled file: each override carries a reason, evidence and an expiry no more than 14 days out, and being reviewed in git history is part of its design. The database stores which override (if any) applied to a check, not the override itself.

---

## 4. Schema (target) 📐

### 4.1 Reference tables

```sql
CREATE TABLE districts (
  district_id  SMALLINT PRIMARY KEY,
  slug         TEXT UNIQUE NOT NULL,         -- 'uttar-pradesh/kheri'
  state        TEXT NOT NULL,
  name         TEXT NOT NULL,
  lat          DOUBLE PRECISION,
  lon          DOUBLE PRECISION,
  UNIQUE (state, name)
);

-- Training/feature history: one row per district-crop-season-year.
-- Source: data/processed/model_ready_v2.csv. Powers history charts
-- and the scenario "weather-year replay".
CREATE TABLE feature_history (
  district_id  SMALLINT REFERENCES districts,
  crop         TEXT NOT NULL CHECK (crop IN ('Wheat','Rice')),
  season       TEXT NOT NULL CHECK (season IN ('Kharif','Rabi')),
  year         SMALLINT NOT NULL,
  yield_kg_ha  REAL,
  area_ha      REAL,
  features     JSONB NOT NULL,               -- the 40 model features
  PRIMARY KEY (district_id, crop, season, year)
);

-- Seasonal environment series for charts (rainfall, temp, NDVI, water, groundwater)
CREATE TABLE env_seasonal (
  district_id  SMALLINT REFERENCES districts,
  year         SMALLINT NOT NULL,
  season       TEXT NOT NULL,
  metric       TEXT NOT NULL,                -- 'rainfall_mm','temp_max_c','ndvi','water_pct','gw_depth_post_m',...
  value        REAL,
  PRIMARY KEY (district_id, year, season, metric)
);

-- Held-out predictions for calibration/residual charts (Methodology page)
CREATE TABLE test_predictions (
  model_version TEXT REFERENCES model_versions,
  district_id   SMALLINT REFERENCES districts,
  crop          TEXT, year SMALLINT,
  actual REAL, predicted REAL, lower REAL, upper REAL,
  PRIMARY KEY (model_version, district_id, crop, year)
);
```

### 4.2 Model and run tables

```sql
CREATE TABLE model_versions (
  model_version TEXT PRIMARY KEY,            -- e.g. 'xgb_tuned_2026-07'
  trained_at    TIMESTAMPTZ,
  feature_list  JSONB NOT NULL,              -- ordered; must match the 40-feature schema
  metrics       JSONB,                       -- test, walk-forward CV, calibration
  artifact_sha256 TEXT NOT NULL,
  is_serving    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE runs (
  run_id        UUID PRIMARY KEY,
  started_at    TIMESTAMPTZ NOT NULL,
  finished_at   TIMESTAMPTZ,
  status        TEXT NOT NULL CHECK (status IN ('running','ok','flagged','failed')),
  gate_passed   BOOLEAN,
  gate_report   JSONB,                       -- per-check results from the sanity gate
  weather_data_as_of DATE,
  model_version TEXT REFERENCES model_versions,
  git_sha       TEXT,
  is_latest     BOOLEAN NOT NULL DEFAULT FALSE
);
-- Exactly one promoted run at a time
CREATE UNIQUE INDEX one_latest_run ON runs (is_latest) WHERE is_latest;
```

### 4.3 Run output tables

```sql
CREATE TABLE predictions (
  run_id       UUID REFERENCES runs ON DELETE CASCADE,
  district_id  SMALLINT REFERENCES districts,
  crop         TEXT NOT NULL,
  season       TEXT NOT NULL,
  pred_yield   REAL NOT NULL,
  pred_lower   REAL,
  pred_upper   REAL,
  baseline_yield_year SMALLINT,              -- lag baseline (e.g. 2019): shown in UI
  prediction_basis    TEXT,                  -- e.g. 'current_2026_weather_with_2019_yield_baseline'
  features     JSONB NOT NULL,               -- exact 40-feature input (reproducibility + scenario baseline)
  shap         JSONB,                        -- per-feature SHAP; subject to retention
  PRIMARY KEY (run_id, district_id, crop, season)
);

CREATE TABLE watch_events (
  run_id UUID REFERENCES runs ON DELETE CASCADE,
  district_id SMALLINT REFERENCES districts,
  crop TEXT NOT NULL,
  alert_level TEXT NOT NULL,                 -- watch / new / elevated_stable / ...
  change_note TEXT,
  yield_change REAL, rai_change REAL, severity_change REAL,
  prev_run_id UUID,
  PRIMARY KEY (run_id, district_id, crop)
);

CREATE TABLE kharif_risk (
  run_id UUID REFERENCES runs ON DELETE CASCADE,
  district_id SMALLINT REFERENCES districts,
  partial_rainfall_mm REAL, days_elapsed SMALLINT,
  hist_mean REAL, hist_std REAL, n_years SMALLINT,
  zscore REAL, risk_level TEXT, explanation TEXT,
  PRIMARY KEY (run_id, district_id)
);

CREATE TABLE sanity_checks (
  run_id UUID REFERENCES runs ON DELETE CASCADE,
  check_name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pass','fail','overridden')),
  detail TEXT,
  override_ref TEXT,                         -- key in pipeline_overrides.json, if applied
  PRIMARY KEY (run_id, check_name)
);

CREATE TABLE drift_checks (
  run_id UUID REFERENCES runs ON DELETE CASCADE,
  feature TEXT NOT NULL,
  baseline_mean REAL, baseline_std REAL, current_mean REAL,
  drift_score REAL, status TEXT,
  PRIMARY KEY (run_id, feature)
);
```

### 4.4 Research / documentation table

```sql
-- Source for the Pipeline Health page's incident taxonomy
CREATE TABLE incidents (
  incident_id SMALLINT PRIMARY KEY,
  occurred_on DATE, title TEXT NOT NULL,
  category TEXT NOT NULL,                    -- failure taxonomy class
  detection TEXT, root_cause TEXT, resolution TEXT
);
```

Note: the 12 incidents are documented in the README and paper draft but not yet in structured form; transcribing them into a seed file is a task, not an existing asset.

---

## 5. Reference data in git (zero-cost handoff)

CI must be able to run the refresh without any paid storage, so the files the live pipeline needs are committed under a whitelisted `data/reference/` (~6 MB), alongside `models/` (~10 MB).

| File | Size | Used by |
|---|---|---|
| `model_ready_v2.csv` | ~3.4 MB | lags, baselines, `feature_history` seed |
| `nasa_seasonal_ALL_DISTRICTS.csv` | ~1 MB | historical rainfall anomaly baselines |
| `ndvi_seasonal.csv`, `surface_water_seasonal.csv`, `groundwater_district.csv` | ~0.4 MB | env charts, baselines |
| `district_coordinates_final.csv`, `districts_for_nasa_fetch.csv`, `district_list_clean.csv` | <20 KB | districts, weather fetch |
| `current_surface_water_baseline.csv`, `drift_baseline.json`, `feature_list_v2.txt` | <10 KB | live features, drift, schema check |
| `test_predictions_FINAL.csv` + metrics CSVs | small | Methodology page |

Everything else under `data/` stays out of git:
- **Raw weather** is refetched from NASA POWER by the pipeline.
- **`data/snapshots/`** is replaced by the `runs`/`predictions` history in Postgres.
- **Superseded versions** (`master_dataset` v1–v5, `train`/`valid`/`test` v1, `model_ready.csv`) move to a local `archive/` after you confirm none are read by a script you still run.

> To confirm before archiving: run `grep -rn "<filename>" src api dashboard tests` for each candidate. My list is based on names and sizes, not a full dependency check.

---

## 6. Retention and size 

Per run: 236 prediction rows (118 districts × 2 crops), each with a ~1 KB feature vector and a ~1 KB SHAP JSON, so roughly 0.5 MB per run before indexes. A daily cadence is about 180 MB/year, which is too much for a small free quota if kept forever.

Policy:
- Keep full `features` + `shap` for the **last 30 runs**.
- Beyond that, null out `shap` (keep `pred_*`, `features`), and beyond **180 days** keep only `pred_*` and drop `features`.
- `watch_events`, `kharif_risk`, `sanity_checks`, `runs` are small; keep all.
- Run the trim as the last step of each refresh.

Reference tables are static and well under 10 MB combined.

---

## 7. Access and roles

| Role | Used by | Rights |
|---|---|---|
| `naip_writer` | GitHub Actions refresh | INSERT/UPDATE on run tables, `runs` promotion, seed on reference tables |
| `naip_reader` | API | SELECT only |

Connection strings live in GitHub Secrets and the API host's environment, never in the repo.

---

## 8. Promotion procedure

1. Insert `runs` row with `status='running'`.
2. Write run outputs under that `run_id`.
3. Run the sanity gate; record `sanity_checks`; set `gate_passed`.
4. In one transaction: if passed, set previous `is_latest=false`, this run `is_latest=true`, `status='ok'`; if not, `status='flagged'` and leave `is_latest` untouched.
5. API queries always filter `WHERE runs.is_latest`.

Pipeline failures set `status='failed'`. The UI shows the last promoted run with its age.

---

## 9. Migrations and local dev

- Schema managed with Alembic; migrations in `pipeline/db/migrations/`.
- Local: `docker compose up` starts Postgres; `python -m pipeline.db.seed_reference` loads `data/reference/`.
- Tests run against a throwaway Postgres in CI.

---

## 10. Indexes (initial)

- `predictions (district_id, crop)` for district detail across runs.
- `watch_events (run_id, alert_level)` for the Watch page.
- `feature_history (district_id, crop, year)` for history/replay.
- `env_seasonal (district_id, metric, year)` for charts.

Add others only after a slow query is observed.

---

## 11. Open items

1. Seed file for the 12 documented incidents and failure taxonomy.
2. District boundary geometry (PostGIS or static GeoJSON in the frontend) deferred; coordinates only for now.
3. Confirm drift metric (z-score vs PSI) from `src/23_drift_detection.py` / `24_drift_v2.py` so `drift_score` is documented correctly.
4. Verify the free-tier storage limit and cold-start behaviour of the chosen host.