"""Initial schema (docs/database.md section 4) and role grants.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES_CREATE_ORDER = [
    "districts",
    "model_versions",
    "feature_history",
    "env_seasonal",
    "test_predictions",
    "runs",
    "predictions",
    "watch_events",
    "kharif_risk",
    "sanity_checks",
    "drift_checks",
    "incidents",
]

STATEMENTS = [
    # ---- 4.1 reference tables ----
    """
    CREATE TABLE districts (
      district_id  SMALLINT PRIMARY KEY,
      slug         TEXT UNIQUE NOT NULL,
      state        TEXT NOT NULL,
      name         TEXT NOT NULL,
      lat          DOUBLE PRECISION,
      lon          DOUBLE PRECISION,
      UNIQUE (state, name)
    )
    """,
    # ---- 4.2 model versions (referenced by test_predictions and runs) ----
    """
    CREATE TABLE model_versions (
      model_version   TEXT PRIMARY KEY,
      trained_at      TIMESTAMPTZ,
      feature_list    JSONB NOT NULL,
      metrics         JSONB,
      artifact_sha256 TEXT NOT NULL,
      is_serving      BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    """
    CREATE TABLE feature_history (
      district_id  SMALLINT REFERENCES districts,
      crop         TEXT NOT NULL CHECK (crop IN ('Wheat', 'Rice')),
      season       TEXT NOT NULL CHECK (season IN ('Kharif', 'Rabi')),
      year         SMALLINT NOT NULL,
      yield_kg_ha  REAL,
      area_ha      REAL,
      features     JSONB NOT NULL,
      PRIMARY KEY (district_id, crop, season, year)
    )
    """,
    """
    CREATE TABLE env_seasonal (
      district_id  SMALLINT REFERENCES districts,
      year         SMALLINT NOT NULL,
      season       TEXT NOT NULL,
      metric       TEXT NOT NULL,
      value        REAL,
      PRIMARY KEY (district_id, year, season, metric)
    )
    """,
    """
    CREATE TABLE test_predictions (
      model_version TEXT REFERENCES model_versions,
      district_id   SMALLINT REFERENCES districts,
      crop          TEXT,
      year          SMALLINT,
      actual        REAL,
      predicted     REAL,
      lower         REAL,
      upper         REAL,
      PRIMARY KEY (model_version, district_id, crop, year)
    )
    """,
    # ---- 4.2 runs ----
    """
    CREATE TABLE runs (
      run_id             UUID PRIMARY KEY,
      started_at         TIMESTAMPTZ NOT NULL,
      finished_at        TIMESTAMPTZ,
      status             TEXT NOT NULL CHECK (status IN ('running', 'ok', 'flagged', 'failed')),
      gate_passed        BOOLEAN,
      gate_report        JSONB,
      weather_data_as_of DATE,
      model_version      TEXT REFERENCES model_versions,
      git_sha            TEXT,
      is_latest          BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    "CREATE UNIQUE INDEX one_latest_run ON runs (is_latest) WHERE is_latest",
    # ---- 4.3 run output tables ----
    """
    CREATE TABLE predictions (
      run_id              UUID REFERENCES runs ON DELETE CASCADE,
      district_id         SMALLINT REFERENCES districts,
      crop                TEXT NOT NULL,
      season              TEXT NOT NULL,
      pred_yield          REAL NOT NULL,
      pred_lower          REAL,
      pred_upper          REAL,
      baseline_yield_year SMALLINT,
      prediction_basis    TEXT,
      features            JSONB NOT NULL,
      shap                JSONB,
      PRIMARY KEY (run_id, district_id, crop, season)
    )
    """,
    """
    CREATE TABLE watch_events (
      run_id          UUID REFERENCES runs ON DELETE CASCADE,
      district_id     SMALLINT REFERENCES districts,
      crop            TEXT NOT NULL,
      alert_level     TEXT NOT NULL,
      change_note     TEXT,
      yield_change    REAL,
      rai_change      REAL,
      severity_change REAL,
      prev_run_id     UUID,
      PRIMARY KEY (run_id, district_id, crop)
    )
    """,
    """
    CREATE TABLE kharif_risk (
      run_id              UUID REFERENCES runs ON DELETE CASCADE,
      district_id         SMALLINT REFERENCES districts,
      partial_rainfall_mm REAL,
      days_elapsed        SMALLINT,
      hist_mean           REAL,
      hist_std            REAL,
      n_years             SMALLINT,
      zscore              REAL,
      risk_level          TEXT,
      explanation         TEXT,
      PRIMARY KEY (run_id, district_id)
    )
    """,
    """
    CREATE TABLE sanity_checks (
      run_id       UUID REFERENCES runs ON DELETE CASCADE,
      check_name   TEXT NOT NULL,
      status       TEXT NOT NULL CHECK (status IN ('pass', 'fail', 'overridden')),
      detail       TEXT,
      override_ref TEXT,
      PRIMARY KEY (run_id, check_name)
    )
    """,
    """
    CREATE TABLE drift_checks (
      run_id        UUID REFERENCES runs ON DELETE CASCADE,
      feature       TEXT NOT NULL,
      baseline_mean REAL,
      baseline_std  REAL,
      current_mean  REAL,
      drift_score   REAL,
      status        TEXT,
      PRIMARY KEY (run_id, feature)
    )
    """,
    # ---- 4.4 research / documentation ----
    """
    CREATE TABLE incidents (
      incident_id SMALLINT PRIMARY KEY,
      occurred_on DATE,
      title       TEXT NOT NULL,
      category    TEXT NOT NULL,
      detection   TEXT,
      root_cause  TEXT,
      resolution  TEXT
    )
    """,
    # ---- section 10 indexes ----
    "CREATE INDEX idx_predictions_district_crop ON predictions (district_id, crop)",
    "CREATE INDEX idx_watch_events_run_level ON watch_events (run_id, alert_level)",
    "CREATE INDEX idx_feature_history_lookup ON feature_history (district_id, crop, year)",
    "CREATE INDEX idx_env_seasonal_lookup ON env_seasonal (district_id, metric, year)",
]

# Section 7: reader = SELECT only; writer = SELECT + INSERT/UPDATE on data tables.
# Guarded so the migration also runs on a database where the roles do not exist
# (for example a throwaway CI database).
GRANTS = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'naip_reader') THEN
    GRANT USAGE ON SCHEMA public TO naip_reader;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO naip_reader;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO naip_reader;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'naip_writer') THEN
    GRANT USAGE ON SCHEMA public TO naip_writer;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO naip_writer;
    GRANT INSERT, UPDATE ON
      districts, model_versions, feature_history, env_seasonal, test_predictions,
      runs, predictions, watch_events, kharif_risk, sanity_checks, drift_checks, incidents
      TO naip_writer;
  END IF;
END
$$
"""


def upgrade() -> None:
    for statement in STATEMENTS:
        op.execute(statement)
    op.execute(GRANTS)


def downgrade() -> None:
    for table in reversed(TABLES_CREATE_ORDER):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
