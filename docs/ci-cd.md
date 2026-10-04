# NAIP CI/CD Design

**Status date:** 2026-10-04
**Tooling:** GitHub Actions (CI and scheduled refresh), Vercel (web), a free container host (API), Neon/Supabase Postgres.

Legend: ✅ Built · 🔧 Built, needs refactor · 📐 Designed, not built

---

## 1. Current state

- **No CI exists.** The uploaded project has no `.github/` folder. 🔧
- Tests exist (`tests/test_pipeline.py`, pytest) and run locally only. ✅
- The refresh runs on one Windows machine via Task Scheduler and `run_refresh.bat` (hardcoded paths, conda). 🔧
- `requirements.txt` mixes pipeline, API, dashboard and test dependencies in one file.
- No deployment of any kind has been done for this project. Dockerfiles/Kubernetes from the README are out of scope for v2.

Everything below is 📐 designed, not built.

---

## 2. Goals

1. **No broken code on `main`:** every change passes lint, tests and a build before merge.
2. **Refresh without a personal machine:** scheduled, observable, and unable to publish a bad run.
3. **Push-to-deploy at zero cost:** `main` deploys the web and API automatically.
4. **Reversible:** any deploy or promoted run can be rolled back quickly.

---

## 3. Pipeline overview

| Workflow | Trigger | Purpose |
|---|---|---|
| `ci.yml` | PR, push to `main` | Lint, test, build, contract check |
| `refresh.yml` | Daily cron, manual | Run the data/prediction pipeline, persist, promote |
| `deploy-api.yml` | Push to `main` touching `api/`, `models/`, `pipeline/features/` | Migrate DB, build and deploy the API |
| `rollback-run.yml` | Manual | Re-promote a previous run |
| Vercel Git integration | PR, push to `main` | Preview per PR; production on `main` |

Vercel deploys the web app through its own GitHub integration (root directory `web/`), so no workflow is needed for it.

---

## 4. Requirements split

Replace the single `requirements.txt` with:

| File | Contents |
|---|---|
| `requirements-pipeline.txt` | pandas, numpy, scipy, scikit-learn, xgboost==2.1.1, shap, requests, earthengine-api, geopy, tavily-python, SQL driver |
| `requirements-api.txt` | fastapi, uvicorn, pydantic, xgboost==2.1.1, shap, pandas, numpy, SQL driver, rate limiter |
| `requirements-dev.txt` | pytest, ruff, pre-commit, httpx |

Streamlit, folium and matplotlib are only needed for the legacy dashboard and notebooks. They stay out of the API image to keep it small.

**Pin xgboost** (already `==2.1.1`) in both pipeline and API: the model files are pickles, and loading them with a different xgboost version can fail or behave differently. The CI "model loads and predicts" test (below) guards this.

**Python version:** use 3.10 everywhere (the repo's existing bytecode is CPython 3.10) unless you deliberately upgrade and retest the pickles.

---

## 5. `ci.yml` (PR and `main`)

Jobs run in parallel; all must pass to merge.

| Job | Steps |
|---|---|
| `lint-python` | `ruff check` and `ruff format --check` |
| `test-python` | Postgres 16 service container → install pipeline + API + dev deps → run migrations → `pytest` (unit, identity, pipeline-gate, API tests, golden regression) |
| `model-smoke` | Load the three serving models, check the feature list matches `feature_list_v2.txt` (40 features), predict one stored row |
| `api-contract` | Export OpenAPI from the app; generate TypeScript types; fail if the committed types differ |
| `web` | `npm ci`, typecheck, lint, `next build` (uses the generated types) |

Sketch:

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  test-python:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env: { POSTGRES_PASSWORD: test, POSTGRES_DB: naip_test }
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready" --health-interval 5s
          --health-timeout 5s --health-retries 10
    env:
      DATABASE_URL: postgresql://postgres:test@localhost:5432/naip_test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10", cache: pip }
      - run: pip install -r requirements-pipeline.txt -r requirements-api.txt -r requirements-dev.txt
      - run: alembic upgrade head
      - run: pytest -q
```

Notes:
- CI never calls NASA POWER, Earth Engine or Tavily; tests use fixtures or recorded samples, so CI stays fast and does not need secrets.
- Fork PRs get no secrets by default; that is fine because CI needs none.
- Branch protection on `main`: require the CI checks, require PRs, block force-push.

---

## 6. `refresh.yml` (scheduled pipeline)

### 6.1 Behaviour
1. Check out, set up Python with pip caching.
2. Restore a cache of `data/raw/` (best-effort; the pipeline must also work from an empty cache).
3. Authenticate to Earth Engine with a **service account** (§7).
4. Run `python -m pipeline.run_refresh --persist`.
5. Pipeline writes the run to Postgres under a new `run_id`, runs the sanity gate, and promotes only on pass (`database.md` §8).
6. Trim retention (`database.md` §6).
7. Upload the run log as an artifact (14-day retention).
8. On failure or a flagged gate, open (or comment on) a GitHub Issue.

### 6.2 Exit codes (orchestrator contract)
| Code | Meaning | Workflow result |
|---|---|---|
| 0 | All stages succeeded, gate passed, run promoted | Success |
| 1 | A stage failed; run marked `failed`, nothing promoted | Failure + issue |
| 2 | Stages ran but the gate flagged problems; run stored as `flagged`, not promoted | Failure + issue |

This extends today's contract (0 = success, 1 = failure or flagged) by separating "pipeline broke" from "data looks wrong".

### 6.3 Sketch

```yaml
name: refresh
on:
  schedule:
    - cron: "30 4 * * *"      # 10:00 IST; cron is UTC
  workflow_dispatch:
concurrency:
  group: refresh
  cancel-in-progress: false
permissions:
  contents: read
  issues: write
jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 90
    environment: production
    env:
      DATABASE_URL: ${{ secrets.DATABASE_URL_WRITER }}
      EE_PROJECT_ID: ${{ vars.EE_PROJECT_ID }}
      TAVILY_API_KEY: ${{ secrets.TAVILY_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10", cache: pip }
      - uses: actions/cache@v4
        with:
          path: data/raw
          key: raw-weather-${{ github.run_id }}
          restore-keys: raw-weather-
      - run: pip install -r requirements-pipeline.txt
      - name: Earth Engine credentials
        run: echo '${{ secrets.EE_SERVICE_ACCOUNT_JSON }}' > "$RUNNER_TEMP/ee.json"
      - run: python -m pipeline.run_refresh --persist
        env:
          EE_SERVICE_ACCOUNT_FILE: ${{ runner.temp }}/ee.json
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: refresh-log
          path: logs/
          retention-days: 14
      - name: Open issue on failure
        if: failure()
        env: { GH_TOKEN: "${{ github.token }}" }
        run: gh issue create --title "Refresh failed ($GITHUB_RUN_ID)" --body "See run ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}" --label pipeline
```

### 6.4 Things to verify before relying on it
- **Duration on a clean runner.** The weather step fetches current weather per district (the raw file is ~25 MB today). I have not measured its runtime. If it is long, fetch incrementally and lean on the cache.
- **Scheduled-workflow behaviour.** GitHub can delay cron runs, and scheduled workflows on repositories with no recent activity can be paused. Check the current rules and add a keep-alive or manual trigger if needed.
- **Free minutes.** Confirm current free-tier limits for your repo's visibility (public vs private).
- **NASA POWER lag.** Data runs a few days behind; the run records `weather_data_as_of`, and the UI shows it.

---

## 7. Secrets and configuration

| Name | Type | Used by | Notes |
|---|---|---|---|
| `DATABASE_URL_WRITER` | Secret (GitHub environment `production`) | refresh, migrations | Write role (`naip_writer`) |
| `DATABASE_URL_READER` | Secret (API host) | API | SELECT-only role |
| `EE_SERVICE_ACCOUNT_JSON` | Secret | refresh | Earth Engine service account key |
| `EE_PROJECT_ID` | Variable | refresh | Moved out of `src/35` (hardcoded today) |
| `TAVILY_API_KEY` | Secret | refresh, API | Replaces `data/secrets/tavily_key.txt` |
| `ALLOWED_ORIGINS` | Env var (API host) | API | Frontend origins for CORS |
| `NEXT_PUBLIC_API_BASE_URL` | Env var (Vercel) | web | API base URL |

Rules:
- No secret in the repo or logs. Rotate the Tavily key if it was ever committed (`git log --all -- data/secrets`).
- Jobs declare minimal `permissions:`.
- Pin third-party actions to a major version (or a commit SHA for anything beyond the official `actions/*`).

### Headless Earth Engine auth (must be proven early) 📐
Today `src/35` calls `ee.Authenticate()`, which is interactive and cannot run in CI. Target: initialise with service-account credentials from `EE_SERVICE_ACCOUNT_FILE` and `EE_PROJECT_ID`. Requirements to check in your Google Cloud / Earth Engine setup: the service account exists, is registered for Earth Engine use, and the project has access. **I have not verified this for your project.** Test it with a one-off `workflow_dispatch` job before building the rice step on top of it. If it fails, the fallback is a manual NDVI step with a clear freshness label in the UI.

---

## 8. `deploy-api.yml`

Triggered on `main` when `api/`, `models/`, `pipeline/features/` or migrations change.

1. Re-run the CI-equivalent tests (or require CI green via branch protection).
2. `alembic upgrade head` against production using the writer URL (a separate `migrate` job so a failed migration stops the deploy).
3. Build the API Docker image (`api/Dockerfile`, installs `requirements-api.txt`, copies `models/`).
4. Deploy.

Deploy mechanism depends on the host (decided at setup):
- **Render:** connect the repo for auto-deploy, or call a deploy hook from the workflow after tests pass.
- **Fly.io:** `flyctl deploy` with a token secret (check whether a payment method is required).
- **Hugging Face Spaces (Docker):** push to the Space repository.

After deploy, a smoke job calls `/api/v1/health` and fails the workflow if the schema check or DB check fails.

---

## 9. Web deploy (Vercel)

- Connect the GitHub repo; set root directory to `web/`.
- Every PR gets a preview URL; `main` deploys to production.
- Previews point at the production API (read-only), so no staging API is needed; this is why the API has no write endpoints.
- `NEXT_PUBLIC_API_BASE_URL` set per environment.
- The CI `web` job must pass before merge; Vercel's build is a second check.

---

## 10. Rollback

| Layer | Rollback |
|---|---|
| Web | Vercel "promote previous deployment" |
| API | Redeploy the previous commit or image (host dashboard or re-run the workflow on an earlier SHA) |
| Data | `rollback-run.yml` (manual) calls `python -m pipeline.db.promote --run-id <id>`, which flips `is_latest` back to an earlier `ok` run in one transaction |
| Schema | Migrations are forward-only by default; keep them additive so the previous API version still works. Destructive changes only in a follow-up release |

---

## 11. Local development

- `docker compose up` starts Postgres (and optionally the API) for local work.
- `make` (or a small `tasks.py`) targets: `seed`, `refresh`, `test`, `lint`, `api`, `web`.
- `pre-commit` with `ruff` so lint failures are caught before pushing.
- The local refresh replaces `run_refresh.bat` for development; Task Scheduler is no longer used.

---

## 12. Observability

- Refresh artifacts: per-run log (14 days).
- Issues auto-opened on failed/flagged runs (label `pipeline`).
- `/api/v1/health` reports last run, gate result and data age; the Pipeline Health page shows run history.
- Optional later: a free uptime ping on `/health` (also keeps a sleeping free API host warm; check the host's policy first).

---

## 13. Rollout order

1. Split requirements; add `ruff` and `pre-commit` (no behaviour change).
2. Add `ci.yml` with lint and the existing pytest suite; turn on branch protection.
3. Add the model smoke test and Postgres service once the DB layer exists.
4. Prove headless Earth Engine auth with a manual workflow.
5. Add `refresh.yml` as `workflow_dispatch` only; run it manually until stable, then enable the cron.
6. Connect Vercel; add `deploy-api.yml`.
7. Add `rollback-run.yml`.

---

## 14. Open items

1. Choose the API host (affects §8 mechanism); measure memory and cold start.
2. Measure refresh duration from an empty cache.
3. Confirm Earth Engine service-account setup works headless.
4. Confirm current free-tier limits (Actions minutes, Neon/Supabase storage, host limits).
5. Decide whether the legacy Streamlit dashboard is archived or kept as an internal view (affects which requirements it needs).