# NAIP Security and Responsible-Use Design

**Status date:** 2026-10-04
**Scope:** a public, read-only research demo (no accounts, no personal data) with a scheduled data pipeline and a model-serving API. The goal is proportionate controls, not enterprise security.

Legend: ✅ In place · 🔧 Present, needs fixing · 📐 Designed, not built

---

## 1. Assets and threats

| Asset | Main risks |
|---|---|
| API keys and credentials (Tavily, Earth Engine, DB) | Leak via repo, logs or CI; misuse and cost |
| Database | Unauthorised writes; data loss; free-tier exhaustion |
| API availability | Abuse of expensive endpoints (scenario grid/map), scraping |
| Model files | Tampering; unsafe deserialisation |
| Pipeline integrity | Bad data promoted and shown as truth |
| Users' browsers | XSS via untrusted text (e.g. advisories) |
| Reputation / claims | Outputs misread as agronomic or financial advice |
| Source data and IP | Licence violations; premature public disclosure of unpublished work |

Not in scope: user accounts, payments, personal data. The system collects **no personal data**; keep it that way (no tracking cookies; if analytics are added, use a privacy-respecting, cookieless option and say so).

---

## 2. Findings in the current repo 🔧

| # | Finding | Severity | Fix |
|---|---|---|---|
| F1 | `data/secrets/tavily_key.txt` (a 58-byte file) is in the project folder. It is gitignored (`data/secrets/`), but I could not check git history from the zip. | High until verified | Run `git log --all -- data/secrets`; if it was ever committed, **revoke and rotate the key**. Delete the folder; use an env var / GitHub Secret |
| F2 | `data/secrets/..gitignore` has a double-dot name, so it does nothing as a gitignore (the root `.gitignore` is what protects the folder) | Low | Delete with the folder |
| F3 | Models are **pickles** (`pickle.load` in the dashboard, `src/27`, `src/20`). Unpickling an untrusted or tampered file can execute arbitrary code | Medium | See §5 |
| F4 | API CORS is `allow_origins=["*"]`; `sort_by` accepts any column name and returns a 400 listing all column names | Low–Medium | Allow-list origins and sort fields (`api.md` §2) |
| F5 | `src/35` calls interactive `ee.Authenticate()` and hardcodes a GCP project id (`certificate-automation-…`) | Low (the id is not a secret) | Service account in CI; use a dedicated project; id in config |
| F6 | The Streamlit dashboard uses `unsafe_allow_html` 60 times, and advisory text built from live web search can reach it | Medium (XSS) | Render untrusted text as plain text in the new UI (§7); keep the old dashboard internal only |
| F7 | Dashboard/advisory text suggests uses such as procurement, insurance estimates and PDS planning | Medium (misuse / reputation) | Replace with the research-use statement (`ui-ux.md` §11) |
| F8 | No CI, no dependency or secret scanning | Medium | `ci-cd.md` + §4 below |
| F9 | The README lists Azure/AKS/Power BI as designed; none was built | Low | Keep the built-vs-designed labelling; Azure is out of scope for v2 |

---

## 3. Secrets management

- **Never in the repo.** Secrets live in GitHub Secrets (environment `production`) and the API host's environment variables. Names and owners are listed in `ci-cd.md` §7.
- **Separate DB roles:** `naip_writer` (refresh only) and `naip_reader` (API, SELECT only). The writer URL is never given to the API host.
- **Earth Engine:** a dedicated service account with only the access it needs; key stored as a secret; project id in config.
- **Rotation:** rotate any key that appears in a screenshot, log, chat or commit. Keep a short rotation note in this doc after each rotation.
- **Scanning:** enable GitHub secret scanning and push protection (check current availability for your repo's visibility) and add `gitleaks` as a pre-commit hook and CI step.
- **Logs:** no secrets or full connection strings in logs; mask in Actions.

---

## 4. Supply chain and repository hygiene

- **Dependabot** (or Renovate) for Python and npm; **`pip-audit`** and **`npm audit`** in CI (fail on high severity, review the rest).
- **Pin versions** (`xgboost==2.1.1` already pinned). Use lockfiles (`package-lock.json`; a pip constraints/lock file for Python).
- **Branch protection** on `main`: PRs required, CI required, no force-push.
- **GitHub Actions hardening:** minimal `permissions:` per workflow; pin third-party actions to a major version or SHA; **do not use `pull_request_target`**; never expose secrets to fork PRs; production secrets only in a protected `production` environment.
- **CODEOWNERS** for `api/`, `pipeline/`, `.github/` (even for a one-person repo, it documents sensitive paths).
- **SBOM / licence check** (optional): list dependency licences before publishing the repo as a research artifact.

---

## 5. Model artifact safety (F3)

Risk: `pickle.load` runs code embedded in the file. The models come from your own training, so the risk today is mainly **tampering** or accidentally loading a file from elsewhere, but the API will be a network-facing service, so tighten this.

Controls, in order of preference:
1. **Export XGBoost models to the native format** (`Booster.save_model("model.ubj")` or `.json`) and load with `load_model`. These do not execute code. Verify by predicting the same test rows with the pickle and the native model and checking they match within tolerance. (I have not checked what each pickle contains; if a pickle wraps a scikit-learn object or extra preprocessing, that part needs separate handling.)
2. **Hash pinning:** store each artifact's SHA-256 in `model_versions.artifact_sha256` (`database.md`); the API refuses to load a file that does not match.
3. **Provenance:** models are only loaded from the repo's `models/` directory, which changes only through reviewed PRs.
4. Never load a model from a user-supplied path or URL.

---

## 6. API security

| Area | Control |
|---|---|
| Surface | Read-only; **no write endpoints**; DB role is SELECT only |
| Input validation | Pydantic models with bounds on every field (scenario operators, steps, grid size); reject unknown fields |
| Injection | Parameterised queries / SQLAlchemy only; no string-built SQL; sort fields from an allow-list |
| Abuse | Per-IP rate limits (strict on scenario endpoints), request size limits, caps on grid size (≤ 15×15), sweep steps (≤ 41), and per-request timeout |
| Concurrency | Limit concurrent scenario computations so a burst cannot exhaust a small free host |
| CORS | Allow only the production and preview frontend origins |
| Errors | Stable error codes, no stack traces, no internal column names or paths |
| Transport | HTTPS only (host default); DB connections require TLS |
| Headers | `X-Content-Type-Options: nosniff`, `Referrer-Policy`, restrictive `Cache-Control` for scenario responses |
| Docs | `/docs` is fine to expose (public API); disable any admin or debug routes in production |
| Logging | Structured logs with request id, route, status, latency; no query bodies containing anything personal (there is none); short retention |

---

## 7. Web security

- **XSS:** React escapes by default. Do not use `dangerouslySetInnerHTML`. Advisory/RAG text is displayed as plain text; links are rendered only for URLs with `http(s)` schemes and open with `rel="noopener noreferrer"`.
- **Content-Security-Policy** via Vercel/Next headers: `default-src 'self'`; scripts from self only (no inline scripts, or use nonces); `img-src` limited to self and the chosen map tile host; `connect-src` limited to the API origin; `frame-ancestors 'none'`.
- **Other headers:** `Strict-Transport-Security`, `X-Content-Type-Options`, `Referrer-Policy: strict-origin-when-cross-origin`, a minimal `Permissions-Policy`.
- **Third parties:** self-host fonts; the only external requests are to the API and the map tile provider. No ad or tracking scripts.
- **URL state:** treat all query parameters as untrusted input; validate against the same bounds as the API before use.

---

## 8. Live web grounding (Tavily) — untrusted content

`src/rag_grounding_31.py` pulls live web results into advisory text. Treat everything retrieved as **untrusted**:
- Show it as plain text with source links; never as HTML.
- If an LLM is ever used to summarise retrieved text, instructions inside retrieved pages must not change behaviour (prompt injection): keep the model's role limited to summarising, never to calling tools or writing data.
- Cap length, strip markup, and label it "external sources, not model output".
- Keep it separate from model predictions; it must never feed into features or the database tables that the predictions depend on.
- If the key is missing or the call fails, the page degrades gracefully (no advisory), never errors.

---

## 9. Pipeline integrity

- **Promotion gate:** a run is served only if the sanity gate passes (`database.md` §8). This is the main control against bad data reaching users.
- **Overrides are time-boxed:** `pipeline_overrides.json` entries need a reason, evidence and an expiry ≤ 14 days. CI should fail if an override is expired or lacks these fields.
- **Schema guard:** the API checks the feature list against the stored list at startup and refuses to serve on mismatch.
- **Run provenance:** each run stores git SHA, model version and data-as-of date; rollback is one command (`ci-cd.md` §10).
- **Network calls from the pipeline:** only to the known providers (NASA POWER, Earth Engine, Tavily); validate response shapes and treat `-999` fill values explicitly (as `src/26` does).
- **Backups:** confirm what the free database tier offers for point-in-time recovery; either way, reference data is rebuildable from `data/reference/` and runs are re-creatable by re-running the pipeline (history older than the retention window is not).

---

## 10. Data licensing and IP

I am not a lawyer; these are things to check, not conclusions.

- **Source terms.** The pipeline uses NASA POWER, MODIS NDVI and JRC surface water via Earth Engine, CGWB groundwater, and crop-yield datasets from the files in `data/raw/` (a Mendeley rice dataset and others). Before publishing derived data in `data/reference/`, check each source's licence and attribution requirements, and whether redistribution of derived tables is allowed. Put attributions on the About page.
- **Earth Engine terms** (free/research-use quota and attribution) should be reviewed for a public, always-on app.
- **Repository licence:** the repo is MIT-licensed. That covers your code; it does not relicense third-party data.
- **Publication and patent timing.** Your goals include a conference paper and a patent filing. Publishing methods or code publicly before filing can affect novelty in many jurisdictions. If a filing is still planned, check timing with your guide or your institution's IP office before making new method details public. The repo's current visibility is something to confirm.

---

## 11. Responsible use

- The app is a **research prototype**. State this on every page footer and on About: *"Predictions are model estimates with uncertainty, not agronomic, insurance or financial advice."*
- The Scenario Explorer shows **model sensitivity**, not causal effects; the ModelNotice stays visible.
- Show intervals, baseline year and data age with every prediction (`ui-ux.md` §6).
- Remove advisory wording that implies operational use (procurement, insurance claims, PDS planning) unless it is validated for that use.
- Document known limitations on the Methodology page, including the 2019 yield-lag baseline and held-fixed features.

---

## 12. Incident response (one-person project)

| Event | Action |
|---|---|
| Secret exposed | Revoke and rotate immediately; remove from history if committed (use `git filter-repo`/BFG and force-push, and assume the old value is compromised regardless); check provider usage logs |
| Bad data promoted | Run `rollback-run.yml`; open an issue; add a sanity-gate check for the failure class; add an incident record (feeds the Pipeline Health taxonomy) |
| API abuse / cost spike | Tighten rate limits; block offending IPs at the host if supported; disable heavy scenario endpoints via a config flag |
| Vulnerable dependency | Update via Dependabot PR; re-run CI; redeploy |
| Model artifact mismatch | API refuses to start (hash check); restore from the last tagged release |

Keep a short `docs/incidents/` log for security events as well as pipeline ones.

---

## 13. Rollout checklist

1. ☐ Check git history for the secrets folder; rotate the Tavily key if needed; delete `data/secrets/`.
2. ☐ Enable secret scanning and push protection; add `gitleaks`.
3. ☐ Enable Dependabot; add `pip-audit` / `npm audit` to CI.
4. ☐ Branch protection on `main`.
5. ☐ Convert models to native XGBoost format and verify predictions; add hash pinning.
6. ☐ Restrict CORS; allow-list sort fields; add Pydantic bounds and rate limits.
7. ☐ Security headers and CSP on the web app.
8. ☐ Advisory text rendered as plain text.
9. ☐ Replace operational-use wording with the research-use statement.
10. ☐ Review data licences; add attributions; settle publication/patent timing.

---

## 14. Open items

1. Whether any key was ever committed (needs `git log` on your machine; the zip has no `.git`).
2. Native-format conversion feasibility for each pickled model.
3. Free-tier backup/PITR offerings of the chosen database host.
4. Current availability of GitHub secret scanning/push protection for your repo visibility.
5. Licence review of each source dataset.
6. Repository visibility and the patent/paper timeline.