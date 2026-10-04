# NAIP UI/UX Design

**Status date:** 2026-10-04
**Stack:** Next.js (App Router) + TypeScript on Vercel, calling `/api/v1` (see `api.md`).

Legend: ✅ Built · 🔧 Built, needs refactor · 📐 Designed, not built

---

## 1. Current state 🔧

The Streamlit dashboard (`dashboard/app.py`, 1,754 lines) has 6 tabs: District Watch, Kharif Early Warning, Current Predictions, Historical, Scenario Explorer (demoted to "Advanced/Testing"), About. It has a developed visual identity ("Earth & Grain": paper/ink/wheat/soil/green palette; Fraunces, Inter and JetBrains Mono), advisory cards, legends and empty states. Weaknesses to fix in v2:

- Almost no interactive, district-linked charts; most content is cards and tables.
- Styling is injected as HTML/CSS strings (60 uses of `unsafe_allow_html`) and has had cascade bugs (e.g. invisible active tab label).
- Provenance (data age, baseline year, model) appears in some places, not consistently.
- Some text overclaims (see §11).

v2 keeps the visual identity and rebuilds structure, charts and honesty cues.

---

## 2. Design principles

1. **Show uncertainty and provenance everywhere.** Every number with its interval, data age and baseline.
2. **Charts answer district questions.** Each chart is tied to the selected district/crop and cross-linked to the others.
3. **Research tool, not an advisory service.** Language describes model behaviour; it does not tell people what to do with money or crops.
4. **Readable before pretty.** Accessibility and legibility constraints win over decoration.
5. **Fast on mobile and slow networks.** Many users will be on phones.

---

## 3. Audiences and key tasks

| Audience | Key tasks |
|---|---|
| Reviewers / examiners | Understand the system fast; verify claims; see validation and limits |
| Researchers | Inspect a district's prediction and drivers; compare districts; run scenarios; export data |
| General viewers | See which districts look at risk this season and why |

Primary flows: **find a district → read its prediction and drivers → test a scenario → compare**, and **check pipeline health / methodology**.

---

## 4. Information architecture

```
/                       Overview (map + ranking)
/districts              Search / list
/districts/[state]/[district]   District detail (crop toggle)
/watch                  District Watch
/kharif                 Kharif Early Warning
/scenario               Scenario Explorer (district + crop via URL)
/compare                Compare (2–4 districts)
/pipeline               Pipeline Health
/methodology            Model validation and limitations
/about                  Project, data sources, citation, licence
```

Global controls (in the header, persisted in the URL): **crop** (Wheat | Rice), and later **language** (P3). Search box jumps to any district.

**State in the URL** (`?crop=Wheat&district=…&rain=-20&temp=1.5&base=replay:2009`) so any view, including a scenario, is shareable and reproducible. No accounts.

---

## 5. Design system

### 5.1 Tokens (reuse existing)
Keep the existing CSS variables as the single source of truth, exposed to Tailwind: `--paper #F8F4EC`, `--paper-deep #EFE8D8`, `--ink #1C1610`, `--ink-soft #4A3F2F`, `--ink-faint #8C7B64`, `--wheat #C9A227`, `--wheat-dark #9C7E1A`, `--wheat-pale #F6EDCB`, `--soil #7A4E2A`, `--green #2E5E4E`, `--green-bg #E4EDE9`, `--alert #9B2C1E`, `--alert-bg #F7E8E5`, `--line #DDD3BD`; radii 6/12/20/28 px; the existing shadow scale. Fonts via `next/font` (self-hosted): Fraunces (headings), Inter (UI), JetBrains Mono (numbers/code).

### 5.2 Contrast audit of the existing palette (WCAG, on `--paper`)
| Pair | Ratio | Use |
|---|---|---|
| ink / paper | 16.3 | Body text ✅ |
| ink-soft / paper | 9.4 | Secondary text ✅ |
| green / paper | 6.8 | Text, chart lines ✅ |
| alert / paper | 6.9 | Text, chart lines ✅ |
| soil / paper | 6.5 | Text, chart lines ✅ |
| ink-faint / paper | 3.7 | ❌ small text (needs 4.5); OK for large text and graphics |
| wheat-dark / paper | 3.5 | ❌ body text; OK for large text and graphics (3:1) |
| wheat / paper | 2.2 | ❌ text and thin lines. Fill only, with an outline, or on dark backgrounds |
| line / paper | 1.4 | Decorative dividers only |

Rules: never put body text in `wheat`, `wheat-dark` or `ink-faint`; use `ink-soft` for secondary text. Chart lines and markers must reach 3:1 against their background. Automated contrast checks run in CI (§15).

### 5.3 Chart colour semantics
- **Categorical series:** green, soil, alert, ink-soft (all ≥ 6:1), then wheat-dark with a distinct marker/dash. Never rely on hue alone: add dash patterns and direct labels.
- **Risk levels:** encode with **icon + label + lightness**, not red/green alone (the legacy UI's ✓ ◐ ● markers are good; keep them).
- **Anomaly (diverging):** dry ↔ wet. Brown-vs-green can be confusable under common colour-vision deficiencies, so add one **water-blue token** for the wet end (proposed `--water`, to be chosen and contrast-checked ≥ 3:1) and show a centred zero line.
- **Intervals:** translucent band of the series colour plus a visible boundary line.

### 5.4 Layout
Mobile-first, 12-column grid, content max-width ~1200 px. Cards use `--paper-deep` with `--r-md`. Wide charts and tables scroll inside their own container, never the page body. Animations are subtle and honour `prefers-reduced-motion`.

---

## 6. Honesty components (used on every page)

| Component | Content | Source |
|---|---|---|
| **RunBadge** (header) | "Data as of 29 Sep 2026 · run b1c2… · model xgb_tuned" + status dot (ok / flagged / stale) | `meta` |
| **BasisNote** | "Prediction basis: current 2026 weather with 2019 yield baseline" shown next to any prediction | `meta.prediction_basis` |
| **IntervalLabel** | "Prediction interval (quantile 2–98%)" plus the validated empirical coverage when known; never "95%" unless that is the verified figure | `meta` / Methodology |
| **ModelNotice** | "Model sensitivity, not a causal forecast." on Scenario pages | `meta.interpretation` |
| **StaleBanner** | Shown if data age exceeds a threshold or the last run was flagged/failed | `/health` |
| **InSeasonNotice** | "Season in progress; no rice prediction yet" | `meta.notice` |

---

## 7. Chart standards

- **Library:** Apache ECharts (React wrapper) for line/band, bar, heatmap, scatter, tornado and waterfall (custom series); one library keeps bundle size and styling consistent. Load charts lazily (dynamic import) per page.
- **Every chart has:** a title that states the takeaway or subject, axis labels with units (kg/ha, mm, °C), a legend or direct labels, a tooltip with exact values, and a **"View as table"** toggle (accessibility and verification).
- **Interactions:** hover tooltip, legend toggle, zoom/brush on time series, click-through (map bubble → district page; ranking row ↔ map highlight), reset.
- **District linkage:** selecting a district anywhere highlights it in every visible chart on the page.
- **Uncertainty:** series with intervals always draw the band; extrapolation regions (§9.3) are hatched.
- **Numbers:** yields rounded to the nearest 10 kg/ha with an "≈" in headline values; full precision in tooltips and tables. Avoid false precision.
- **Empty/failed data:** a chart placeholder that says why (e.g. "No groundwater data for this district"), never a blank box.

**Map:** start as a **bubble map** (district lat/lon; size or colour = metric). Basemap tiles must come from a provider whose terms allow this use (verify; the main OpenStreetMap tile servers have usage limits), or use no basemap with state outlines. District polygons are a later upgrade once boundaries are sourced and name-matched. Always pair the map with a sortable ranking list (keyboard- and screen-reader-friendly alternative).

---

## 8. Page specifications

Phase labels refer to `systemdesign.md`.

### 8.1 Overview `/` (MVP)
- **Map** (metric toggle: predicted yield, change vs baseline, drought/flood flag, RAI) + **ranking bar chart**, linked by hover/click.
- Summary strip: districts predicted, flagged count, run badge.
- Data: `GET /predictions?crop=`.
- States: rice shows the InSeasonNotice until a rice run exists.

### 8.2 District detail `/districts/[state]/[district]` (MVP)
Header: name, crop toggle, prediction headline (≈ value, interval, BasisNote), flags.

| Chart | Content | Endpoint |
|---|---|---|
| Yield history | Actual yield line (1998–2019) with model predictions and interval band; current prediction marked | `/history` |
| Driver waterfall | SHAP contributions for the current prediction, each bar labelled with the feature's input value | `/shap` |
| Rainfall anomaly by year | RAI bars (diverging), current season highlighted | `/environment` |
| Temperature / heat | Seasonal max temp and heat-stress days by year | `/environment` |
| Vegetation | Seasonal NDVI by year (seasonal means, **not** a within-season curve) | `/environment` |
| Water | Surface-water share and groundwater depth (pre/post monsoon) by year; groundwater series ends at the dataset's last year, labelled | `/environment` |

Interactions: brush a year range on the history chart to filter the environment charts below; "Open in Scenario Explorer" button preserves district and crop.

### 8.3 District Watch `/watch` (MVP)
- Sortable, filterable table by alert level (with icon + label), crop, state.
- Row sparkline: prediction across the last runs.
- Drill-down (P2): which features moved the prediction since the last run (bar of feature-level deltas).
- Data: `/watch`, `/districts/…/watch`.

### 8.4 Kharif Early Warning `/kharif` (MVP)
- Map and list coloured by risk level (icon + label).
- Per-district **bullet chart**: season-to-date rainfall vs the same-window historical mean ± std (window totals).
- Explanation text from the API; data-lag notice (NASA POWER) shown prominently.
- Upgrade when a daily cumulative series is stored: cumulative rainfall vs historical envelope.

### 8.5 Compare `/compare` (P2)
2–4 districts. Overlaid yield history, a **dumbbell/small-multiples** view of key drivers (radar charts are hard to read accurately, so avoid them), and a summary table.

### 8.6 Pipeline Health `/pipeline` (P2)
- **Run timeline:** one tile per run coloured by status (ok / flagged / failed) with icon + label.
- **Gate checks:** table for the selected run (pass / fail / overridden).
- **Drift strip:** per-feature drift status for the selected run.
- **Incident taxonomy:** bar chart of documented incidents by failure class, with a table of the incidents.
- This is the page that shows the "self-monitoring pipeline" contribution; link it from the About page.

### 8.7 Methodology `/methodology` (P3)
Interactive versions of the existing figures: actual vs predicted (crop filter), residuals, interval calibration/coverage, SHAP importance (wheat vs rice), walk-forward CV by year, baseline comparison. Plus a "Limitations" section (2019 lag baseline, associational model, held-fixed features, data coverage).

### 8.8 About `/about` (MVP)
What it is, data sources and licences, how to cite, repo link, API docs link, the research-use statement.

---

## 9. Scenario Explorer (P2)

### 9.1 Layout
Desktop: left **controls panel** (sticky) + right **results area**. Mobile: controls in a bottom sheet / collapsible section above results.

**Controls panel**
1. District (search) and crop toggle.
2. **Baseline:** *Current season* (when available) or *Replay a past year* (year select). Rice defaults to replay until a live rice baseline exists.
3. **Rain:** rainfall change slider (−60% … +100%); optional dry-spell (length, when).
4. **Temperature:** shift (−3 … +5 °C); optional heatwave (days, extra °C).
5. **Inputs:** irrigation change, nitrogen change.
6. **Presets (data-driven):** replay this district's driest, wettest and hottest recorded year; "Reset to baseline".
Each control has a slider **and** a numeric field (keyboard and precision), shows its bounds, and shows the baseline value.

### 9.2 Results area
1. **Result card:** baseline vs scenario as paired intervals (dumbbell with interval bars), delta in kg/ha and %, BasisNote, ModelNotice.
2. **Response curves:** yield vs rainfall change and vs temperature shift, with interval bands; the current setting is marked on the curve.
3. **Heatmap:** rainfall × temperature grid with the current point marked.
4. **Tornado:** one-at-a-time sensitivity, ranked, for this district.
5. **SHAP delta waterfall:** which features moved between baseline and scenario.
6. **Features changed:** table of recomputed features (baseline → scenario), plus a visible **"held fixed"** list (NDVI, water, yield lags…).
7. **Map mode toggle:** apply the same scenario to all districts and show the yield change on the map.
8. **Scenario slots (A/B):** keep two scenarios and compare them against the baseline.

### 9.3 Validity cues
- **Extrapolation:** parts of a curve/heatmap outside the district's observed range are **hatched** and the result card shows an OUT_OF_RANGE warning.
- **Clamped inputs** show a note.
- **No live baseline** (rice `current`) disables the option with an explanation instead of returning a guess.
- The ModelNotice and the "lag features carry much of the signal" note stay visible; the UI does not hide that weather may move predictions modestly.

### 9.4 Behaviour
- Debounce slider changes (~300 ms), cancel in-flight requests, show a progress state on the result card (not the whole page).
- Grid/map requests are triggered explicitly ("Run grid") because they cost more.
- All state is in the URL; "Copy link" button.
- Rate-limit responses (429) show a clear "slow down" message.

---

## 10. States

| State | Treatment |
|---|---|
| Loading | Skeletons matching the final layout (no layout shift) |
| Cold start (free API host asleep) | After ~3 s show "Waking the server, this can take up to a minute on the first visit"; auto-retry with backoff |
| Error | Plain-language message + retry; `error.code` visible in a "details" disclosure |
| Empty | Explains why (no data, season in progress, crop not yet available) |
| Stale / flagged run | StaleBanner on every page; numbers remain visible, clearly labelled |
| Offline | Last loaded view stays; a notice explains it is not refreshing |

---

## 11. Content and language rules

- Describe model behaviour, not decisions: "the model predicts…", "the model's prediction falls by…", not "farmers should…".
- **Remove** the legacy claims that the Scenario Explorer helps with "procurement or insurance decisions" and the PDS-planning/insurance-claim advisory wording. Replace with the research-use statement: *"A research prototype. Predictions are model estimates with uncertainty and are not a substitute for local agronomic or financial advice."*
- Use the same terms everywhere: *prediction interval*, *baseline*, *scenario*, *model sensitivity*.
- Units always shown (kg/ha, mm, °C). Dates in an unambiguous format.
- Advisory text from live web grounding (Tavily) is rendered as **plain text with source links**, never as HTML (see `security.md`).
- English first; Hindi (P3) requires translating UI strings only, with a glossary for technical terms agreed before translating.

---

## 12. Performance budget (targets, to be verified)

- Server-render shell and above-the-fold content; lazy-load charts and the map.
- Initial JS per route kept small; ECharts imported per chart type (tree-shaken build) rather than the full bundle.
- Self-host fonts, `font-display: swap`, subset to needed weights.
- Cache API reads by `meta.run_id`; avoid refetching on navigation.
- Measure with Lighthouse/Web Vitals on a throttled mobile profile in CI later; set numeric budgets after the first measurement.

---

## 13. Frontend structure

```
web/
├─ app/                     routes (see §4)
├─ components/
│  ├─ charts/               YieldHistory, ShapWaterfall, AnomalyBars, Tornado, Heatmap, Bullet, RunTimeline…
│  ├─ scenario/             ControlsPanel, ResultCard, ScenarioSlots…
│  ├─ map/                  DistrictMap, RankingList
│  └─ ui/                   Card, Badge, Toggle, Table, Skeleton, Notice, RunBadge…
├─ lib/                     api client (generated types), url-state, formatters
└─ styles/tokens.css        the design tokens (§5.1)
```
Types come from the generated OpenAPI client; the UI never builds URLs or shapes by hand. Tailwind maps to the CSS variables so tokens stay in one place. URL state via a small helper (or a library such as `nuqs`).

---

## 14. Accessibility checklist

- Colour contrast ≥ 4.5:1 for text, ≥ 3:1 for graphics and UI components (see §5.2).
- Information never by colour alone (icon, label, pattern).
- Full keyboard operation, visible focus ring, logical tab order; sliders have numeric inputs.
- Charts: text summary, "View as table", ARIA labels; map has the ranking list alternative.
- Respect `prefers-reduced-motion` and `prefers-color-scheme` where applicable (dark theme is optional and only after contrast is re-audited).
- Language attribute set; text scales to 200% without loss.
- CI: axe-core checks on key pages.

---

## 15. Testing

- Component tests for charts' data mapping (inputs → series) and for formatters (rounding, units).
- Playwright smoke tests: load Overview, open a district, run a scenario, copy a link and reopen it.
- Visual regression on a few key pages (optional).
- axe accessibility scan on key pages.
- Contract: generated API types must compile (`api.md` / `ci-cd.md`).

---

## 16. Open items

1. Choose the water-blue token and re-check all chart colours for contrast and colour-vision safety.
2. Decide the basemap provider (or no basemap) after checking tile terms.
3. District boundary polygons: source, licence, name matching.
4. Add a scenario-presets endpoint (driest/wettest/hottest year per district) or derive them client-side from `/environment` data.
5. Kharif daily cumulative series (for the envelope chart).
6. Hindi: scope and glossary (P3).
7. Whether the legacy Streamlit dashboard stays as an internal analyst view or is archived.