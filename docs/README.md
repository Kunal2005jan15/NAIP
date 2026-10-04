# NAIP Documentation

Status date: 2026-10-04. Docs mark each item as ✅ Built, 🔧 Built but needs refactor, or 📐 Designed, not built.

| Doc | What it answers |
|---|---|
| [system-architecture.md](system-architecture.md) | What the parts are, where they run, how data flows |
| [systemdesign.md](systemdesign.md) | Repo restructure, pipeline and scenario internals, live rice step, roadmap |
| [database.md](database.md) | Postgres schema, retention, run promotion, reference data in git |
| [api.md](api.md) | `/api/v1` conventions, endpoints, scenario engine contract |
| [ui-ux.md](ui-ux.md) | Pages, charts, scenario UX, design system, accessibility |
| [ci-cd.md](ci-cd.md) | GitHub Actions, scheduled refresh, deploys, rollback |
| [security.md](security.md) | Findings, secrets, supply chain, model safety, responsible use |

Reading order for a newcomer: architecture → systemdesign → database → api → ui-ux → ci-cd → security.