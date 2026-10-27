"""Response envelope, pagination, sorting and ETag helpers (pure, no web framework).

Conventions follow docs/api.md section 2.
"""

import hashlib
import re
from datetime import date, datetime

from api.app.errors import ApiError

DEFAULT_LIMIT = 50
MAX_LIMIT = 200
CACHE_CONTROL = "public, s-maxage=300, stale-while-revalidate=3600"
REFERENCE_CACHE_CONTROL = "public, s-maxage=3600"
STALE_AFTER_DAYS = 7

# Training quantiles of xgb_lower / xgb_upper (quantile_alpha in src/20_finalize_tuned_model.py).
# The observed coverage on held-out data is lower than this nominal level, so clients must
# show the observed figure (meta.interval.empirical_coverage), not the nominal one.
NOMINAL_QUANTILES = [0.02, 0.98]
INTERVAL_NOTE = (
    "Band from quantile models trained at 0.02 and 0.98. Observed coverage on held-out test "
    "data is lower than the nominal level; quote the observed figure, not the nominal one."
)
CROPS = {"wheat": "Wheat", "rice": "Rice"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def iso(value):
    """JSON-safe value for dates, datetimes and UUIDs."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def build_meta(run: dict, extras: dict) -> dict:
    return {
        "run_id": iso(run["run_id"]),
        "weather_data_as_of": iso(run["weather_data_as_of"]),
        "model_version": run["model_version"],
        "prediction_basis": extras.get("prediction_basis"),
        "baseline_yield_year": extras.get("baseline_yield_year"),
        "gate_passed": run["gate_passed"],
        "generated_at": iso(run["finished_at"]),
        "interval": {
            "nominal_quantiles": NOMINAL_QUANTILES,
            "empirical_coverage": extras.get("empirical_coverage", {}),
            "note": INTERVAL_NOTE,
        },
    }


def page_info(limit: int, offset: int, total: int) -> dict:
    return {"limit": limit, "offset": offset, "total": total}


def ok(data, meta: dict | None = None, page: dict | None = None) -> dict:
    body = {"data": data}
    if page is not None:
        body["page"] = page
    if meta is not None:
        body["meta"] = meta
    return body


def resolve_sort(sort: str | None, order: str, allowed: dict, default: str) -> tuple[str, str]:
    """Map a client sort name to a trusted SQL expression. Raw column names are never accepted."""
    key = sort or default
    if key not in allowed:
        raise ApiError(
            400, "INVALID_SORT", f"Cannot sort by '{key}'.", {"allowed": sorted(allowed)}
        )
    if order not in ("asc", "desc"):
        raise ApiError(400, "INVALID_ORDER", "order must be 'asc' or 'desc'.")
    return allowed[key], order.upper()


def normalise_crop(crop: str | None) -> str | None:
    if crop is None:
        return None
    value = CROPS.get(crop.strip().lower())
    if value is None:
        raise ApiError(
            400, "INVALID_CROP", f"Unknown crop '{crop}'.", {"allowed": sorted(CROPS.values())}
        )
    return value


def etag_for(run_id, path_and_query: str) -> str:
    digest = hashlib.sha1(f"{run_id}|{path_and_query}".encode()).hexdigest()[:16]
    return f'W/"{digest}"'


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    if not if_none_match:
        return False
    candidates = [c.strip() for c in if_none_match.split(",")]
    return "*" in candidates or etag in candidates
