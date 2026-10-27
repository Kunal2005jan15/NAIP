"""Read-only /api/v1 endpoints (slice 1: system, districts, predictions)."""

from datetime import date
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response

from api.app import repository
from api.app.deps import get_conn
from api.app.envelope import (
    CACHE_CONTROL,
    DEFAULT_LIMIT,
    MAX_LIMIT,
    REFERENCE_CACHE_CONTROL,
    STALE_AFTER_DAYS,
    build_meta,
    etag_for,
    etag_matches,
    iso,
    normalise_crop,
    ok,
    page_info,
    resolve_sort,
    slugify,
)
from api.app.errors import ApiError

router = APIRouter(prefix="/api/v1")

Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Limit = Annotated[int, Query(ge=1, le=MAX_LIMIT)]
Offset = Annotated[int, Query(ge=0)]

# Metadata only changes when a new run is promoted, so keep the latest one in memory.
_META_CACHE: dict = {}

LIMITATIONS = [
    "Research prototype: model estimates with uncertainty, not advice for operational decisions.",
    "Live predictions cover Wheat (Rabi) only.",
    "Live predictions combine current weather with the most recent published yield year "
    "(see meta.baseline_yield_year); government yield statistics lag by 1-2 years.",
    "The prediction band comes from separate quantile models; the point estimate can fall "
    "outside it (see contains_point).",
]


def require_run(conn) -> dict:
    run = repository.latest_run(conn)
    if run is None:
        raise ApiError(409, "NO_PROMOTED_RUN", "No refresh run has been promoted yet.")
    return run


def meta_for(conn, run: dict) -> dict:
    key = run["run_id"]
    if key not in _META_CACHE:
        _META_CACHE.clear()
        extras = repository.run_extras(conn, key, run["model_version"])
        _META_CACHE[key] = build_meta(run, extras)
    return dict(_META_CACHE[key])


def cached_json(request: Request, run: dict, payload: dict) -> Response:
    """JSON with an ETag derived from run_id + query; answers 304 when the client is current."""
    target = request.url.path + "?" + request.url.query
    etag = etag_for(run["run_id"], target)
    headers = {"ETag": etag, "Cache-Control": CACHE_CONTROL}
    if etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


def district_slug(state: str, district: str) -> str:
    return f"{slugify(state)}/{slugify(district)}"


# ---------------------------------------------------------------- system
@router.get("/health")
def health(conn: Conn):
    run = repository.latest_run(conn)
    if run is None:
        data = {"status": "no_promoted_run", "database": "ok", "latest_run": None}
    else:
        as_of = run["weather_data_as_of"]
        age = (date.today() - as_of).days if as_of else None
        data = {
            "status": "stale" if age is not None and age > STALE_AFTER_DAYS else "ok",
            "database": "ok",
            "data_age_days": age,
            "latest_run": {
                "run_id": iso(run["run_id"]),
                "status": run["status"],
                "gate_passed": run["gate_passed"],
                "weather_data_as_of": iso(run["weather_data_as_of"]),
                "finished_at": iso(run["finished_at"]),
            },
        }
    return JSONResponse(ok(data), headers={"Cache-Control": "no-store"})


@router.get("/meta")
def meta(request: Request, conn: Conn):
    run = require_run(conn)
    data = {"limitations": LIMITATIONS, "stale_after_days": STALE_AFTER_DAYS}
    return cached_json(request, run, ok(data, meta=meta_for(conn, run)))


# ---------------------------------------------------------------- districts
@router.get("/districts")
def districts(
    conn: Conn,
    state: str | None = None,
    sort: str | None = None,
    order: str = "asc",
    limit: Limit = DEFAULT_LIMIT,
    offset: Offset = 0,
):
    expr, direction = resolve_sort(sort, order, repository.DISTRICT_SORTS, "state")
    rows, total = repository.list_districts(
        conn, slugify(state) if state else None, expr, direction, limit, offset
    )
    body = ok(rows, page=page_info(limit, offset, total))
    return JSONResponse(body, headers={"Cache-Control": REFERENCE_CACHE_CONTROL})


@router.get("/districts/{state}/{district}")
def district_detail(state: str, district: str, conn: Conn):
    slug = district_slug(state, district)
    row = repository.get_district(conn, slug)
    if row is None:
        raise ApiError(404, "DISTRICT_NOT_FOUND", f"No district '{slug}'.", {"slug": slug})
    run = repository.latest_run(conn)
    district_id = row.pop("district_id")
    row["history_coverage"] = repository.district_coverage(conn, district_id)
    crops = repository.live_crops(conn, run["run_id"], district_id) if run else []
    row["live_prediction_crops"] = crops
    return JSONResponse(ok(row), headers={"Cache-Control": REFERENCE_CACHE_CONTROL})


# ---------------------------------------------------------------- predictions
@router.get("/predictions")
def predictions(
    request: Request,
    conn: Conn,
    crop: str | None = None,
    state: str | None = None,
    sort: str | None = None,
    order: str = "asc",
    limit: Limit = DEFAULT_LIMIT,
    offset: Offset = 0,
):
    crop_name = normalise_crop(crop)
    expr, direction = resolve_sort(sort, order, repository.PREDICTION_SORTS, "district")
    run = require_run(conn)
    rows, total = repository.list_predictions(
        conn,
        run["run_id"],
        crop_name,
        slugify(state) if state else None,
        expr,
        direction,
        limit,
        offset,
    )
    meta_block = meta_for(conn, run)
    if total == 0 and crop_name == "Rice":
        meta_block["notice"] = "Live predictions are currently available for Wheat only."
    data = [repository.shape_prediction(r) for r in rows]
    body = ok(data, meta=meta_block, page=page_info(limit, offset, total))
    return cached_json(request, run, body)


@router.get("/districts/{state}/{district}/prediction")
def district_prediction(
    request: Request, state: str, district: str, conn: Conn, crop: str = "Wheat"
):
    crop_name = normalise_crop(crop)
    slug = district_slug(state, district)
    if repository.get_district(conn, slug) is None:
        raise ApiError(404, "DISTRICT_NOT_FOUND", f"No district '{slug}'.", {"slug": slug})
    run = require_run(conn)
    row = repository.get_prediction(conn, run["run_id"], slug, crop_name)
    if row is None:
        raise ApiError(
            404,
            "NO_LIVE_PREDICTION",
            f"No live {crop_name} prediction for '{slug}' in the latest run.",
            {"slug": slug, "crop": crop_name},
        )
    body = ok(repository.shape_prediction(row, detail=True), meta=meta_for(conn, run))
    return cached_json(request, run, body)
