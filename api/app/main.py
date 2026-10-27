"""NAIP API application factory.

Run locally:  uvicorn api.app.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

import os

import psycopg
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.app.errors import ApiError
from api.app.routers.v1 import router as v1_router

DESCRIPTION = (
    "Read-only API for NAIP/ANNA district-level yield predictions and alerts. "
    "Research prototype: estimates carry uncertainty and are not operational advice."
)


def cors_origins() -> list[str]:
    raw = os.environ.get("NAIP_CORS_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def error_response(status: int, code: str, message: str, details: dict | None = None):
    return JSONResponse(ApiError(status, code, message, details).body(), status_code=status)


def create_app() -> FastAPI:
    app = FastAPI(title="NAIP API", version="1.0.0", description=DESCRIPTION)
    app.add_middleware(
        CORSMiddleware, allow_origins=cors_origins(), allow_methods=["GET"], allow_headers=["*"]
    )

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        details = {"errors": jsonable_encoder(exc.errors())}
        return error_response(422, "VALIDATION_ERROR", "Invalid request parameters.", details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException):
        codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        code = codes.get(exc.status_code, "HTTP_ERROR")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(psycopg.OperationalError)
    async def handle_db_down(request: Request, exc: psycopg.OperationalError):
        return error_response(503, "DB_UNAVAILABLE", "The database is unavailable.")

    app.include_router(v1_router)
    return app


app = create_app()
