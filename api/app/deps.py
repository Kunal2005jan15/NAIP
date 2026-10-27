"""Request dependencies: one short-lived read-only database connection per request."""

from collections.abc import Iterator

import psycopg

from api.app.errors import ApiError
from pipeline.db.connection import connect


def get_conn() -> Iterator[psycopg.Connection]:
    try:
        conn = connect("reader")
    except RuntimeError as exc:  # DATABASE_URL_READER is not set
        raise ApiError(503, "DB_NOT_CONFIGURED", "The database is not configured.") from exc
    except psycopg.OperationalError as exc:
        raise ApiError(503, "DB_UNAVAILABLE", "The database is unavailable.") from exc
    try:
        yield conn
    finally:
        conn.close()
