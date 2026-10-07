"""Database connection helper.

Reads the connection string for a role from the environment, loading a
local .env file if one exists. Credentials never appear in code.

Roles (see docs/database.md section 7):
  owner   runs migrations and seeding (local use only)
  writer  the refresh job: INSERT/UPDATE on run tables
  reader  the API: SELECT only
"""

import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

_ENV_VARS = {
    "owner": "DATABASE_URL_OWNER",
    "writer": "DATABASE_URL_WRITER",
    "reader": "DATABASE_URL_READER",
}


def get_url(role: str = "owner") -> str:
    """Return the connection URL for a role, or raise a clear error."""
    if role not in _ENV_VARS:
        raise ValueError(f"Unknown role {role!r}; expected one of {sorted(_ENV_VARS)}")
    var = _ENV_VARS[role]
    url = os.environ.get(var)
    if not url:
        raise RuntimeError(f"{var} is not set. Copy .env.example to .env and fill it in.")
    return url


def connect(role: str = "owner") -> psycopg.Connection:
    """Open a connection as the given role (15 s timeout for cold starts)."""
    return psycopg.connect(get_url(role), connect_timeout=15)
