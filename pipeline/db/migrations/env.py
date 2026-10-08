"""Alembic environment: runs migrations as the owner role."""

from alembic import context
from sqlalchemy import create_engine, pool

from pipeline.db.connection import get_url


def _sqlalchemy_url() -> str:
    url = get_url("owner")
    # SQLAlchemy defaults to psycopg2; this project uses psycopg 3.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(url=_sqlalchemy_url(), literal_binds=True, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sqlalchemy_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
