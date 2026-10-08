"""Check role privileges after migrating: python -m pipeline.db.verify_grants."""

import psycopg

from pipeline.db.connection import connect

READER_INSERT = "INSERT INTO districts (district_id, slug, state, name) VALUES (-1, 'x', 'x', 'x')"
WRITER_INSERT = (
    "INSERT INTO runs (run_id, started_at, status) VALUES (gen_random_uuid(), now(), 'running')"
)


def main() -> None:
    with connect("reader") as conn:
        n = conn.execute("SELECT count(*) FROM districts").fetchone()[0]
        print(f"OK: reader can SELECT districts ({n} rows)")
        try:
            conn.execute(READER_INSERT)
            print("FAIL: reader was able to INSERT")
        except psycopg.errors.InsufficientPrivilege:
            print("OK: reader cannot INSERT")
        conn.rollback()

    with connect("writer") as conn:
        conn.execute(WRITER_INSERT)
        print("OK: writer can INSERT into runs (rolled back)")
        conn.rollback()
        try:
            conn.execute("DELETE FROM districts")
            print("FAIL: writer was able to DELETE")
        except psycopg.errors.InsufficientPrivilege:
            print("OK: writer cannot DELETE")
        conn.rollback()


if __name__ == "__main__":
    main()
