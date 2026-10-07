"""Connectivity check: python -m pipeline.db.check [owner|writer|reader]."""

import sys

from pipeline.db.connection import connect


def main() -> None:
    role = sys.argv[1] if len(sys.argv) > 1 else "owner"
    with connect(role) as conn:
        version, user, now = conn.execute("select version(), current_user, now()").fetchone()
    print(f"Connected as {user} (role arg: {role})")
    print(f"Server time: {now}")
    print(f"Server: {version.split(',')[0]}")


if __name__ == "__main__":
    main()
