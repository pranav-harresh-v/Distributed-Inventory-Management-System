"""Create this node's local SQLite database and all of its tables.

Usage (from anywhere)::

    python scripts/init_db.py

Safe to re-run: ``create_all`` only creates tables that are missing, and
the script inserts no data. Use ``scripts/seed_data.py`` for demo rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when this file is run as a script rather than
# as a module (``python scripts/init_db.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import inspect

from app.storage.database import Base, engine, get_database_path, init_db


def main() -> int:
    db_path = get_database_path()
    # SQLite creates the file but not its directory.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    init_db(engine)

    created = sorted(inspect(engine).get_table_names())
    expected = sorted(Base.metadata.tables)
    missing = [name for name in expected if name not in created]

    print(f"Initialised database at {db_path}")
    print(f"Tables present ({len(created)}): {', '.join(created)}")
    if missing:
        print(f"ERROR: expected tables missing: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
