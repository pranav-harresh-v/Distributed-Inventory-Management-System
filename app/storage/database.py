"""SQLAlchemy engine, session and schema bootstrap for a single node.

Architectural rule (see CLAUDE.md): SQLite is *local to each application
node*. This module therefore resolves exactly one database file per
process and never exposes a connection to a database shared between
nodes. Replication between nodes is Raft's job in Milestone 2, not the
storage layer's.

The database path is resolved relative to the project root so that the
same code works no matter which directory the caller runs from.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# app/storage/database.py -> app/storage -> app -> <project root>
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DEFAULT_DB_FILENAME: Final[str] = "node1.db"

#: Optional override so a second/third node can run from the same checkout
#: with its own local file (e.g. ``data/node2.db``). Unset -> node1.
DB_PATH_ENV_VAR: Final[str] = "INVENTORY_DB_PATH"

#: Applied to *every* pooled connection, because SQLite settings are
#: per-connection rather than per-database.
#:  - foreign_keys:   SQLite ignores FK constraints unless switched on.
#:  - journal_mode:   WAL lets readers proceed during a writer's transaction.
#:  - busy_timeout:   wait instead of failing instantly on a locked database.
SQLITE_PRAGMAS: Final[tuple[str, ...]] = (
    "PRAGMA foreign_keys=ON",
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
)

#: ``check_same_thread`` must be off because the gRPC server will hand
#: sessions to worker threads; SQLAlchemy's pool still keeps one
#: connection per checkout, so this stays safe.
CONNECT_ARGS: Final[dict[str, Any]] = {"check_same_thread": False}


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model in this package."""


def sqlite_url_for_path(path: str | os.PathLike[str]) -> str:
    """Build a SQLite URL from a filesystem path.

    ``as_posix()`` normalises Windows separators so a drive-letter path
    stays valid inside the URL.
    """
    return f"sqlite:///{Path(path).as_posix()}"


def get_database_path() -> Path:
    """Absolute path of this node's SQLite file."""
    override = os.environ.get(DB_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return DATA_DIR / DEFAULT_DB_FILENAME


def get_database_url() -> str:
    """SQLAlchemy URL for this node's SQLite file."""
    return sqlite_url_for_path(get_database_path())


def _apply_sqlite_pragmas(dbapi_connection: Any, connection_record: Any) -> None:
    """``connect`` event hook applying :data:`SQLITE_PRAGMAS`.

    PRAGMAs are one of the few things that genuinely require raw SQL.
    """
    cursor = dbapi_connection.cursor()
    try:
        for pragma in SQLITE_PRAGMAS:
            cursor.execute(pragma)
    finally:
        cursor.close()


def create_db_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create an engine with this project's SQLite settings applied.

    Tests use this factory with a temporary path so that they exercise the
    same PRAGMA/connect configuration as production code.
    """
    new_engine = create_engine(
        url or get_database_url(),
        echo=echo,
        connect_args=CONNECT_ARGS,
    )
    event.listen(new_engine, "connect", _apply_sqlite_pragmas)
    return new_engine


def create_session_factory(bound_engine: Engine) -> sessionmaker[Session]:
    """Build a session factory bound to ``bound_engine``.

    ``expire_on_commit=False`` keeps loaded attributes readable after the
    surrounding :func:`session_scope` commits and closes the session.
    """
    return sessionmaker(bind=bound_engine, autoflush=False, expire_on_commit=False)


#: Default engine/session factory for this node. Engines connect lazily, so
#: importing this module does not create the database file.
engine: Final[Engine] = create_db_engine()
SessionLocal: Final[sessionmaker[Session]] = create_session_factory(engine)


def _ensure_database_directory(target_engine: Engine) -> None:
    """Create the parent directory of a file-backed SQLite database."""
    database = target_engine.url.database
    if database and database != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    """Transactional scope around a series of operations.

    Commits on success, rolls back and **re-raises** on failure -- database
    errors are never swallowed here -- and always closes the session, so no
    long-lived global session ever exists.
    """
    factory = session_factory or SessionLocal
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """Generator form of :func:`session_scope` for injection-style callers."""
    with session_scope() as session:
        yield session


def init_db(target_engine: Engine | None = None) -> Engine:
    """Create every table declared on :class:`Base` if it does not exist.

    Importing the models module here (rather than at module import time)
    registers the mappings on ``Base.metadata`` without a circular import.
    """
    from app.storage import models  # noqa: F401  (populates Base.metadata)

    bound = target_engine or engine
    _ensure_database_directory(bound)
    Base.metadata.create_all(bound)
    return bound
