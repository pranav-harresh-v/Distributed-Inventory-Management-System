"""Shared pytest fixtures.

Every fixture here builds a throwaway SQLite file under pytest's
``tmp_path``. The node's real database (``data/node1.db``) is never opened
by the test suite.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

# Make the project importable regardless of where pytest is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.domain import auth
from app.storage.database import (
    create_db_engine,
    create_session_factory,
    init_db,
    sqlite_url_for_path,
)
from app.storage.models import Product, User, Warehouse

#: Password behind the `auth_user` fixture. Test-only, never seeded.
TEST_PASSWORD = "correct horse battery staple"


@pytest.fixture(scope="session", autouse=True)
def protect_the_node_database() -> Iterator[None]:
    """Fail the run if the suite writes to the real node database.

    Every fixture here builds a database under tmp_path, but a stray import
    of the module-level `engine` would quietly target data/node1.db. This
    compares the file's size and mtime across the session so that mistake
    surfaces as a test failure rather than corrupted development data.
    """
    from app.storage.database import get_database_path

    node_db = get_database_path()
    before = node_db.stat() if node_db.exists() else None
    yield
    after = node_db.stat() if node_db.exists() else None

    if before is None:
        assert after is None, f"the test suite created {node_db}"
        return
    assert after is not None, f"the test suite deleted {node_db}"
    assert (before.st_size, before.st_mtime_ns) == (
        after.st_size,
        after.st_mtime_ns,
    ), f"the test suite modified {node_db}"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path of the temporary database file for one test."""
    return tmp_path / "test_node.db"


@pytest.fixture
def db_engine(db_path: Path) -> Iterator[Engine]:
    """Engine on a temporary file, configured exactly like production."""
    engine = create_db_engine(sqlite_url_for_path(db_path))
    init_db(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def session_factory(db_engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the temporary engine."""
    return create_session_factory(db_engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A session for one test, always closed afterwards."""
    with session_factory() as db_session:
        yield db_session


@pytest.fixture
def user(session: Session) -> User:
    """A persisted user, flushed so its primary key is available."""
    record = User(
        username="alice",
        email="alice@example.com",
        password_hash="not-a-real-hash",
        role="staff",
    )
    session.add(record)
    session.flush()
    return record


@pytest.fixture
def product(session: Session) -> Product:
    record = Product(sku="SKU-1", name="Widget", unit_price=Decimal("9.99"))
    session.add(record)
    session.flush()
    return record


@pytest.fixture
def warehouse(session: Session) -> Warehouse:
    record = Warehouse(code="WH-1", name="Main", location="Bengaluru")
    session.add(record)
    session.flush()
    return record


@pytest.fixture
def auth_user(session_factory: sessionmaker[Session]) -> User:
    """A committed user whose password is :data:`TEST_PASSWORD`.

    Committed rather than flushed, because the code under test opens its own
    sessions and must be able to see this row.
    """
    with session_factory() as db:
        record = User(
            username="operator",
            email="operator@example.com",
            password_hash=auth.hash_password(TEST_PASSWORD),
            role="manager",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
