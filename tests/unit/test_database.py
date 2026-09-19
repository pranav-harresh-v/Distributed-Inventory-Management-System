"""Engine, PRAGMA and transaction-handling tests for the storage layer."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.storage.database import (
    DEFAULT_DB_FILENAME,
    PROJECT_ROOT,
    SQLITE_PRAGMAS,
    get_database_path,
    session_scope,
    sqlite_url_for_path,
)
from app.storage.models import Product, User
from app.storage.repositories import ProductRepository, UserRepository

EXPECTED_TABLES = {
    "users",
    "sessions",
    "products",
    "warehouses",
    "inventory",
    "suppliers",
    "supplier_products",
    "orders",
    "order_items",
    "restock_orders",
    "sales_history",
}


# --- configuration ---------------------------------------------------------
def test_default_database_path_is_project_relative() -> None:
    """The node database resolves from the project root, not the cwd."""
    path = get_database_path()
    assert path == PROJECT_ROOT / "data" / DEFAULT_DB_FILENAME
    assert path.is_absolute()


def test_sqlite_url_is_well_formed(tmp_path: Path) -> None:
    url = sqlite_url_for_path(tmp_path / "node9.db")
    assert url.startswith("sqlite:///")
    assert "\\" not in url


# --- schema creation -------------------------------------------------------
def test_init_db_creates_all_tables(db_engine: Engine) -> None:
    """Requirement 1: every table can be created."""
    tables = set(inspect(db_engine).get_table_names())
    assert EXPECTED_TABLES <= tables
    assert len(EXPECTED_TABLES) == 11


def test_init_db_creates_the_database_file(db_engine: Engine, db_path: Path) -> None:
    with db_engine.connect():
        pass
    assert db_path.exists()


def test_init_db_is_idempotent(db_engine: Engine) -> None:
    from app.storage.database import init_db

    init_db(db_engine)  # must not raise on an already-populated database
    assert set(inspect(db_engine).get_table_names()) >= EXPECTED_TABLES


# --- PRAGMAs ---------------------------------------------------------------
def test_foreign_key_enforcement_is_active(session: Session) -> None:
    """Requirement 2: SQLite ignores foreign keys unless this is on."""
    assert session.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_journal_mode_is_wal(session: Session) -> None:
    assert str(session.execute(text("PRAGMA journal_mode")).scalar()).lower() == "wal"


def test_busy_timeout_is_configured(session: Session) -> None:
    assert session.execute(text("PRAGMA busy_timeout")).scalar() == 5000


def test_pragmas_apply_to_every_new_connection(db_engine: Engine) -> None:
    """A second checkout must be configured too, not just the first."""
    for _ in range(2):
        with db_engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
    assert len(SQLITE_PRAGMAS) == 3


# --- transaction handling --------------------------------------------------
def test_session_scope_commits_on_success(
    session_factory: sessionmaker[Session],
) -> None:
    with session_scope(session_factory) as scoped:
        scoped.add(
            User(
                username="committed",
                email="committed@example.com",
                password_hash="hash",
            )
        )

    with session_factory() as verify:
        assert UserRepository(verify).get_by_username("committed") is not None


def test_rollback_leaves_no_partial_writes(
    session_factory: sessionmaker[Session],
) -> None:
    """Requirement 15: a failure mid-transaction persists nothing.

    This is the invariant the order workflow will rely on later -- a failed
    order must not partially modify the database.
    """

    class BusinessRuleError(RuntimeError):
        pass

    with pytest.raises(BusinessRuleError), session_scope(session_factory) as scoped:
        scoped.add(
            User(
                username="ghost",
                email="ghost@example.com",
                password_hash="hash",
            )
        )
        scoped.flush()
        scoped.add(Product(sku="SKU-GHOST", name="Ghost", unit_price=Decimal("1.00")))
        scoped.flush()
        raise BusinessRuleError("failed after writing two rows")

    with session_factory() as verify:
        assert verify.scalars(select(User)).all() == []
        assert verify.scalars(select(Product)).all() == []


def test_session_scope_reraises_database_errors(
    session_factory: sessionmaker[Session],
) -> None:
    """Constraint violations surface to the caller, never swallowed."""
    from sqlalchemy.exc import IntegrityError

    with session_factory() as setup:
        setup.add(User(username="dup", email="dup@example.com", password_hash="hash"))
        setup.commit()

    with pytest.raises(IntegrityError), session_scope(session_factory) as scoped:
        scoped.add(
            User(username="dup", email="other@example.com", password_hash="hash")
        )


# --- repository layer ------------------------------------------------------
def test_repositories_do_not_commit(session_factory: sessionmaker[Session]) -> None:
    """The caller owns the transaction boundary, not the repository."""
    with session_factory() as writer:
        ProductRepository(writer).create(
            sku="SKU-UNCOMMITTED", name="Temp", unit_price=Decimal("5.00")
        )
        # No commit; closing the session discards the flush.

    with session_factory() as verify:
        assert ProductRepository(verify).get_by_sku("SKU-UNCOMMITTED") is None


def test_repository_round_trip(session_factory: sessionmaker[Session]) -> None:
    with session_scope(session_factory) as scoped:
        repo = ProductRepository(scoped)
        repo.create(sku="SKU-A", name="Alpha", unit_price=Decimal("10.00"))
        repo.create(
            sku="SKU-B", name="Beta", unit_price=Decimal("20.00"), is_active=False
        )

    with session_factory() as verify:
        repo = ProductRepository(verify)
        assert [p.sku for p in repo.list()] == ["SKU-A", "SKU-B"]
        assert [p.sku for p in repo.list(active_only=True)] == ["SKU-A"]
        assert repo.get_by_sku("SKU-A") is not None
        assert repo.get_by_sku("SKU-MISSING") is None
