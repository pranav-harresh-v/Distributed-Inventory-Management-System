"""Seeding behaviour for the two development accounts.

Every test runs against a throwaway SQLite session from ``tests/conftest.py``
-- ``scripts/seed_data.py``'s own ``main()`` is never called here, because
that would open ``data/node1.db``. The seeding *functions* take a session, so
they can be driven against a temporary database directly.

What is worth pinning down is idempotency. The script's contract is that
running it twice produces exactly the same rows as running it once, and the
subtle half of that is the password: bcrypt salts differ on every call, so a
seed that re-hashed unconditionally would rewrite the row each run and
silently replace a password a developer had changed by hand.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.domain import auth
from app.storage.models import ADMIN_ROLE, STAFF_ROLE, USER_ROLES
from app.storage.repositories import UserRepository
from scripts.seed_data import (
    DEV_ADMIN_EMAIL,
    DEV_ADMIN_PASSWORD,
    DEV_ADMIN_USERNAME,
    DEV_USER_EMAIL,
    DEV_USER_PASSWORD,
    DEV_USER_USERNAME,
    seed_admin_user,
    seed_demo_user,
)


def user(session: Session, username: str) -> Any:
    return UserRepository(session).get_by_username(username)


# --- the demo user ---------------------------------------------------------
def test_the_demo_user_is_created_on_a_fresh_database(session: Session) -> None:
    assert seed_demo_user(session) is True
    session.flush()

    created = user(session, DEV_USER_USERNAME)
    assert created is not None
    assert created.username == "user1"
    assert created.email == DEV_USER_EMAIL
    assert created.role == STAFF_ROLE
    assert created.is_active is True


def test_the_demo_user_gets_a_role_the_schema_already_defines(
    session: Session,
) -> None:
    """No new role is invented; `staff` is one of the three that existed."""
    seed_demo_user(session)
    session.flush()

    assert user(session, DEV_USER_USERNAME).role in USER_ROLES
    assert STAFF_ROLE == "staff"


def test_the_demo_password_is_hashed_never_stored_in_the_clear(
    session: Session,
) -> None:
    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert stored != DEV_USER_PASSWORD
    assert DEV_USER_PASSWORD not in stored
    assert stored.startswith("$2")  # bcrypt


def test_the_demo_user_can_actually_log_in(session: Session) -> None:
    """Hashed by the same code that verifies it, so the account really works."""
    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert auth.verify_password(DEV_USER_PASSWORD, stored) is True
    assert auth.verify_password("not-the-password", stored) is False


def test_seeding_the_demo_user_twice_adds_one_row(session: Session) -> None:
    assert seed_demo_user(session) is True
    session.flush()
    assert seed_demo_user(session) is False
    session.flush()

    assert len(UserRepository(session).list()) == 1


def test_a_rerun_does_not_rewrite_an_existing_demo_password(
    session: Session,
) -> None:
    """The subtle half of idempotency.

    bcrypt salts differ per call, so an unconditional re-hash would change
    the stored value on every run -- and would clobber a password the
    developer had changed by hand.
    """
    seed_demo_user(session)
    session.flush()
    before = user(session, DEV_USER_USERNAME).password_hash

    seed_demo_user(session)
    session.flush()

    assert user(session, DEV_USER_USERNAME).password_hash == before


def test_an_environment_override_sets_the_demo_password(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEED_USER_PASSWORD", "Another-Dev-Only-456")

    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert auth.verify_password("Another-Dev-Only-456", stored) is True
    assert auth.verify_password(DEV_USER_PASSWORD, stored) is False


# --- the admin account, unchanged -----------------------------------------
def test_the_admin_account_is_unchanged(session: Session) -> None:
    """The constants the rest of the tooling imports must not have moved."""
    assert DEV_ADMIN_USERNAME == "admin"
    assert DEV_ADMIN_EMAIL == "admin@example.com"
    assert DEV_ADMIN_PASSWORD == "ChangeMe-Dev-Only-123"

    assert seed_admin_user(session) is True
    session.flush()

    admin = user(session, DEV_ADMIN_USERNAME)
    assert admin.role == ADMIN_ROLE
    assert auth.verify_password(DEV_ADMIN_PASSWORD, admin.password_hash) is True


def test_the_admin_seed_is_still_idempotent(session: Session) -> None:
    assert seed_admin_user(session) is True
    session.flush()
    assert seed_admin_user(session) is False
    session.flush()

    assert len(UserRepository(session).list()) == 1


def test_seeding_the_demo_user_leaves_the_admin_alone(session: Session) -> None:
    """Adding the second account must not disturb the first."""
    seed_admin_user(session)
    session.flush()
    before = user(session, DEV_ADMIN_USERNAME).password_hash

    seed_demo_user(session)
    session.flush()

    admin = user(session, DEV_ADMIN_USERNAME)
    assert admin.password_hash == before
    assert admin.role == ADMIN_ROLE
    assert admin.email == DEV_ADMIN_EMAIL


# --- both together ---------------------------------------------------------
def test_both_accounts_coexist_with_distinct_identities(session: Session) -> None:
    """`users.username` and `users.email` are both unique in the schema."""
    seed_admin_user(session)
    seed_demo_user(session)
    session.flush()

    rows = UserRepository(session).list()
    assert len(rows) == 2
    assert {row.username for row in rows} == {"admin", "user1"}
    assert len({row.email for row in rows}) == 2
    assert {row.role for row in rows} == {ADMIN_ROLE, STAFF_ROLE}


def test_a_full_reseed_adds_nothing_the_second_time(session: Session) -> None:
    """What `python scripts/seed_data.py` twice must do to the user rows."""
    first = (int(seed_admin_user(session)), int(seed_demo_user(session)))
    session.flush()
    second = (int(seed_admin_user(session)), int(seed_demo_user(session)))
    session.flush()

    assert first == (1, 1)
    assert second == (0, 0)
    assert len(UserRepository(session).list()) == 2
