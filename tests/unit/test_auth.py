"""Unit tests for the authentication domain logic (no gRPC involved)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import auth
from app.storage.database import session_scope
from app.storage.models import Session as SessionRecord
from app.storage.models import User, ensure_utc, utc_now
from app.storage.repositories import SessionRepository
from tests.conftest import TEST_PASSWORD


# --- password hashing ------------------------------------------------------
def test_hash_password_produces_bcrypt_hash() -> None:
    hashed = auth.hash_password("hunter2")
    assert hashed.startswith("$2b$")
    assert "hunter2" not in hashed


def test_hash_password_is_salted() -> None:
    """Two hashes of the same password differ, so hashes are not lookup keys."""
    assert auth.hash_password("same") != auth.hash_password("same")


def test_verify_password_round_trip() -> None:
    hashed = auth.hash_password(TEST_PASSWORD)
    assert auth.verify_password(TEST_PASSWORD, hashed) is True
    assert auth.verify_password("wrong", hashed) is False


def test_verify_password_rejects_overlong_input() -> None:
    """bcrypt raises past 72 bytes; login must not crash on a long password."""
    hashed = auth.hash_password("short")
    assert auth.verify_password("a" * 200, hashed) is False


def test_hash_password_rejects_overlong_input() -> None:
    with pytest.raises(ValueError, match="72-byte"):
        auth.hash_password("a" * 73)


def test_verify_password_rejects_malformed_stored_hash() -> None:
    assert auth.verify_password("anything", "not-a-bcrypt-hash") is False


# --- token generation and hashing -----------------------------------------
def test_generated_tokens_are_unique_and_long() -> None:
    tokens = {auth.generate_session_token() for _ in range(50)}
    assert len(tokens) == 50
    assert all(len(t) >= 40 for t in tokens)


def test_token_hash_is_deterministic() -> None:
    """Required: the indexed token_hash lookup depends on this."""
    token = auth.generate_session_token()
    assert auth.hash_session_token(token) == auth.hash_session_token(token)


def test_token_hash_differs_per_token_and_hides_the_token() -> None:
    a, b = auth.generate_session_token(), auth.generate_session_token()
    digest = auth.hash_session_token(a)
    assert digest != auth.hash_session_token(b)
    assert a not in digest
    assert len(digest) == 64  # sha256 hex


# --- TTL policy ------------------------------------------------------------
def test_default_session_ttl() -> None:
    assert auth.session_ttl() == timedelta(minutes=auth.DEFAULT_SESSION_TTL_MINUTES)


def test_session_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(auth.SESSION_TTL_ENV_VAR, "15")
    assert auth.session_ttl() == timedelta(minutes=15)


@pytest.mark.parametrize("bad", ["0", "-5", "abc"])
def test_session_ttl_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv(auth.SESSION_TTL_ENV_VAR, bad)
    with pytest.raises(ValueError):
        auth.session_ttl()


# --- login -----------------------------------------------------------------
def test_login_succeeds_and_persists_only_a_hash(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with session_scope(session_factory) as db:
        result = auth.login(db, auth_user.username, TEST_PASSWORD)
        raw_token = result.raw_token

    assert result.user.id == auth_user.id
    with session_factory() as db:
        stored = SessionRepository(db).list()
        assert len(stored) == 1
        record = stored[0]
        # The raw token must not be recoverable from the database.
        assert record.token_hash != raw_token
        assert record.token_hash == auth.hash_session_token(raw_token)
        assert record.revoked is False
        assert record.user_id == auth_user.id
        assert ensure_utc(record.expires_at) > utc_now()


def test_login_rejects_unknown_user(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with (
        pytest.raises(auth.InvalidCredentialsError),
        session_scope(session_factory) as db,
    ):
        auth.login(db, "nobody", TEST_PASSWORD)


def test_login_rejects_wrong_password(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with (
        pytest.raises(auth.InvalidCredentialsError),
        session_scope(session_factory) as db,
    ):
        auth.login(db, auth_user.username, "not the password")


def test_login_rejects_inactive_user(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with session_factory() as db:
        db.get(User, auth_user.id).is_active = False
        db.commit()

    with (
        pytest.raises(auth.InvalidCredentialsError),
        session_scope(session_factory) as db,
    ):
        auth.login(db, auth_user.username, TEST_PASSWORD)


def test_failed_login_messages_are_identical(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    """Unknown user and wrong password must be indistinguishable."""
    messages = []
    for username, password in [("nobody", "x"), (auth_user.username, "wrong")]:
        with session_factory() as db:
            try:
                auth.login(db, username, password)
            except auth.InvalidCredentialsError as exc:
                messages.append(str(exc))
    assert messages[0] == messages[1] == auth.GENERIC_AUTH_FAILURE_MESSAGE


def test_failed_login_writes_no_session(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with (
        pytest.raises(auth.InvalidCredentialsError),
        session_scope(session_factory) as db,
    ):
        auth.login(db, auth_user.username, "wrong")
    with session_factory() as db:
        assert SessionRepository(db).list() == []


# --- session validation ----------------------------------------------------
def _issue_token(session_factory: sessionmaker[DbSession], user: User) -> str:
    with session_scope(session_factory) as db:
        return auth.login(db, user.username, TEST_PASSWORD).raw_token


def test_authenticate_accepts_a_fresh_token(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    token = _issue_token(session_factory, auth_user)
    with session_scope(session_factory) as db:
        context = auth.authenticate(db, token)
        assert context.user.id == auth_user.id
        assert context.session.revoked is False


@pytest.mark.parametrize("token", ["", "not-a-real-token"])
def test_authenticate_rejects_empty_and_unknown_tokens(
    session_factory: sessionmaker[DbSession], auth_user: User, token: str
) -> None:
    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.authenticate(db, token)


def test_authenticate_rejects_revoked_session(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    token = _issue_token(session_factory, auth_user)
    with session_scope(session_factory) as db:
        auth.logout(db, token)

    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.authenticate(db, token)


def test_authenticate_rejects_expired_session(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    """Expiry is enforced on read, not by a background job."""
    token = _issue_token(session_factory, auth_user)
    with session_factory() as db:
        record = SessionRepository(db).get_by_token_hash(auth.hash_session_token(token))
        record.expires_at = utc_now() - timedelta(seconds=1)
        db.commit()

    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.authenticate(db, token)


def test_expiry_comparison_survives_naive_storage(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    """SQLite returns naive datetimes; comparing them must not raise."""
    token = _issue_token(session_factory, auth_user)
    with session_factory() as db:
        record = SessionRepository(db).get_by_token_hash(auth.hash_session_token(token))
        assert record.expires_at.tzinfo is None  # naive, as stored
        assert auth.is_session_expired(record) is False


def test_authenticate_rejects_deactivated_user_mid_session(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    """Deactivating a user invalidates tokens already issued to them."""
    token = _issue_token(session_factory, auth_user)
    with session_factory() as db:
        db.get(User, auth_user.id).is_active = False
        db.commit()

    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.authenticate(db, token)


# --- logout ----------------------------------------------------------------
def test_logout_revokes_the_token_immediately(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    token = _issue_token(session_factory, auth_user)
    with session_scope(session_factory) as db:
        auth.logout(db, token)

    with session_factory() as db:
        record = SessionRepository(db).get_by_token_hash(auth.hash_session_token(token))
        assert record.revoked is True


def test_logout_leaves_other_sessions_alone(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    """Signing out on one device must not sign the user out everywhere."""
    first = _issue_token(session_factory, auth_user)
    second = _issue_token(session_factory, auth_user)

    with session_scope(session_factory) as db:
        auth.logout(db, first)

    with session_scope(session_factory) as db:
        assert auth.authenticate(db, second).user.id == auth_user.id

    with session_factory() as db:
        records = SessionRepository(db).list_for_user(auth_user.id)
        assert sorted(r.revoked for r in records) == [False, True]


def test_logout_rejects_an_unknown_token(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.logout(db, "never-issued")


def test_logout_is_not_silently_repeatable(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    token = _issue_token(session_factory, auth_user)
    with session_scope(session_factory) as db:
        auth.logout(db, token)
    with pytest.raises(auth.InvalidSessionError), session_scope(session_factory) as db:
        auth.logout(db, token)


# --- repository ------------------------------------------------------------
def test_session_repository_create_and_lookup(
    session_factory: sessionmaker[DbSession], auth_user: User
) -> None:
    with session_scope(session_factory) as db:
        repo = SessionRepository(db)
        repo.create(
            user_id=auth_user.id,
            token_hash="deadbeef",
            expires_at=utc_now() + timedelta(hours=1),
        )

    with session_factory() as db:
        repo = SessionRepository(db)
        record = repo.get_by_token_hash("deadbeef")
        assert isinstance(record, SessionRecord)
        assert record.revoked is False
        assert repo.get_by_token_hash("missing") is None
