"""Authentication and session lifecycle.

This module owns every credential and session decision in the system. It is
deliberately free of gRPC imports so it can be unit-tested directly, and --
in Milestone 2 -- so the same deterministic logic can run behind a Raft
command without dragging the transport layer along.

Two different hashes are used on purpose:

* **Passwords** are hashed with bcrypt, a deliberately slow salted KDF.
  Passwords are low-entropy and guessable, so the cost per guess is the
  defence. Because every bcrypt hash carries a random salt, the same
  password hashes differently each time and cannot be looked up by value.
* **Session tokens** are hashed with a single SHA-256 pass. A token is 256
  bits from :mod:`secrets`, so there is no dictionary to slow down, and a
  deterministic digest is what allows the unique, indexed
  ``sessions.token_hash`` column to be probed in one indexed lookup.

The raw token exists only in the login response. Only its digest is stored,
so a leaked database cannot be replayed against the server.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import bcrypt
from sqlalchemy.orm import Session as DbSession

from app.storage.models import Session as SessionRecord
from app.storage.models import User, ensure_utc, utc_now
from app.storage.repositories import SessionRepository, UserRepository

__all__ = [
    "AuthError",
    "AuthenticatedContext",
    "InvalidCredentialsError",
    "InvalidSessionError",
    "LoginResult",
    "authenticate",
    "generate_session_token",
    "hash_password",
    "hash_session_token",
    "login",
    "logout",
    "revoke_session",
    "session_expires_at",
    "session_ttl",
    "verify_password",
]

# --------------------------------------------------------------------------
# Configuration
#
# Phase D defined `sessions.expires_at` as NOT NULL with no default and left
# the policy to whoever creates sessions. This is that policy.
# --------------------------------------------------------------------------
SESSION_TTL_ENV_VAR: Final[str] = "INVENTORY_SESSION_TTL_MINUTES"
DEFAULT_SESSION_TTL_MINUTES: Final[int] = 480  # 8 hours: one working shift.

#: bcrypt hashes at most 72 bytes and raises past that rather than
#: truncating, so both hashing and verification bound the input explicitly.
BCRYPT_MAX_PASSWORD_BYTES: Final[int] = 72

#: Number of random bytes behind a session token. ``token_urlsafe(32)``
#: yields 256 bits of entropy in a 43-character URL-safe string.
SESSION_TOKEN_BYTES: Final[int] = 32

#: One message for every credential failure -- unknown user, wrong password
#: and deactivated account are indistinguishable from outside, so the RPC
#: cannot be used to discover which usernames exist.
GENERIC_AUTH_FAILURE_MESSAGE: Final[str] = "Invalid username or password."

#: Likewise for session failures: unknown, revoked and expired look alike.
GENERIC_SESSION_FAILURE_MESSAGE: Final[str] = "Invalid or expired session."

#: Compared against when no user matched, so a failed login costs roughly
#: the same time whether or not the username exists. Without this, the
#: early return for an unknown user is measurably faster than a bcrypt
#: verification and leaks which usernames are real.
_TIMING_DECOY_HASH: Final[str] = (
    "$2b$12$C6UzMDM.H6dfI/f/IKcEe.2PFtjTNvjAHnGSQP1aTzDTBTGw/8xaG"
)


class AuthError(Exception):
    """Base class for every authentication failure."""


class InvalidCredentialsError(AuthError):
    """Username unknown, password wrong, or the account is deactivated."""


class InvalidSessionError(AuthError):
    """Session token missing, unknown, revoked, or expired."""


@dataclass(frozen=True)
class LoginResult:
    """Outcome of a successful login.

    ``raw_token`` is the only place the unhashed token ever exists; it is
    returned to the caller and then forgotten by the server.
    """

    user: User
    raw_token: str
    expires_at: datetime


@dataclass(frozen=True)
class AuthenticatedContext:
    """A validated session and the user it belongs to."""

    user: User
    session: SessionRecord


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------
def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash. The plaintext never leaves this function."""
    encoded = plaintext.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password exceeds bcrypt's {BCRYPT_MAX_PASSWORD_BYTES}-byte limit."
        )
    return bcrypt.hashpw(encoded, bcrypt.gensalt()).decode("utf-8")


def verify_password(plaintext: str, password_hash: str) -> bool:
    """Check a password against its stored bcrypt hash.

    Returns ``False`` rather than raising for input bcrypt cannot process
    (over-long passwords, a malformed stored hash), because from the
    caller's point of view those are simply failed authentications.
    """
    encoded = plaintext.encode("utf-8")
    if len(encoded) > BCRYPT_MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("utf-8"))
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Session tokens
# --------------------------------------------------------------------------
def generate_session_token() -> str:
    """Create a new raw session token from the OS CSPRNG."""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(raw_token: str) -> str:
    """Digest a raw token for storage and lookup.

    Deterministic and unsalted by design: the same token must always
    produce the same digest, or the indexed ``token_hash`` lookup in
    :func:`authenticate` could not find it.
    """
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def session_ttl() -> timedelta:
    """How long a newly issued session stays valid."""
    raw = os.environ.get(SESSION_TTL_ENV_VAR)
    minutes = DEFAULT_SESSION_TTL_MINUTES
    if raw:
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{SESSION_TTL_ENV_VAR} must be an integer number of minutes, "
                f"got {raw!r}"
            ) from exc
        if parsed <= 0:
            raise ValueError(f"{SESSION_TTL_ENV_VAR} must be positive, got {parsed}")
        minutes = parsed
    return timedelta(minutes=minutes)


def session_expires_at(issued_at: datetime | None = None) -> datetime:
    """Expiry timestamp for a session issued now (or at ``issued_at``)."""
    return (issued_at or utc_now()) + session_ttl()


def is_session_expired(record: SessionRecord, now: datetime | None = None) -> bool:
    """Whether a stored session has passed its expiry.

    ``expires_at`` goes through :func:`~app.storage.models.ensure_utc`
    because SQLite returns it without timezone information.
    """
    return ensure_utc(record.expires_at) <= (now or utc_now())


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------
def login(db: DbSession, username: str, password: str) -> LoginResult:
    """Verify credentials and issue a session.

    Raises :class:`InvalidCredentialsError` for an unknown user, a wrong
    password, or a deactivated account -- all three are the same failure
    from outside. The caller owns the transaction; this only flushes.
    """
    user = UserRepository(db).get_by_username(username)

    if user is None:
        # Spend comparable time to a real check so timing does not reveal
        # whether the username exists, then fail identically.
        verify_password(password, _TIMING_DECOY_HASH)
        raise InvalidCredentialsError(GENERIC_AUTH_FAILURE_MESSAGE)

    if not verify_password(password, user.password_hash):
        raise InvalidCredentialsError(GENERIC_AUTH_FAILURE_MESSAGE)

    if not user.is_active:
        raise InvalidCredentialsError(GENERIC_AUTH_FAILURE_MESSAGE)

    raw_token = generate_session_token()
    expires_at = session_expires_at()
    SessionRepository(db).create(
        user_id=user.id,
        token_hash=hash_session_token(raw_token),
        expires_at=expires_at,
    )
    return LoginResult(user=user, raw_token=raw_token, expires_at=expires_at)


def authenticate(db: DbSession, raw_token: str) -> AuthenticatedContext:
    """Validate a raw session token.

    The single entry point for session validation -- every authenticated
    RPC funnels through here rather than re-deriving the rules. Raises
    :class:`InvalidSessionError` for every failure mode, so callers cannot
    accidentally leak which check failed.
    """
    if not raw_token:
        raise InvalidSessionError(GENERIC_SESSION_FAILURE_MESSAGE)

    record = SessionRepository(db).get_by_token_hash(hash_session_token(raw_token))
    if record is None:
        raise InvalidSessionError(GENERIC_SESSION_FAILURE_MESSAGE)
    if record.revoked:
        raise InvalidSessionError(GENERIC_SESSION_FAILURE_MESSAGE)
    if is_session_expired(record):
        raise InvalidSessionError(GENERIC_SESSION_FAILURE_MESSAGE)

    user = record.user
    if user is None or not user.is_active:
        raise InvalidSessionError(GENERIC_SESSION_FAILURE_MESSAGE)

    return AuthenticatedContext(user=user, session=record)


def revoke_session(db: DbSession, context: AuthenticatedContext) -> None:
    """Revoke one already-validated session.

    Only this session is touched: the user's other sessions, on other
    devices, keep working.
    """
    SessionRepository(db).revoke(context.session)


def logout(db: DbSession, raw_token: str) -> AuthenticatedContext:
    """Validate ``raw_token`` and revoke its session.

    Validates first, so an unknown, expired or already-revoked token is
    rejected rather than silently reporting success. Callers that have
    already authenticated should use :func:`revoke_session` instead of
    paying for a second lookup.
    """
    context = authenticate(db, raw_token)
    revoke_session(db, context)
    return context
