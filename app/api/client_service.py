"""gRPC implementation of ``inventory.client.v1.ClientService``.

This layer only translates: protobuf in, domain call, protobuf out, with
domain errors mapped onto gRPC status codes. Every authentication decision
belongs to :mod:`app.domain.auth`.

Error-reporting rule: clients are told *what* category of thing went wrong
and never *why* internally. Database errors are logged server-side with a
full traceback and reported as a bare ``INTERNAL``; no SQL, exception text,
credential, hash or stack trace crosses the wire.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final, NoReturn

import grpc
from google.protobuf.timestamp_pb2 import Timestamp
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import auth
from app.storage.database import SessionLocal, session_scope
from app.storage.models import User
from proto import client_pb2, client_pb2_grpc

_logger = logging.getLogger(__name__)

#: Returned for any unexpected server-side failure. Deliberately opaque.
INTERNAL_ERROR_MESSAGE: Final[str] = "Internal server error."

#: Phase E exposes post/get/processBusinessRequest as authenticated stubs;
#: the operations behind them arrive in a later phase.
_NOT_IMPLEMENTED_TEMPLATE: Final[str] = (
    "'{rpc}' is authenticated but not implemented yet."
)

__all__ = ["ClientService", "build_user_info"]


def _abort(
    context: grpc.ServicerContext, code: grpc.StatusCode, message: str
) -> NoReturn:
    """Terminate the RPC with ``code``.

    ``abort`` raises, so this never returns; the explicit ``raise`` after it
    documents that and keeps type checkers honest if a context
    implementation ever fails to raise.
    """
    context.abort(code, message)
    raise RuntimeError("unreachable: ServicerContext.abort must raise")


def build_user_info(user: User) -> client_pb2.UserInfo:
    """Project a user row onto the wire message.

    Only the five safe columns are copied. ``password_hash`` has no field to
    land in, which is the point.
    """
    return client_pb2.UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


def _to_timestamp(value) -> Timestamp:
    timestamp = Timestamp()
    timestamp.FromDatetime(value)
    return timestamp


class ClientService(client_pb2_grpc.ClientServiceServicer):
    """Client-facing RPCs for one application node.

    The session factory is injected so tests can bind the service to a
    temporary database; production passes nothing and gets Phase D's
    ``SessionLocal``.
    """

    def __init__(self, session_factory: sessionmaker[DbSession] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    # -- shared authentication ---------------------------------------------
    @contextmanager
    def _authenticated(
        self, raw_token: str, context: grpc.ServicerContext
    ) -> Iterator[tuple[DbSession, auth.AuthenticatedContext]]:
        """Run a block inside a transaction, with the caller authenticated.

        The single place every authenticated RPC validates a session, so the
        rules cannot drift between methods. The block sees an open
        transaction that commits on clean exit and rolls back otherwise.
        """
        try:
            with session_scope(self._session_factory) as db:
                authenticated = auth.authenticate(db, raw_token)
                yield db, authenticated
        except auth.AuthError:
            _abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                auth.GENERIC_SESSION_FAILURE_MESSAGE,
            )
        except SQLAlchemyError:
            _logger.exception("database failure during session validation")
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

    def _authenticated_stub(
        self, raw_token: str, context: grpc.ServicerContext, rpc_name: str
    ) -> NoReturn:
        """Authenticate, then report the RPC as not yet implemented.

        Order matters: an invalid token must fail as UNAUTHENTICATED even
        though the operation behind it does not exist yet, so a caller can
        never use an unimplemented RPC to probe whether a token is valid.
        """
        with self._authenticated(raw_token, context):
            pass
        _abort(
            context,
            grpc.StatusCode.UNIMPLEMENTED,
            _NOT_IMPLEMENTED_TEMPLATE.format(rpc=rpc_name),
        )

    # -- RPCs ---------------------------------------------------------------
    def login(
        self, request: client_pb2.LoginRequest, context: grpc.ServicerContext
    ) -> client_pb2.LoginResponse:
        # Whitespace-only usernames are malformed input. Passwords are not
        # stripped or rejected for whitespace -- spaces are legal in a
        # password, and only a genuinely empty one is malformed.
        if not request.username.strip() or not request.password:
            _abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "Both 'username' and 'password' are required.",
            )

        try:
            with session_scope(self._session_factory) as db:
                result = auth.login(db, request.username, request.password)
                response = client_pb2.LoginResponse(
                    success=True,
                    session_token=result.raw_token,
                    message="Login successful.",
                    user=build_user_info(result.user),
                    expires_at=_to_timestamp(result.expires_at),
                )
        except auth.InvalidCredentialsError:
            _abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                auth.GENERIC_AUTH_FAILURE_MESSAGE,
            )
        except SQLAlchemyError:
            _logger.exception("database failure during login")
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

        _logger.info("login succeeded for user id=%s", result.user.id)
        return response

    def logout(
        self, request: client_pb2.LogoutRequest, context: grpc.ServicerContext
    ) -> client_pb2.StatusResponse:
        with self._authenticated(request.session_token, context) as (db, authenticated):
            auth.revoke_session(db, authenticated)
            user_id = authenticated.user.id

        _logger.info("logout revoked one session for user id=%s", user_id)
        return client_pb2.StatusResponse(success=True, message="Logged out.")

    def post(
        self, request: client_pb2.PostRequest, context: grpc.ServicerContext
    ) -> client_pb2.StatusResponse:
        self._authenticated_stub(request.session_token, context, "post")

    def get(
        self, request: client_pb2.GetRequest, context: grpc.ServicerContext
    ) -> client_pb2.GetResponse:
        self._authenticated_stub(request.session_token, context, "get")

    def processBusinessRequest(
        self, request: client_pb2.BusinessRequest, context: grpc.ServicerContext
    ) -> client_pb2.BusinessResponse:
        self._authenticated_stub(
            request.session_token, context, "processBusinessRequest"
        )
