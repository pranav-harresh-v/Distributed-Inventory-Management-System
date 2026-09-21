"""Unit tests for the ClientService servicer, driven without a network.

The servicer is called directly with a stand-in context, so these tests
cover status-code mapping and response shaping without the cost of starting
a server. End-to-end behaviour over a real channel lives in
tests/integration/test_grpc_client_service.py.
"""

from __future__ import annotations

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth
from app.storage.models import User
from proto import client_pb2
from tests.conftest import TEST_PASSWORD


class FakeContext:
    """Minimal ServicerContext stand-in.

    ``abort`` raises, exactly as grpc's real context does; without that the
    servicer would carry on past a failure and the tests would not reflect
    real behaviour.
    """

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise _Aborted(code, details)


class _Aborted(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


@pytest.fixture
def service(session_factory: sessionmaker[DbSession]) -> ClientService:
    return ClientService(session_factory)


def _login(service: ClientService, username: str, password: str) -> str:
    response = service.login(
        client_pb2.LoginRequest(username=username, password=password), FakeContext()
    )
    return response.session_token


# --- login -----------------------------------------------------------------
def test_login_returns_token_and_safe_user(
    service: ClientService, auth_user: User
) -> None:
    context = FakeContext()
    response = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        context,
    )

    assert response.success is True
    assert response.HasField("session_token")
    assert response.session_token
    assert response.user.username == auth_user.username
    assert response.user.role == "manager"
    assert response.user.is_active is True
    assert response.HasField("expires_at")
    assert context.code is None


def test_login_response_never_carries_secrets(
    service: ClientService, auth_user: User
) -> None:
    response = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    )
    serialized = response.SerializeToString()
    assert TEST_PASSWORD.encode() not in serialized
    assert auth_user.password_hash.encode() not in serialized
    assert auth.hash_session_token(response.session_token).encode() not in serialized


@pytest.mark.parametrize(
    ("username", "password"),
    [("", TEST_PASSWORD), ("operator", ""), ("", ""), ("   ", TEST_PASSWORD)],
)
def test_login_rejects_empty_input_with_invalid_argument(
    service: ClientService, auth_user: User, username: str, password: str
) -> None:
    context = FakeContext()
    with pytest.raises(_Aborted) as excinfo:
        service.login(
            client_pb2.LoginRequest(username=username, password=password), context
        )
    assert excinfo.value.code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    ("username", "password"),
    [("ghost", TEST_PASSWORD), ("operator", "wrong-password")],
)
def test_login_failures_are_unauthenticated_and_generic(
    service: ClientService, auth_user: User, username: str, password: str
) -> None:
    context = FakeContext()
    with pytest.raises(_Aborted) as excinfo:
        service.login(
            client_pb2.LoginRequest(username=username, password=password), context
        )
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED
    assert excinfo.value.details == auth.GENERIC_AUTH_FAILURE_MESSAGE


def test_login_rejects_inactive_user(
    service: ClientService,
    auth_user: User,
    session_factory: sessionmaker[DbSession],
) -> None:
    with session_factory() as db:
        db.get(User, auth_user.id).is_active = False
        db.commit()

    with pytest.raises(_Aborted) as excinfo:
        service.login(
            client_pb2.LoginRequest(
                username=auth_user.username, password=TEST_PASSWORD
            ),
            FakeContext(),
        )
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED
    assert excinfo.value.details == auth.GENERIC_AUTH_FAILURE_MESSAGE


def test_overlong_password_does_not_crash_the_rpc(
    service: ClientService, auth_user: User
) -> None:
    """bcrypt raises past 72 bytes; that must surface as UNAUTHENTICATED."""
    with pytest.raises(_Aborted) as excinfo:
        service.login(
            client_pb2.LoginRequest(username=auth_user.username, password="a" * 500),
            FakeContext(),
        )
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


# --- logout ----------------------------------------------------------------
def test_logout_succeeds_then_token_stops_working(
    service: ClientService, auth_user: User
) -> None:
    token = _login(service, auth_user.username, TEST_PASSWORD)

    response = service.logout(
        client_pb2.LogoutRequest(session_token=token), FakeContext()
    )
    assert response.success is True

    with pytest.raises(_Aborted) as excinfo:
        service.logout(client_pb2.LogoutRequest(session_token=token), FakeContext())
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize("token", ["", "bogus-token"])
def test_logout_rejects_bad_tokens(
    service: ClientService, auth_user: User, token: str
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        service.logout(client_pb2.LogoutRequest(session_token=token), FakeContext())
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


# --- Phase F stubs ---------------------------------------------------------
def _call_stub(service: ClientService, rpc: str, token: str):
    request = {
        "post": client_pb2.PostRequest(session_token=token, resource="order"),
        "get": client_pb2.GetRequest(session_token=token, resource="inventory"),
        "processBusinessRequest": client_pb2.BusinessRequest(
            session_token=token, operation="place_order", request_id="r-1"
        ),
    }[rpc]
    return getattr(service, rpc)(request, FakeContext())


@pytest.mark.parametrize("rpc", ["post", "get", "processBusinessRequest"])
def test_stub_rpcs_are_unimplemented_when_authenticated(
    service: ClientService, auth_user: User, rpc: str
) -> None:
    token = _login(service, auth_user.username, TEST_PASSWORD)
    with pytest.raises(_Aborted) as excinfo:
        _call_stub(service, rpc, token)
    assert excinfo.value.code == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.parametrize("rpc", ["post", "get", "processBusinessRequest"])
@pytest.mark.parametrize("token", ["", "bogus-token"])
def test_stub_rpcs_are_unauthenticated_without_a_valid_token(
    service: ClientService, auth_user: User, rpc: str, token: str
) -> None:
    """Auth is checked before the not-implemented reply.

    Otherwise UNIMPLEMENTED would leak that a token was valid.
    """
    with pytest.raises(_Aborted) as excinfo:
        _call_stub(service, rpc, token)
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize("rpc", ["post", "get", "processBusinessRequest"])
def test_revoked_token_cannot_reach_stub_rpcs(
    service: ClientService, auth_user: User, rpc: str
) -> None:
    token = _login(service, auth_user.username, TEST_PASSWORD)
    service.logout(client_pb2.LogoutRequest(session_token=token), FakeContext())

    with pytest.raises(_Aborted) as excinfo:
        _call_stub(service, rpc, token)
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED
