"""End-to-end tests over a real gRPC channel.

These exercise the full path a Streamlit client will take: serialized
protobuf over a socket, through the servicer, into the Phase D database and
back. They are the proof that the wire contract and the storage layer agree.
"""

from __future__ import annotations

from datetime import timedelta

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import auth
from app.storage.models import User, utc_now
from app.storage.repositories import SessionRepository
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub


def _login(stub: Stub, username: str, password: str) -> client_pb2.LoginResponse:
    return stub.login(client_pb2.LoginRequest(username=username, password=password))


def test_full_login_use_logout_cycle(stub: Stub, auth_user: User) -> None:
    """The complete session lifecycle a client will perform."""
    login = _login(stub, auth_user.username, TEST_PASSWORD)
    assert login.success is True
    token = login.session_token
    assert token

    # A valid token reaches a real handler (step 23 routes "products").
    assert (
        stub.get(
            client_pb2.GetRequest(session_token=token, resource="products")
        ).success
        is True
    )

    assert stub.logout(client_pb2.LogoutRequest(session_token=token)).success is True

    # After logout the same token must stop working immediately.
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.get(client_pb2.GetRequest(session_token=token, resource="products"))
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_login_over_the_wire_returns_user_and_expiry(
    stub: Stub, auth_user: User
) -> None:
    login = _login(stub, auth_user.username, TEST_PASSWORD)
    assert login.user.id == auth_user.id
    assert login.user.username == auth_user.username
    assert login.user.email == auth_user.email
    assert login.user.role == "manager"
    assert login.HasField("expires_at")
    assert login.expires_at.ToDatetime().year >= 2024


def test_login_persists_only_the_token_hash(
    stub: Stub, auth_user: User, session_factory: sessionmaker[DbSession]
) -> None:
    login = _login(stub, auth_user.username, TEST_PASSWORD)
    with session_factory() as db:
        records = SessionRepository(db).list()
        assert len(records) == 1
        assert records[0].token_hash == auth.hash_session_token(login.session_token)
        assert records[0].token_hash != login.session_token


@pytest.mark.parametrize(
    ("username", "password"),
    [("", TEST_PASSWORD), ("operator", "")],
)
def test_empty_credentials_are_invalid_argument(
    stub: Stub, auth_user: User, username: str, password: str
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        _login(stub, username, password)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize(
    ("username", "password"),
    [("ghost", TEST_PASSWORD), ("operator", "wrong")],
)
def test_bad_credentials_are_unauthenticated(
    stub: Stub, auth_user: User, username: str, password: str
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        _login(stub, username, password)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert excinfo.value.details() == auth.GENERIC_AUTH_FAILURE_MESSAGE


def test_error_details_never_leak_internals(stub: Stub, auth_user: User) -> None:
    """A client must not learn schema, SQL or hash details from an error."""
    with pytest.raises(grpc.RpcError) as excinfo:
        _login(stub, "ghost", "whatever")
    details = excinfo.value.details().lower()
    for leak in ("select", "sqlite", "traceback", "bcrypt", "$2b$", "users.", "hash"):
        assert leak not in details


@pytest.mark.parametrize("rpc_name", ["post", "get", "processBusinessRequest"])
def test_stub_rpcs_require_authentication(
    stub: Stub, auth_user: User, rpc_name: str
) -> None:
    requests = {
        "post": client_pb2.PostRequest(session_token="bogus", resource="order"),
        "get": client_pb2.GetRequest(session_token="bogus", resource="inventory"),
        "processBusinessRequest": client_pb2.BusinessRequest(
            session_token="bogus", operation="place_order", request_id="r-1"
        ),
    }
    with pytest.raises(grpc.RpcError) as excinfo:
        getattr(stub, rpc_name)(requests[rpc_name])
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_process_business_request_rejects_an_unknown_operation(
    stub: Stub, auth_user: User
) -> None:
    """Phase G routed this RPC, so an unrecognised operation is bad input.

    It reported UNIMPLEMENTED while it was a stub. Now it dispatches from its
    own table of AI operations, and `place_order` is not in it -- the same
    failure `post` and `get` give for an unknown resource.
    """
    token = _login(stub, auth_user.username, TEST_PASSWORD).session_token
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.processBusinessRequest(
            client_pb2.BusinessRequest(
                session_token=token, operation="place_order", request_id="r-1"
            )
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Supported values are" in excinfo.value.details()


@pytest.mark.parametrize("rpc_name", ["post", "get"])
def test_routed_rpcs_reject_an_unknown_resource(
    stub: Stub, auth_user: User, rpc_name: str
) -> None:
    token = _login(stub, auth_user.username, TEST_PASSWORD).session_token
    requests = {
        "post": client_pb2.PostRequest(session_token=token, resource="no.such.action"),
        "get": client_pb2.GetRequest(session_token=token, resource="no.such.action"),
    }
    with pytest.raises(grpc.RpcError) as excinfo:
        getattr(stub, rpc_name)(requests[rpc_name])
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_two_logins_issue_independent_sessions(stub: Stub, auth_user: User) -> None:
    """Logging out one device leaves the other signed in."""
    first = _login(stub, auth_user.username, TEST_PASSWORD).session_token
    second = _login(stub, auth_user.username, TEST_PASSWORD).session_token
    assert first != second

    stub.logout(client_pb2.LogoutRequest(session_token=first))

    with pytest.raises(grpc.RpcError) as excinfo:
        stub.get(client_pb2.GetRequest(session_token=first, resource="products"))
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED

    # The second session is untouched and still serves real reads.
    assert (
        stub.get(
            client_pb2.GetRequest(session_token=second, resource="products")
        ).success
        is True
    )


def test_expired_session_is_rejected_over_grpc(
    stub: Stub, auth_user: User, session_factory: sessionmaker[DbSession]
) -> None:
    """Expiry is enforced on the server, not merely in the domain layer.

    The TTL is backdated directly in the database rather than by waiting,
    since the shortest configurable TTL is a minute.
    """
    token = _login(stub, auth_user.username, TEST_PASSWORD).session_token

    # The token works right now.
    assert (
        stub.get(
            client_pb2.GetRequest(session_token=token, resource="products")
        ).success
        is True
    )

    with session_factory() as db:
        record = SessionRepository(db).get_by_token_hash(auth.hash_session_token(token))
        record.expires_at = utc_now() - timedelta(minutes=1)
        db.commit()

    with pytest.raises(grpc.RpcError) as excinfo:
        stub.get(client_pb2.GetRequest(session_token=token, resource="products"))
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED

    # An expired session is also not a valid thing to log out of.
    with pytest.raises(grpc.RpcError) as excinfo:
        stub.logout(client_pb2.LogoutRequest(session_token=token))
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_unknown_user_and_wrong_password_are_indistinguishable(
    stub: Stub, auth_user: User
) -> None:
    """Login must not leak which usernames exist."""
    outcomes = []
    for username, password in [("no-such-user", "x"), (auth_user.username, "wrong")]:
        with pytest.raises(grpc.RpcError) as excinfo:
            _login(stub, username, password)
        outcomes.append((excinfo.value.code(), excinfo.value.details()))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == grpc.StatusCode.UNAUTHENTICATED
    assert auth_user.username not in outcomes[0][1]


def test_no_response_over_the_wire_contains_credentials(
    stub: Stub, auth_user: User
) -> None:
    """Sweep every Phase E response for password, hash and token-hash bytes."""
    login = _login(stub, auth_user.username, TEST_PASSWORD)
    token = login.session_token
    token_hash = auth.hash_session_token(token)

    responses = [login, stub.logout(client_pb2.LogoutRequest(session_token=token))]
    secrets = [
        TEST_PASSWORD.encode(),
        auth_user.password_hash.encode(),
        token_hash.encode(),
        b"$2b$",
    ]
    for response in responses:
        payload = response.SerializeToString()
        for secret in secrets:
            assert secret not in payload, f"{type(response).__name__} leaked a secret"


def test_concurrent_requests_are_served(stub: Stub, auth_user: User) -> None:
    """The threaded server handles overlapping RPCs without pool exhaustion."""
    from concurrent.futures import ThreadPoolExecutor

    def attempt(_: int) -> grpc.StatusCode:
        try:
            _login(stub, auth_user.username, "wrong-password")
        except grpc.RpcError as exc:
            return exc.code()
        return grpc.StatusCode.OK

    with ThreadPoolExecutor(max_workers=6) as pool:
        codes = list(pool.map(attempt, range(6)))
    assert all(code == grpc.StatusCode.UNAUTHENTICATED for code in codes)
