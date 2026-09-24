"""Product catalogue over a real gRPC channel (Phase F step 23).

Proves the full path a client takes: protobuf Struct over a socket, through
session validation, into the catalogue domain, into SQLite, and back as
Struct-safe values.
"""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import auth
from app.storage.database import session_scope
from app.storage.models import User
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub

PRODUCT = {"sku": "SKU-500", "name": "27in Monitor", "unit_price": 249.00}


@pytest.fixture
def token(stub: Stub, auth_user: User) -> str:
    """A live session token obtained through the real login RPC."""
    response = stub.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD)
    )
    return response.session_token


def post(stub: Stub, token: str, resource: str, payload: dict[str, Any]):
    request = client_pb2.PostRequest(session_token=token, resource=resource)
    request.payload.update(payload)
    return stub.post(request)


def get(stub: Stub, token: str, resource: str, params: dict[str, Any]):
    request = client_pb2.GetRequest(session_token=token, resource=resource)
    request.params.update(params)
    return stub.get(request)


def test_create_then_read_a_product(stub: Stub, token: str) -> None:
    created = post(stub, token, "product.create", PRODUCT)
    assert created.success is True

    found = get(stub, token, "product", {"sku": "SKU-500"})
    assert found.success is True
    assert len(found.items) == 1

    data = found.items[0].data
    assert data["sku"] == "SKU-500"
    assert data["name"] == "27in Monitor"
    # Money crosses the wire as an exact decimal string, never a double.
    assert data["unit_price"] == "249.00"
    assert data["is_active"] is True


def test_full_catalogue_lifecycle(stub: Stub, token: str) -> None:
    post(stub, token, "product.create", PRODUCT)

    updated = post(
        stub,
        token,
        "product.update",
        {"sku_key": "SKU-500", "name": "27in 1440p Monitor", "unit_price": 279.5},
    )
    assert updated.success is True

    data = get(stub, token, "product", {"sku": "SKU-500"}).items[0].data
    assert data["name"] == "27in 1440p Monitor"
    assert data["unit_price"] == "279.50"
    assert data["sku"] == "SKU-500"  # untouched field preserved


def test_list_products_over_grpc_is_ordered(stub: Stub, token: str) -> None:
    for sku in ["SKU-C", "SKU-A", "SKU-B"]:
        post(stub, token, "product.create", {**PRODUCT, "sku": sku, "name": sku})

    listed = get(stub, token, "products", {})
    assert listed.success is True
    assert [item.data["sku"] for item in listed.items] == ["SKU-C", "SKU-A", "SKU-B"]
    assert [item.id for item in listed.items] == ["1", "2", "3"]


def test_list_supports_active_only(stub: Stub, token: str) -> None:
    post(stub, token, "product.create", {**PRODUCT, "sku": "SKU-ON"})
    post(
        stub,
        token,
        "product.create",
        {**PRODUCT, "sku": "SKU-OFF", "is_active": False},
    )

    listed = get(stub, token, "products", {"active_only": True})
    assert [item.data["sku"] for item in listed.items] == ["SKU-ON"]


def test_duplicate_sku_is_already_exists(stub: Stub, token: str) -> None:
    post(stub, token, "product.create", PRODUCT)
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "product.create", PRODUCT)
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_missing_product_is_not_found(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, token, "product", {"sku": "NO-SUCH-SKU"})
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_updating_a_missing_product_is_not_found(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "product.update", {"sku_key": "GHOST", "name": "X"})
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "No sku", "unit_price": 1.0},
        {"sku": "SKU-X", "name": "Negative", "unit_price": -1.0},
        {"sku": "SKU-X", "name": "Bad field", "unit_price": 1.0, "id": 7},
    ],
)
def test_invalid_create_is_invalid_argument(
    stub: Stub, token: str, payload: dict[str, Any]
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "product.create", payload)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_unknown_resource_is_invalid_argument(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "widget.create", {})
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "product.create" in excinfo.value.details()


@pytest.mark.parametrize("resource", ["product.create", "product.update"])
def test_writes_require_a_valid_session(stub: Stub, resource: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, "bogus-token", resource, PRODUCT)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


@pytest.mark.parametrize("resource", ["product", "products"])
def test_reads_require_a_valid_session(stub: Stub, resource: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, "bogus-token", resource, {})
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_revoked_session_cannot_write(stub: Stub, token: str) -> None:
    """Logging out immediately closes the catalogue to that token."""
    stub.logout(client_pb2.LogoutRequest(session_token=token))
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "product.create", PRODUCT)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_failed_create_is_not_persisted(
    stub: Stub, token: str, session_factory: sessionmaker[DbSession]
) -> None:
    """A rejected write must leave the database untouched."""
    with pytest.raises(grpc.RpcError):
        post(stub, token, "product.create", {"sku": "SKU-BAD", "name": "X"})

    from app.storage.repositories import ProductRepository

    with session_factory() as db:
        assert ProductRepository(db).list() == []


def test_error_details_do_not_leak_internals(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, token, "product", {"sku": "GHOST"})
    details = excinfo.value.details().lower()
    for leak in ("select", "sqlite", "traceback", "sqlalchemy"):
        assert leak not in details


def test_product_rpcs_do_not_disturb_authentication(
    stub: Stub, token: str, session_factory: sessionmaker[DbSession]
) -> None:
    """Catalogue traffic must not consume or invalidate the session."""
    post(stub, token, "product.create", PRODUCT)
    get(stub, token, "products", {})

    with session_scope(session_factory) as db:
        assert auth.authenticate(db, token).session.revoked is False
