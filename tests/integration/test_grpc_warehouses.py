"""Warehouses over a real gRPC channel (Phase F step 24)."""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.storage.models import User
from app.storage.repositories import WarehouseRepository
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub

WAREHOUSE = {"code": "WH-001", "name": "North Depot", "location": "Bengaluru, IN"}


@pytest.fixture
def token(stub: Stub, auth_user: User) -> str:
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


def test_create_then_read_a_warehouse(stub: Stub, token: str) -> None:
    assert post(stub, token, "warehouse.create", WAREHOUSE).success is True

    found = get(stub, token, "warehouse", {"code": "WH-001"})
    assert found.success is True
    assert len(found.items) == 1

    data = found.items[0].data
    assert data["code"] == "WH-001"
    assert data["name"] == "North Depot"
    assert data["location"] == "Bengaluru, IN"
    assert data["is_active"] is True


def test_get_warehouse_by_id(stub: Stub, token: str) -> None:
    post(stub, token, "warehouse.create", WAREHOUSE)
    found = get(stub, token, "warehouse", {"id": 1})
    assert found.items[0].data["code"] == "WH-001"


def test_list_warehouses_over_grpc_is_ordered(stub: Stub, token: str) -> None:
    for code in ["WH-C", "WH-A", "WH-B"]:
        post(stub, token, "warehouse.create", {**WAREHOUSE, "code": code})

    listed = get(stub, token, "warehouses", {})
    assert listed.success is True
    assert [item.data["code"] for item in listed.items] == ["WH-C", "WH-A", "WH-B"]
    assert [item.id for item in listed.items] == ["1", "2", "3"]


def test_list_supports_active_only(stub: Stub, token: str) -> None:
    post(stub, token, "warehouse.create", {**WAREHOUSE, "code": "WH-ON"})
    post(
        stub,
        token,
        "warehouse.create",
        {**WAREHOUSE, "code": "WH-OFF", "is_active": False},
    )

    listed = get(stub, token, "warehouses", {"active_only": True})
    assert [item.data["code"] for item in listed.items] == ["WH-ON"]


def test_duplicate_code_is_already_exists(stub: Stub, token: str) -> None:
    post(stub, token, "warehouse.create", WAREHOUSE)
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "warehouse.create", WAREHOUSE)
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_missing_warehouse_is_not_found(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, token, "warehouse", {"code": "NO-SUCH-CODE"})
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "No code", "location": "L"},
        {"code": "WH-X", "name": "No location"},
        {"code": "WH-X", "name": "N", "location": "L", "id": 7},
    ],
)
def test_invalid_create_is_invalid_argument(
    stub: Stub, token: str, payload: dict[str, Any]
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "warehouse.create", payload)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_failed_create_is_not_persisted(
    stub: Stub, token: str, session_factory: sessionmaker[DbSession]
) -> None:
    with pytest.raises(grpc.RpcError):
        post(stub, token, "warehouse.create", {"code": "WH-BAD", "name": "X"})

    with session_factory() as db:
        assert WarehouseRepository(db).list() == []


@pytest.mark.parametrize("resource", ["warehouse.create", "warehouse", "warehouses"])
def test_warehouse_actions_require_a_valid_session(stub: Stub, resource: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        if resource.endswith(".create"):
            post(stub, "bogus-token", resource, WAREHOUSE)
        else:
            get(stub, "bogus-token", resource, {})
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_revoked_session_cannot_create_warehouses(stub: Stub, token: str) -> None:
    stub.logout(client_pb2.LogoutRequest(session_token=token))
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "warehouse.create", WAREHOUSE)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_error_details_do_not_leak_internals(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, token, "warehouse", {"code": "GHOST"})
    details = excinfo.value.details().lower()
    for leak in ("select", "sqlite", "traceback", "sqlalchemy"):
        assert leak not in details


def test_products_and_warehouses_are_independent(stub: Stub, token: str) -> None:
    """Step 23 routes must keep working now that step 24 shares the service."""
    post(stub, token, "warehouse.create", WAREHOUSE)
    post(
        stub,
        token,
        "product.create",
        {"sku": "SKU-1", "name": "Widget", "unit_price": 9.99},
    )

    warehouses = get(stub, token, "warehouses", {})
    products = get(stub, token, "products", {})
    assert [i.data["code"] for i in warehouses.items] == ["WH-001"]
    assert [i.data["sku"] for i in products.items] == ["SKU-1"]
    # Separate tables, so both start their own id sequence at 1.
    assert warehouses.items[0].id == "1"
    assert products.items[0].id == "1"
