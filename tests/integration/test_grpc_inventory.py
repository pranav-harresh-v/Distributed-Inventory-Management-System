"""Stock levels over a real gRPC channel (Phase F step 25)."""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.storage.models import User
from app.storage.repositories import InventoryRepository
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub


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


@pytest.fixture
def catalogue(stub: Stub, token: str) -> None:
    """Warehouses and products created through the public API."""
    for code, name in [("WH-1", "North"), ("WH-2", "South")]:
        post(
            stub,
            token,
            "warehouse.create",
            {"code": code, "name": name, "location": "IN"},
        )
    for sku in ["SKU-1", "SKU-2"]:
        post(
            stub,
            token,
            "product.create",
            {"sku": sku, "name": f"Item {sku}", "unit_price": 10.0},
        )


KEY = {"warehouse_code": "WH-1", "sku": "SKU-1"}


def test_set_initial_stock_then_read(stub: Stub, token: str, catalogue: None) -> None:
    created = post(
        stub,
        token,
        "inventory.set_initial_stock",
        {**KEY, "quantity": 12, "reorder_point": 4, "safety_stock": 2},
    )
    assert created.success is True

    found = get(stub, token, "inventory.stock", KEY)
    data = found.items[0].data
    assert data["quantity"] == 12
    assert data["reorder_point"] == 4
    assert data["sku"] == "SKU-1"
    assert data["warehouse_code"] == "WH-1"
    assert data["is_low_stock"] is False


def test_add_stock_over_grpc(stub: Stub, token: str, catalogue: None) -> None:
    post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 10})
    post(stub, token, "inventory.add_stock", {**KEY, "amount": 5})
    post(stub, token, "inventory.add_stock", {**KEY, "amount": 5})

    assert get(stub, token, "inventory.stock", KEY).items[0].data["quantity"] == 20


def test_list_warehouse_inventory(stub: Stub, token: str, catalogue: None) -> None:
    post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 1})
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-2", "quantity": 2},
    )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-2", "sku": "SKU-1", "quantity": 99},
    )

    listed = get(stub, token, "inventory.warehouse", {"warehouse_code": "WH-1"})
    assert [i.data["sku"] for i in listed.items] == ["SKU-1", "SKU-2"]
    assert [i.data["quantity"] for i in listed.items] == [1, 2]


def test_low_stock_over_grpc(stub: Stub, token: str, catalogue: None) -> None:
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {**KEY, "quantity": 3, "reorder_point": 5},
    )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-2", "quantity": 50, "reorder_point": 5},
    )

    low = get(stub, token, "inventory.low_stock", {})
    assert [i.data["sku"] for i in low.items] == ["SKU-1"]
    assert low.items[0].data["is_low_stock"] is True

    # Restocking removes it from the report.
    post(stub, token, "inventory.add_stock", {**KEY, "amount": 10})
    assert get(stub, token, "inventory.low_stock", {}).items == []


@pytest.mark.parametrize("quantity", [-1, -50])
def test_negative_initial_quantity_is_invalid_argument(
    stub: Stub, token: str, catalogue: None, quantity: int
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(
            stub,
            token,
            "inventory.set_initial_stock",
            {**KEY, "quantity": quantity},
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize("amount", [0, -5])
def test_non_positive_add_is_invalid_argument(
    stub: Stub, token: str, catalogue: None, amount: int
) -> None:
    post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 10})
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "inventory.add_stock", {**KEY, "amount": amount})
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    assert get(stub, token, "inventory.stock", KEY).items[0].data["quantity"] == 10


def test_duplicate_initial_stock_is_already_exists(
    stub: Stub, token: str, catalogue: None
) -> None:
    post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 10})
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 999})
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS

    # The original quantity survives the rejected overwrite.
    assert get(stub, token, "inventory.stock", KEY).items[0].data["quantity"] == 10


def test_missing_stock_is_not_found(stub: Stub, token: str, catalogue: None) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        get(stub, token, "inventory.stock", KEY)
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_unknown_warehouse_or_product_is_not_found(
    stub: Stub, token: str, catalogue: None
) -> None:
    for params in [
        {"warehouse_code": "GHOST", "sku": "SKU-1"},
        {"warehouse_code": "WH-1", "sku": "GHOST"},
    ]:
        with pytest.raises(grpc.RpcError) as excinfo:
            get(stub, token, "inventory.stock", params)
        assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_add_to_missing_record_is_not_found(
    stub: Stub, token: str, catalogue: None
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "inventory.add_stock", {**KEY, "amount": 5})
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_failed_write_is_not_persisted(
    stub: Stub,
    token: str,
    catalogue: None,
    session_factory: sessionmaker[DbSession],
) -> None:
    with pytest.raises(grpc.RpcError):
        post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": -1})

    with session_factory() as db:
        assert InventoryRepository(db).list() == []


@pytest.mark.parametrize(
    "resource",
    [
        "inventory.set_initial_stock",
        "inventory.stock",
        "inventory.warehouse",
        "inventory.low_stock",
    ],
)
def test_inventory_actions_require_a_valid_session(stub: Stub, resource: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        if resource == "inventory.set_initial_stock":
            post(stub, "bogus-token", resource, {**KEY, "quantity": 1})
        else:
            get(stub, "bogus-token", resource, KEY)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_concurrent_adds_do_not_lose_updates(
    stub: Stub, token: str, catalogue: None
) -> None:
    """Overlapping increments must all land.

    This is what the database-side ``quantity = quantity + :amount`` buys:
    a read-modify-write would let two callers read the same value and one
    increment would vanish.
    """
    from concurrent.futures import ThreadPoolExecutor

    post(stub, token, "inventory.set_initial_stock", {**KEY, "quantity": 0})

    def add_one(_: int) -> None:
        post(stub, token, "inventory.add_stock", {**KEY, "amount": 1})

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(add_one, range(10)))

    assert get(stub, token, "inventory.stock", KEY).items[0].data["quantity"] == 10


def test_earlier_phase_f_steps_still_work(
    stub: Stub, token: str, catalogue: None
) -> None:
    assert len(get(stub, token, "products", {}).items) == 2
    assert len(get(stub, token, "warehouses", {}).items) == 2
