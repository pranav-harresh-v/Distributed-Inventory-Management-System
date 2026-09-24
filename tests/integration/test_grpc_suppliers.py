"""Suppliers over a real gRPC channel (Phase F step 26)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.storage.models import User
from app.storage.repositories import SupplierProductRepository, SupplierRepository
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub

SUPPLIER = {
    "name": "Nimbus Components",
    "email": "sales@nimbus.example.com",
    "phone": "+91-80-5550-0101",
}
TERMS = {"unit_cost": 14.20, "lead_time_days": 5}


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
    for sku in ["SKU-1", "SKU-2"]:
        post(
            stub,
            token,
            "product.create",
            {"sku": sku, "name": f"Item {sku}", "unit_price": 10.0},
        )


def test_create_supplier_then_list(stub: Stub, token: str) -> None:
    assert post(stub, token, "supplier.create", SUPPLIER).success is True

    listed = get(stub, token, "suppliers", {})
    assert listed.success is True
    assert len(listed.items) == 1

    data = listed.items[0].data
    assert data["name"] == "Nimbus Components"
    assert data["email"] == "sales@nimbus.example.com"
    assert data["is_active"] is True


def test_list_ordering_is_deterministic(stub: Stub, token: str) -> None:
    for name in ["Zeta", "Alpha", "Mu"]:
        post(stub, token, "supplier.create", {"name": name})

    listed = get(stub, token, "suppliers", {})
    assert [i.data["name"] for i in listed.items] == ["Zeta", "Alpha", "Mu"]
    assert [i.id for i in listed.items] == ["1", "2", "3"]


def test_list_supports_active_only(stub: Stub, token: str) -> None:
    post(stub, token, "supplier.create", {"name": "Active Co"})
    post(stub, token, "supplier.create", {"name": "Dormant Co", "is_active": False})

    listed = get(stub, token, "suppliers", {"active_only": True})
    assert [i.data["name"] for i in listed.items] == ["Active Co"]


def test_associate_stores_price_and_lead_time(
    stub: Stub,
    token: str,
    catalogue: None,
    session_factory: sessionmaker[DbSession],
) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    response = post(
        stub,
        token,
        "supplier.associate_product",
        {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS},
    )
    assert response.success is True

    # No read action for links in this step, so verify through storage.
    with session_factory() as db:
        link = SupplierProductRepository(db).get_by_supplier_and_product(1, 1)
        assert link is not None
        assert link.unit_cost == Decimal("14.20")
        assert link.lead_time_days == 5
        assert link.minimum_order_qty == 1  # schema default
        assert link.is_preferred is False  # schema default


def test_associate_with_full_terms(
    stub: Stub,
    token: str,
    catalogue: None,
    session_factory: sessionmaker[DbSession],
) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    post(
        stub,
        token,
        "supplier.associate_product",
        {
            "supplier_name": "Nimbus Components",
            "sku": "SKU-1",
            "unit_cost": 182.40,
            "lead_time_days": 14,
            "supplier_sku": "NC-MON-03",
            "minimum_order_qty": 5,
            "is_preferred": True,
        },
    )

    with session_factory() as db:
        link = SupplierProductRepository(db).get_by_supplier_and_product(1, 1)
    assert link.unit_cost == Decimal("182.40")
    assert link.lead_time_days == 14
    assert link.supplier_sku == "NC-MON-03"
    assert link.minimum_order_qty == 5
    assert link.is_preferred is True


def test_duplicate_supplier_is_already_exists(stub: Stub, token: str) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "supplier.create", SUPPLIER)
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS


def test_duplicate_association_is_already_exists(
    stub: Stub, token: str, catalogue: None
) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    payload = {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS}
    post(stub, token, "supplier.associate_product", payload)

    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "supplier.associate_product", payload)
    assert excinfo.value.code() == grpc.StatusCode.ALREADY_EXISTS


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"supplier_name": "GHOST", "sku": "SKU-1"}, "missing supplier"),
        ({"supplier_name": "Nimbus Components", "sku": "GHOST"}, "missing product"),
    ],
)
def test_missing_side_is_not_found(
    stub: Stub, token: str, catalogue: None, params: dict[str, Any], reason: str
) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "supplier.associate_product", {**params, **TERMS})
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize(
    "overrides",
    [
        {"unit_cost": -1},
        {"lead_time_days": -1},
        {"lead_time_days": 2.5},
        {"minimum_order_qty": 0},
        {"unit_cost": "free"},
    ],
)
def test_invalid_terms_are_invalid_argument(
    stub: Stub, token: str, catalogue: None, overrides: dict[str, Any]
) -> None:
    post(stub, token, "supplier.create", SUPPLIER)
    with pytest.raises(grpc.RpcError) as excinfo:
        post(
            stub,
            token,
            "supplier.associate_product",
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                **TERMS,
                **overrides,
            },
        )
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.parametrize("payload", [{}, {"name": "  "}, {"name": "X", "rating": 5}])
def test_invalid_supplier_is_invalid_argument(
    stub: Stub, token: str, payload: dict[str, Any]
) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "supplier.create", payload)
    assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_failed_write_is_not_persisted(
    stub: Stub,
    token: str,
    catalogue: None,
    session_factory: sessionmaker[DbSession],
) -> None:
    with pytest.raises(grpc.RpcError):
        post(stub, token, "supplier.create", {"name": ""})

    with session_factory() as db:
        assert SupplierRepository(db).list() == []


def test_error_details_never_expose_internals(stub: Stub, token: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        post(stub, token, "supplier.create", {})
    details = excinfo.value.details().lower()
    for leak in ("sqlalchemy", "traceback", "select", "sqlite", "integrityerror"):
        assert leak not in details


@pytest.mark.parametrize(
    "resource", ["supplier.create", "supplier.associate_product", "suppliers"]
)
def test_supplier_actions_require_a_valid_session(stub: Stub, resource: str) -> None:
    with pytest.raises(grpc.RpcError) as excinfo:
        if resource == "suppliers":
            get(stub, "bogus-token", resource, {})
        else:
            post(stub, "bogus-token", resource, SUPPLIER)
    assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_all_phase_f_steps_coexist(stub: Stub, token: str, catalogue: None) -> None:
    """Steps 23-26 all reachable through one session."""
    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-1", "name": "North", "location": "IN"},
    )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-1", "quantity": 8, "reorder_point": 10},
    )
    post(stub, token, "supplier.create", SUPPLIER)
    post(
        stub,
        token,
        "supplier.associate_product",
        {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS},
    )

    assert len(get(stub, token, "products", {}).items) == 2
    assert len(get(stub, token, "warehouses", {}).items) == 1
    assert len(get(stub, token, "suppliers", {}).items) == 1
    low = get(stub, token, "inventory.low_stock", {})
    assert [i.data["sku"] for i in low.items] == ["SKU-1"]
