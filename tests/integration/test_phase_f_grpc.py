"""Phase F final integration pass -- every operation over real gRPC.

Each test drives the actual ClientService through a socket: protobuf Struct
in, session validation, domain logic, SQLite, protobuf out. Repositories are
used only to *verify* rows that have no read action of their own (orders,
order items, sales history); never to perform an operation.

Isolation: every test gets its own temporary SQLite file and its own server
on an OS-assigned port, both function-scoped, so nothing here depends on
execution order and nothing touches data/node1.db.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.storage.models import User
from app.storage.repositories import (
    OrderItemRepository,
    OrderRepository,
    SalesHistoryRepository,
)
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

Stub = client_pb2_grpc.ClientServiceStub


# --- helpers ---------------------------------------------------------------
def login(stub: Stub, user: User) -> str:
    response = stub.login(
        client_pb2.LoginRequest(username=user.username, password=TEST_PASSWORD)
    )
    assert response.success is True
    return response.session_token


def post(stub: Stub, token: str, resource: str, payload: dict[str, Any]):
    request = client_pb2.PostRequest(session_token=token, resource=resource)
    request.payload.update(payload)
    return stub.post(request)


def get(stub: Stub, token: str, resource: str, params: dict[str, Any] | None = None):
    request = client_pb2.GetRequest(session_token=token, resource=resource)
    request.params.update(params or {})
    return stub.get(request)


def code_of(call, *args) -> grpc.StatusCode:
    """Invoke an RPC and return its status code instead of raising."""
    try:
        call(*args)
    except grpc.RpcError as exc:
        return exc.code()
    return grpc.StatusCode.OK


@pytest.fixture
def token(stub: Stub, auth_user: User) -> str:
    return login(stub, auth_user)


@pytest.fixture
def seeded(stub: Stub, token: str) -> dict[str, Any]:
    """A small world built entirely through the public gRPC surface."""
    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-1", "name": "North Depot", "location": "Bengaluru"},
    )
    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-2", "name": "South Depot", "location": "Chennai"},
    )
    for sku, name, price in [
        ("SKU-1", "Wireless Mouse", 24.99),
        ("SKU-2", "Mechanical Keyboard", 79.50),
    ]:
        post(
            stub,
            token,
            "product.create",
            {"sku": sku, "name": name, "unit_price": price},
        )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-1", "quantity": 50, "reorder_point": 10},
    )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-2", "quantity": 6, "reorder_point": 2},
    )
    return {"warehouse": "WH-1", "skus": ["SKU-1", "SKU-2"]}


def stock_of(stub: Stub, token: str, sku: str, warehouse: str = "WH-1") -> int:
    response = get(
        stub, token, "inventory.stock", {"warehouse_code": warehouse, "sku": sku}
    )
    return int(response.items[0].data["quantity"])


# =========================================================================
# 1. products: create, get, list, update
# =========================================================================
def test_product_lifecycle_over_grpc(stub: Stub, token: str) -> None:
    created = post(
        stub,
        token,
        "product.create",
        {"sku": "SKU-100", "name": "Widget", "unit_price": 12.50},
    )
    assert created.success is True

    fetched = get(stub, token, "product", {"sku": "SKU-100"})
    assert fetched.items[0].data["name"] == "Widget"
    assert fetched.items[0].data["unit_price"] == "12.50"

    post(
        stub,
        token,
        "product.create",
        {"sku": "SKU-200", "name": "Gadget", "unit_price": 5.00},
    )
    listed = get(stub, token, "products")
    assert [i.data["sku"] for i in listed.items] == ["SKU-100", "SKU-200"]

    post(
        stub,
        token,
        "product.update",
        {"sku_key": "SKU-100", "name": "Widget Pro", "unit_price": 19.99},
    )
    updated = get(stub, token, "product", {"id": 1}).items[0].data
    assert updated["name"] == "Widget Pro"
    assert updated["unit_price"] == "19.99"
    assert updated["sku"] == "SKU-100"  # untouched field preserved


def test_product_error_mapping_over_grpc(stub: Stub, token: str) -> None:
    post(
        stub,
        token,
        "product.create",
        {"sku": "SKU-1", "name": "X", "unit_price": 1.0},
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "product.create",
            {"sku": "SKU-1", "name": "X", "unit_price": 1.0},
        )
        is grpc.StatusCode.ALREADY_EXISTS
    )
    assert (
        code_of(
            post, stub, token, "product.create", {"name": "no sku", "unit_price": 1.0}
        )
        is grpc.StatusCode.INVALID_ARGUMENT
    )
    assert (
        code_of(get, stub, token, "product", {"sku": "GHOST"})
        is grpc.StatusCode.NOT_FOUND
    )


# =========================================================================
# 2. invalid session rejected
# =========================================================================
@pytest.mark.parametrize(
    "resource",
    [
        "product.create",
        "product.update",
        "warehouse.create",
        "inventory.set_initial_stock",
        "inventory.add_stock",
        "supplier.create",
        "supplier.associate_product",
        "order.place",
    ],
)
def test_every_write_rejects_an_invalid_session(stub: Stub, resource: str) -> None:
    assert code_of(post, stub, "not-a-real-token", resource, {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )


@pytest.mark.parametrize(
    "resource",
    [
        "product",
        "products",
        "warehouse",
        "warehouses",
        "suppliers",
        "inventory.stock",
        "inventory.warehouse",
        "inventory.low_stock",
    ],
)
def test_every_read_rejects_an_invalid_session(stub: Stub, resource: str) -> None:
    assert code_of(get, stub, "not-a-real-token", resource, {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )


@pytest.mark.parametrize("bad_token", ["", "   ", "bogus", "a" * 200])
def test_malformed_tokens_are_rejected(stub: Stub, bad_token: str) -> None:
    assert code_of(get, stub, bad_token, "products", {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )


def test_authentication_is_checked_before_the_action_is_resolved(stub: Stub) -> None:
    """An unknown resource with a bad token must not leak that it is unknown."""
    assert code_of(post, stub, "bogus", "no.such.action", {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )


# =========================================================================
# 3. warehouses: create, get, list
# =========================================================================
def test_warehouse_lifecycle_over_grpc(stub: Stub, token: str) -> None:
    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-A", "name": "Alpha", "location": "Delhi"},
    )
    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-B", "name": "Beta", "location": "Mumbai"},
    )

    single = get(stub, token, "warehouse", {"code": "WH-A"})
    assert single.items[0].data["location"] == "Delhi"
    assert single.items[0].data["is_active"] is True

    by_id = get(stub, token, "warehouse", {"id": 2})
    assert by_id.items[0].data["code"] == "WH-B"

    listed = get(stub, token, "warehouses")
    assert [i.data["code"] for i in listed.items] == ["WH-A", "WH-B"]

    assert (
        code_of(
            post,
            stub,
            token,
            "warehouse.create",
            {"code": "WH-A", "name": "Dup", "location": "X"},
        )
        is grpc.StatusCode.ALREADY_EXISTS
    )
    assert (
        code_of(get, stub, token, "warehouse", {"code": "GHOST"})
        is grpc.StatusCode.NOT_FOUND
    )


# =========================================================================
# 4. inventory: initialize, add, read
# =========================================================================
def test_inventory_initialize_add_and_read(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    assert stock_of(stub, token, "SKU-1") == 50

    post(
        stub,
        token,
        "inventory.add_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-1", "amount": 25},
    )
    assert stock_of(stub, token, "SKU-1") == 75

    listed = get(stub, token, "inventory.warehouse", {"warehouse_code": "WH-1"})
    rows = {i.data["sku"]: i.data["quantity"] for i in listed.items}
    assert rows == {"SKU-1": 75, "SKU-2": 6}


def test_inventory_rejects_bad_values_over_grpc(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    key = {"warehouse_code": "WH-1", "sku": "SKU-1"}
    assert (
        code_of(post, stub, token, "inventory.add_stock", {**key, "amount": 0})
        is grpc.StatusCode.INVALID_ARGUMENT
    )
    assert (
        code_of(post, stub, token, "inventory.add_stock", {**key, "amount": -5})
        is grpc.StatusCode.INVALID_ARGUMENT
    )
    assert (
        code_of(
            post, stub, token, "inventory.set_initial_stock", {**key, "quantity": 99}
        )
        is grpc.StatusCode.ALREADY_EXISTS
    )
    assert (
        code_of(
            get,
            stub,
            token,
            "inventory.stock",
            {"warehouse_code": "WH-2", "sku": "SKU-1"},
        )
        is grpc.StatusCode.NOT_FOUND
    )

    assert stock_of(stub, token, "SKU-1") == 50  # nothing changed


# =========================================================================
# 5. low-stock detection
# =========================================================================
def test_low_stock_detection_over_grpc(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    # Nothing is low initially: 50 > 10 and 6 > 2.
    assert get(stub, token, "inventory.low_stock").items == []

    # Consume SKU-2 down to its reorder point of 2.
    post(
        stub,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-2", "quantity": 4}]},
    )

    low = get(stub, token, "inventory.low_stock")
    assert [i.data["sku"] for i in low.items] == ["SKU-2"]
    assert low.items[0].data["is_low_stock"] is True
    assert low.items[0].data["quantity"] == 2

    # Restocking clears the report.
    post(
        stub,
        token,
        "inventory.add_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-2", "amount": 10},
    )
    assert get(stub, token, "inventory.low_stock").items == []


# =========================================================================
# 6. suppliers: create, associate, persist terms, list
# =========================================================================
def test_supplier_creation_and_association_over_grpc(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    post(
        stub,
        token,
        "supplier.create",
        {"name": "Nimbus Components", "email": "sales@nimbus.example.com"},
    )
    post(stub, token, "supplier.create", {"name": "Vertex Peripherals"})

    listed = get(stub, token, "suppliers")
    assert [i.data["name"] for i in listed.items] == [
        "Nimbus Components",
        "Vertex Peripherals",
    ]

    response = post(
        stub,
        token,
        "supplier.associate_product",
        {
            "supplier_name": "Nimbus Components",
            "sku": "SKU-1",
            "unit_cost": 14.20,
            "lead_time_days": 5,
            "minimum_order_qty": 25,
            "is_preferred": True,
        },
    )
    assert response.success is True

    # Lead time and price persisted exactly (no read action for links yet).
    from app.storage.repositories import SupplierProductRepository

    with session_factory() as db:
        link = SupplierProductRepository(db).get_by_supplier_and_product(1, 1)
        assert link is not None
        assert link.unit_cost == Decimal("14.20")
        assert link.lead_time_days == 5
        assert link.minimum_order_qty == 25
        assert link.is_preferred is True


def test_supplier_error_mapping_over_grpc(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    post(stub, token, "supplier.create", {"name": "Nimbus"})
    assert (
        code_of(post, stub, token, "supplier.create", {"name": "Nimbus"})
        is grpc.StatusCode.ALREADY_EXISTS
    )

    terms = {"unit_cost": 1.0, "lead_time_days": 1}
    assert (
        code_of(
            post,
            stub,
            token,
            "supplier.associate_product",
            {"supplier_name": "GHOST", "sku": "SKU-1", **terms},
        )
        is grpc.StatusCode.NOT_FOUND
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "supplier.associate_product",
            {"supplier_name": "Nimbus", "sku": "GHOST", **terms},
        )
        is grpc.StatusCode.NOT_FOUND
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "supplier.associate_product",
            {
                "supplier_name": "Nimbus",
                "sku": "SKU-1",
                "unit_cost": -1,
                "lead_time_days": 1,
            },
        )
        is grpc.StatusCode.INVALID_ARGUMENT
    )


# =========================================================================
# 7-11. a successful order, end to end
# =========================================================================
def test_successful_single_item_order_over_grpc(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    response = post(
        stub,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 4}]},
    )
    assert response.success is True
    assert "ORD-" in response.message

    # 8. inventory decremented, observed through gRPC
    assert stock_of(stub, token, "SKU-1") == 46

    with session_factory() as db:
        orders_rows = OrderRepository(db).list()
        items = OrderItemRepository(db).list()
        sales = SalesHistoryRepository(db).list()

    # 9. order persisted
    assert len(orders_rows) == 1
    assert orders_rows[0].status == "COMPLETED"
    assert orders_rows[0].total_amount == Decimal("99.96")  # 4 * 24.99
    assert orders_rows[0].order_number in response.message

    # 10. order items persisted
    assert len(items) == 1
    assert items[0].quantity == 4
    assert items[0].unit_price == Decimal("24.99")

    # 11. sales history persisted
    assert len(sales) == 1
    assert sales[0].quantity == 4
    assert sales[0].warehouse_id == 1
    assert sales[0].sold_at is not None


def test_successful_multi_item_order_over_grpc(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    post(
        stub,
        token,
        "order.place",
        {
            "warehouse_code": "WH-1",
            "items": [
                {"sku": "SKU-1", "quantity": 2},
                {"sku": "SKU-2", "quantity": 3},
            ],
        },
    )

    assert stock_of(stub, token, "SKU-1") == 48
    assert stock_of(stub, token, "SKU-2") == 3

    with session_factory() as db:
        order = OrderRepository(db).list()[0]
        items = {i.product_id: i.quantity for i in OrderItemRepository(db).list()}
        sales = {s.product_id: s.quantity for s in SalesHistoryRepository(db).list()}

    # 2 * 24.99 + 3 * 79.50
    assert order.total_amount == Decimal("288.48")
    assert items == {1: 2, 2: 3}
    assert sales == {1: 2, 2: 3}


def test_duplicate_order_lines_consolidated_over_grpc(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    post(
        stub,
        token,
        "order.place",
        {
            "warehouse_code": "WH-1",
            "items": [
                {"sku": "SKU-1", "quantity": 3},
                {"sku": "SKU-1", "quantity": 2},
            ],
        },
    )

    assert stock_of(stub, token, "SKU-1") == 45
    with session_factory() as db:
        items = OrderItemRepository(db).list()
    assert len(items) == 1
    assert items[0].quantity == 5


# =========================================================================
# 12-14. an insufficient-stock order, rejected atomically
# =========================================================================
def test_insufficient_stock_order_returns_failed_precondition(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    """SKU-2 holds 6; ask for 7. Nothing at all may change."""
    before_1 = stock_of(stub, token, "SKU-1")
    before_2 = stock_of(stub, token, "SKU-2")
    assert before_2 == 6

    with pytest.raises(grpc.RpcError) as excinfo:
        post(
            stub,
            token,
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-2", "quantity": 7}]},
        )

    # 12. the status code and a message naming the product
    assert excinfo.value.code() is grpc.StatusCode.FAILED_PRECONDITION
    assert "SKU-2" in excinfo.value.details()

    # 13. stock unchanged
    assert stock_of(stub, token, "SKU-1") == before_1
    assert stock_of(stub, token, "SKU-2") == before_2

    # 14. nothing persisted
    with session_factory() as db:
        assert OrderRepository(db).list() == []
        assert OrderItemRepository(db).list() == []
        assert SalesHistoryRepository(db).list() == []


def test_multi_item_order_rolls_back_every_line(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    """The first line has stock and is decremented; the second does not."""
    before_1 = stock_of(stub, token, "SKU-1")
    before_2 = stock_of(stub, token, "SKU-2")

    with pytest.raises(grpc.RpcError) as excinfo:
        post(
            stub,
            token,
            "order.place",
            {
                "warehouse_code": "WH-1",
                "items": [
                    {"sku": "SKU-1", "quantity": 10},
                    {"sku": "SKU-2", "quantity": 500},
                ],
            },
        )
    assert excinfo.value.code() is grpc.StatusCode.FAILED_PRECONDITION

    assert stock_of(stub, token, "SKU-1") == before_1, "first line not rolled back"
    assert stock_of(stub, token, "SKU-2") == before_2

    with session_factory() as db:
        assert OrderRepository(db).list() == []
        assert OrderItemRepository(db).list() == []
        assert SalesHistoryRepository(db).list() == []


def test_order_error_mapping_over_grpc(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    warehouse = "WH-1"
    assert (
        code_of(
            post,
            stub,
            token,
            "order.place",
            {"warehouse_code": warehouse, "items": [{"sku": "SKU-1", "quantity": 0}]},
        )
        is grpc.StatusCode.INVALID_ARGUMENT
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "order.place",
            {"warehouse_code": "GHOST", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        is grpc.StatusCode.NOT_FOUND
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "order.place",
            {"warehouse_code": warehouse, "items": [{"sku": "GHOST", "quantity": 1}]},
        )
        is grpc.StatusCode.NOT_FOUND
    )
    assert (
        code_of(
            post,
            stub,
            token,
            "order.place",
            {"warehouse_code": "WH-2", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        is grpc.StatusCode.NOT_FOUND
    )


# =========================================================================
# 15. logout closes the whole Phase F surface
# =========================================================================
def test_logout_then_protected_calls_fail(
    stub: Stub, token: str, seeded: dict[str, Any]
) -> None:
    # The token works before logout.
    assert get(stub, token, "products").success is True

    assert stub.logout(client_pb2.LogoutRequest(session_token=token)).success is True

    for resource, params in [
        ("products", {}),
        ("warehouses", {}),
        ("suppliers", {}),
        ("inventory.low_stock", {}),
        ("inventory.stock", {"warehouse_code": "WH-1", "sku": "SKU-1"}),
    ]:
        assert code_of(get, stub, token, resource, params) is (
            grpc.StatusCode.UNAUTHENTICATED
        ), resource

    for resource, payload in [
        ("product.create", {"sku": "X", "name": "X", "unit_price": 1.0}),
        ("warehouse.create", {"code": "X", "name": "X", "location": "X"}),
        ("supplier.create", {"name": "X"}),
        (
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        ),
    ]:
        assert code_of(post, stub, token, resource, payload) is (
            grpc.StatusCode.UNAUTHENTICATED
        ), resource


def test_revoked_session_cannot_mutate_state(
    stub: Stub,
    token: str,
    seeded: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    """A logged-out token must not be able to change anything."""
    stub.logout(client_pb2.LogoutRequest(session_token=token))

    assert (
        code_of(
            post,
            stub,
            token,
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        is grpc.StatusCode.UNAUTHENTICATED
    )

    with session_factory() as db:
        assert OrderRepository(db).list() == []
        assert SalesHistoryRepository(db).list() == []


def test_a_second_session_is_unaffected_by_the_first_logging_out(
    stub: Stub, auth_user: User, seeded: dict[str, Any]
) -> None:
    first = login(stub, auth_user)
    second = login(stub, auth_user)

    stub.logout(client_pb2.LogoutRequest(session_token=first))

    assert code_of(get, stub, first, "products", {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )
    assert get(stub, second, "products").success is True


# =========================================================================
# cross-cutting: the whole Phase F workflow in one session
# =========================================================================
def test_complete_phase_f_workflow_end_to_end(
    stub: Stub,
    auth_user: User,
    session_factory: sessionmaker[DbSession],
) -> None:
    """Login, build a catalogue, stock it, supply it, sell it, log out."""
    token = login(stub, auth_user)

    post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-MAIN", "name": "Main", "location": "Bengaluru"},
    )
    post(
        stub,
        token,
        "product.create",
        {"sku": "SKU-X", "name": "Monitor", "unit_price": 249.00},
    )
    post(
        stub,
        token,
        "inventory.set_initial_stock",
        {
            "warehouse_code": "WH-MAIN",
            "sku": "SKU-X",
            "quantity": 12,
            "reorder_point": 5,
        },
    )
    post(stub, token, "supplier.create", {"name": "Vertex"})
    post(
        stub,
        token,
        "supplier.associate_product",
        {
            "supplier_name": "Vertex",
            "sku": "SKU-X",
            "unit_cost": 176.75,
            "lead_time_days": 10,
        },
    )

    # Sell enough to cross the reorder point.
    post(
        stub,
        token,
        "order.place",
        {"warehouse_code": "WH-MAIN", "items": [{"sku": "SKU-X", "quantity": 8}]},
    )

    assert stock_of(stub, token, "SKU-X", warehouse="WH-MAIN") == 4
    low = get(stub, token, "inventory.low_stock", {"warehouse_code": "WH-MAIN"})
    assert [i.data["sku"] for i in low.items] == ["SKU-X"]

    with session_factory() as db:
        order = OrderRepository(db).list()[0]
        assert order.total_amount == Decimal("1992.00")  # 8 * 249.00
        assert len(SalesHistoryRepository(db).list()) == 1

    assert stub.logout(client_pb2.LogoutRequest(session_token=token)).success is True
    assert code_of(get, stub, token, "products", {}) is (
        grpc.StatusCode.UNAUTHENTICATED
    )


def test_every_registered_action_is_reachable(stub: Stub, token: str) -> None:
    """No registered action is unroutable, and no unknown one is accepted.

    Guards against a handler being registered but never wired, which unit
    tests on the servicer alone would not catch.
    """
    from app.api.client_service import GET_ACTIONS, POST_ACTIONS

    for resource in POST_ACTIONS:
        # Empty payloads are invalid, but the action must resolve -- an
        # unroutable resource would also be INVALID_ARGUMENT, so the
        # message distinguishes them.
        try:
            post(stub, token, resource, {})
        except grpc.RpcError as exc:
            assert "Unknown resource" not in exc.details(), resource

    for resource in GET_ACTIONS:
        try:
            get(stub, token, resource, {})
        except grpc.RpcError as exc:
            assert "Unknown resource" not in exc.details(), resource

    assert code_of(post, stub, token, "not.a.real.action", {}) is (
        grpc.StatusCode.INVALID_ARGUMENT
    )


def test_each_test_starts_from_a_clean_database(stub: Stub, token: str) -> None:
    """Fixture isolation: no residue from any other test in this module."""
    assert get(stub, token, "products").items == []
    assert get(stub, token, "warehouses").items == []
    assert get(stub, token, "suppliers").items == []
    assert get(stub, token, "inventory.low_stock").items == []
