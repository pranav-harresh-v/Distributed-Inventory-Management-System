"""Unit tests for stock levels (Phase F step 25).

The invariant under test throughout is ``quantity >= 0``. It is checked at
three layers -- domain validation, the increment-only write path, and the
schema's own CheckConstraint -- and the last of those is exercised directly
so the guarantee does not rest on Python alone.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth, inventory
from app.storage.database import session_scope
from app.storage.models import Inventory, Product, User, Warehouse
from app.storage.repositories import InventoryRepository
from proto import client_pb2
from tests.conftest import TEST_PASSWORD


# --- fixtures --------------------------------------------------------------
@pytest.fixture
def catalogue(session_factory: sessionmaker[DbSession]) -> dict[str, Any]:
    """Two warehouses and two products to hang stock rows on."""
    with session_scope(session_factory) as db:
        for code, name in [("WH-1", "North"), ("WH-2", "South")]:
            inventory.create_warehouse(
                db, {"code": code, "name": name, "location": "IN"}
            )
        for sku in ["SKU-1", "SKU-2"]:
            inventory.create_product(
                db, {"sku": sku, "name": f"Item {sku}", "unit_price": 10.0}
            )
    return {"warehouses": ["WH-1", "WH-2"], "products": ["SKU-1", "SKU-2"]}


def stock_key(warehouse: str = "WH-1", sku: str = "SKU-1") -> dict[str, Any]:
    return {"warehouse_code": warehouse, "sku": sku}


def set_stock(
    session_factory: sessionmaker[DbSession],
    *,
    warehouse: str = "WH-1",
    sku: str = "SKU-1",
    **overrides: Any,
) -> Inventory:
    payload = {**stock_key(warehouse, sku), "quantity": 10, **overrides}
    with session_scope(session_factory) as db:
        record = inventory.set_initial_stock(db, payload)
        db.flush()
        return record


def read(
    session_factory: sessionmaker[DbSession],
    warehouse: str = "WH-1",
    sku: str = "SKU-1",
) -> Inventory:
    with session_factory() as db:
        return inventory.get_stock(db, stock_key(warehouse, sku))


class FakeContext:
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


@pytest.fixture
def token(session_factory: sessionmaker[DbSession], auth_user: User) -> str:
    with session_scope(session_factory) as db:
        return auth.login(db, auth_user.username, TEST_PASSWORD).raw_token


def post(service: ClientService, token: str, resource: str, payload: dict[str, Any]):
    request = client_pb2.PostRequest(session_token=token, resource=resource)
    request.payload.update(payload)
    return service.post(request, FakeContext())


def get(service: ClientService, token: str, resource: str, params: dict[str, Any]):
    request = client_pb2.GetRequest(session_token=token, resource=resource)
    request.params.update(params)
    return service.get(request, FakeContext())


# --- set initial stock -----------------------------------------------------
def test_set_initial_stock(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    record = set_stock(session_factory, quantity=25, reorder_point=5, safety_stock=2)
    assert record.id is not None

    stored = read(session_factory)
    assert stored.quantity == 25
    assert stored.reorder_point == 5
    assert stored.safety_stock == 2


def test_initial_quantity_zero_is_valid(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """Zero stock is a legitimate state; only negative is impossible."""
    set_stock(session_factory, quantity=0)
    assert read(session_factory).quantity == 0


def test_defaults_are_used_when_thresholds_are_omitted(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=7)
    stored = read(session_factory)
    assert stored.reorder_point == 0
    assert stored.safety_stock == 0


def test_can_identify_by_id_instead_of_code(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with session_scope(session_factory) as db:
        inventory.set_initial_stock(
            db, {"warehouse_id": 1, "product_id": 1, "quantity": 3}
        )
    assert read(session_factory).quantity == 3


@pytest.mark.parametrize("quantity", [-1, -100])
def test_negative_initial_quantity_rejected(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any], quantity: int
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.set_initial_stock(db, {**stock_key(), "quantity": quantity})

    with session_factory() as db:
        assert InventoryRepository(db).list() == []


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"warehouse_code": "WH-1", "quantity": 1}, "no product"),
        ({"sku": "SKU-1", "quantity": 1}, "no warehouse"),
        (stock_key(), "no quantity"),
        ({**stock_key(), "quantity": 1.5}, "fractional quantity"),
        ({**stock_key(), "quantity": True}, "bool quantity"),
        ({**stock_key(), "quantity": "abc"}, "non-numeric quantity"),
        ({**stock_key(), "quantity": 1, "reorder_point": -1}, "negative reorder"),
        ({**stock_key(), "quantity": 1, "safety_stock": -1}, "negative safety"),
        ({**stock_key(), "quantity": 1, "extra": 1}, "unknown field"),
        (
            {
                "warehouse_code": "WH-1",
                "warehouse_id": 1,
                "sku": "SKU-1",
                "quantity": 1,
            },
            "two warehouse keys",
        ),
    ],
)
def test_invalid_initial_stock_rejected(
    session_factory: sessionmaker[DbSession],
    catalogue: dict[str, Any],
    payload: dict[str, Any],
    reason: str,
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.set_initial_stock(db, payload)


def test_duplicate_initial_stock_rejected(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=10)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.set_initial_stock(db, {**stock_key(), "quantity": 999})


def test_duplicate_initial_stock_does_not_overwrite(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """The rejection must leave the original quantity untouched."""
    set_stock(session_factory, quantity=10)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.set_initial_stock(db, {**stock_key(), "quantity": 999})

    assert read(session_factory).quantity == 10


# --- add stock -------------------------------------------------------------
def test_add_stock(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=10)
    with session_scope(session_factory) as db:
        updated = inventory.add_stock(db, {**stock_key(), "amount": 15})
        assert updated.quantity == 25

    assert read(session_factory).quantity == 25


def test_add_stock_accumulates(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=0)
    for _ in range(3):
        with session_scope(session_factory) as db:
            inventory.add_stock(db, {**stock_key(), "amount": 5})
    assert read(session_factory).quantity == 15


@pytest.mark.parametrize("amount", [0, -1, -50])
def test_add_zero_or_negative_rejected(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any], amount: int
) -> None:
    """add_stock is increment-only: it can never reduce a quantity."""
    set_stock(session_factory, quantity=10)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.add_stock(db, {**stock_key(), "amount": amount})

    assert read(session_factory).quantity == 10


def test_add_stock_requires_an_existing_record(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.add_stock(db, {**stock_key(), "amount": 5})


def test_add_stock_is_a_database_side_increment(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """The new value is computed by the database, not read into Python.

    Simulated by changing the row behind an already-loaded object: a
    read-modify-write would overwrite that change, an atomic increment adds
    to it.
    """
    set_stock(session_factory, quantity=10)

    with session_scope(session_factory) as db:
        stale = inventory.get_stock(db, stock_key())
        assert stale.quantity == 10

        # Another writer commits +100 while `stale` is held.
        with session_scope(session_factory) as other:
            inventory.add_stock(other, {**stock_key(), "amount": 100})

        inventory.add_stock(db, {**stock_key(), "amount": 5})

    # 10 + 100 + 5; a lost update would show 15.
    assert read(session_factory).quantity == 115


# --- get stock -------------------------------------------------------------
def test_get_stock(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=42)
    record = read(session_factory)
    assert record.quantity == 42
    assert record.warehouse.code == "WH-1"
    assert record.product.sku == "SKU-1"


def test_missing_stock_is_not_found(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with session_factory() as db, pytest.raises(inventory.NotFoundError):
        inventory.get_stock(db, stock_key())


def test_nonexistent_warehouse_or_product_is_not_found(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with session_factory() as db:
        with pytest.raises(inventory.NotFoundError):
            inventory.get_stock(db, stock_key(warehouse="NO-SUCH-WH"))
        with pytest.raises(inventory.NotFoundError):
            inventory.get_stock(db, stock_key(sku="NO-SUCH-SKU"))


def test_set_initial_stock_rejects_unknown_references(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.set_initial_stock(
            db, {"warehouse_code": "GHOST", "sku": "SKU-1", "quantity": 1}
        )
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.set_initial_stock(
            db, {"warehouse_code": "WH-1", "sku": "GHOST", "quantity": 1}
        )


# --- list warehouse inventory ---------------------------------------------
def test_list_warehouse_inventory(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, sku="SKU-1", quantity=5)
    set_stock(session_factory, sku="SKU-2", quantity=7)

    with session_factory() as db:
        rows = inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-1"})
    assert [(r.product.sku, r.quantity) for r in rows] == [("SKU-1", 5), ("SKU-2", 7)]


def test_warehouse_inventory_is_isolated(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """One warehouse's stock must never appear under another."""
    set_stock(session_factory, warehouse="WH-1", sku="SKU-1", quantity=5)
    set_stock(session_factory, warehouse="WH-2", sku="SKU-1", quantity=99)

    with session_factory() as db:
        first = inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-1"})
        second = inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-2"})

    assert [r.quantity for r in first] == [5]
    assert [r.quantity for r in second] == [99]


def test_list_ordering_is_deterministic(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, sku="SKU-2", quantity=1)
    set_stock(session_factory, sku="SKU-1", quantity=2)

    with session_factory() as db:
        first = [
            r.product_id
            for r in inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-1"})
        ]
        second = [
            r.product_id
            for r in inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-1"})
        ]
    assert first == second == sorted(first)


def test_list_for_unknown_warehouse_is_not_found(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """An unknown warehouse must not look like an empty one."""
    with session_factory() as db, pytest.raises(inventory.NotFoundError):
        inventory.list_warehouse_inventory(db, {"warehouse_code": "GHOST"})


def test_list_for_warehouse_with_no_stock_is_empty(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with session_factory() as db:
        assert (
            list(inventory.list_warehouse_inventory(db, {"warehouse_code": "WH-2"}))
            == []
        )


# --- low stock -------------------------------------------------------------
def test_low_stock_when_quantity_below_reorder_point(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    record = set_stock(session_factory, quantity=3, reorder_point=10)
    assert inventory.is_low_stock(record) is True
    with session_factory() as db:
        assert [r.product.sku for r in inventory.list_low_stock(db)] == ["SKU-1"]


def test_low_stock_when_quantity_equals_reorder_point(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """The rule is `quantity <= reorder_point`, so equal counts as low."""
    record = set_stock(session_factory, quantity=10, reorder_point=10)
    assert inventory.is_low_stock(record) is True
    with session_factory() as db:
        assert len(inventory.list_low_stock(db)) == 1


def test_not_low_when_quantity_above_reorder_point(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    record = set_stock(session_factory, quantity=11, reorder_point=10)
    assert inventory.is_low_stock(record) is False
    with session_factory() as db:
        assert list(inventory.list_low_stock(db)) == []


def test_low_stock_can_be_scoped_to_one_warehouse(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(
        session_factory, warehouse="WH-1", sku="SKU-1", quantity=1, reorder_point=5
    )
    set_stock(
        session_factory, warehouse="WH-2", sku="SKU-1", quantity=1, reorder_point=5
    )

    with session_factory() as db:
        everywhere = inventory.list_low_stock(db)
        just_one = inventory.list_low_stock(db, {"warehouse_code": "WH-1"})
    assert len(everywhere) == 2
    assert [r.warehouse.code for r in just_one] == ["WH-1"]


def test_adding_stock_clears_the_low_flag(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=2, reorder_point=5)
    with session_scope(session_factory) as db:
        assert len(inventory.list_low_stock(db)) == 1
        inventory.add_stock(db, {**stock_key(), "amount": 10})
    with session_factory() as db:
        assert list(inventory.list_low_stock(db)) == []


# --- the invariant, at the database level ----------------------------------
def test_database_rejects_negative_quantity_directly(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """The CHECK constraint holds even when the domain layer is bypassed."""
    with pytest.raises(IntegrityError), session_factory() as db:
        db.add(Inventory(warehouse_id=1, product_id=1, quantity=-1))
        db.flush()


def test_database_rejects_negative_thresholds_directly(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    for field in ("reorder_point", "safety_stock"):
        with pytest.raises(IntegrityError), session_factory() as db:
            db.add(Inventory(warehouse_id=1, product_id=1, **{field: -1}))
            db.flush()


def test_duplicate_pair_rejected_by_the_database(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=1)
    with pytest.raises(IntegrityError), session_factory() as db:
        db.add(Inventory(warehouse_id=1, product_id=1, quantity=2))
        db.flush()


# --- failure leaves state unchanged ---------------------------------------
def test_failed_operation_leaves_previous_quantity_unchanged(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=20)

    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.add_stock(db, {**stock_key(), "amount": 5})
        inventory.add_stock(db, {**stock_key(), "amount": -1})  # rolls both back

    assert read(session_factory).quantity == 20


def test_failed_operation_does_not_disturb_other_rows(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, warehouse="WH-1", quantity=20)
    set_stock(session_factory, warehouse="WH-2", quantity=30)

    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.add_stock(db, {**stock_key(), "amount": 5})
        inventory.add_stock(db, {**stock_key(sku="SKU-2"), "amount": 5})

    assert read(session_factory, "WH-1").quantity == 20
    assert read(session_factory, "WH-2").quantity == 30


# --- serialization ---------------------------------------------------------
def test_serialize_inventory_is_struct_safe(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    set_stock(session_factory, quantity=4, reorder_point=10)
    with session_factory() as db:
        payload = inventory.serialize_inventory(inventory.get_stock(db, stock_key()))

    assert payload["sku"] == "SKU-1"
    assert payload["warehouse_code"] == "WH-1"
    assert payload["is_low_stock"] is True
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool, type(None)))

    item = client_pb2.DataItem(id="1")
    item.data.update(payload)
    assert item.data["quantity"] == 4


# --- servicer status mapping -----------------------------------------------
def test_servicer_set_add_and_read(
    service: ClientService, token: str, catalogue: dict[str, Any]
) -> None:
    created = post(
        service,
        token,
        "inventory.set_initial_stock",
        {**stock_key(), "quantity": 10, "reorder_point": 4},
    )
    assert created.success is True

    post(service, token, "inventory.add_stock", {**stock_key(), "amount": 5})

    read_back = get(service, token, "inventory.stock", stock_key())
    assert read_back.items[0].data["quantity"] == 15
    assert read_back.items[0].data["is_low_stock"] is False


def test_servicer_low_stock_listing(
    service: ClientService, token: str, catalogue: dict[str, Any]
) -> None:
    post(
        service,
        token,
        "inventory.set_initial_stock",
        {**stock_key(), "quantity": 2, "reorder_point": 5},
    )
    post(
        service,
        token,
        "inventory.set_initial_stock",
        {**stock_key(sku="SKU-2"), "quantity": 50, "reorder_point": 5},
    )

    low = get(service, token, "inventory.low_stock", {})
    assert [item.data["sku"] for item in low.items] == ["SKU-1"]


@pytest.mark.parametrize(
    ("resource", "payload", "expected"),
    [
        (
            "inventory.set_initial_stock",
            {"warehouse_code": "WH-1", "sku": "SKU-1", "quantity": -1},
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            "inventory.set_initial_stock",
            {"warehouse_code": "GHOST", "sku": "SKU-1", "quantity": 1},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            "inventory.add_stock",
            {"warehouse_code": "WH-1", "sku": "SKU-1", "amount": 0},
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            "inventory.add_stock",
            {"warehouse_code": "WH-1", "sku": "SKU-1", "amount": 5},
            grpc.StatusCode.NOT_FOUND,
        ),
    ],
)
def test_servicer_post_status_mapping(
    service: ClientService,
    token: str,
    catalogue: dict[str, Any],
    resource: str,
    payload: dict[str, Any],
    expected: grpc.StatusCode,
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, resource, payload)
    assert excinfo.value.code == expected


def test_servicer_duplicate_initial_stock_is_already_exists(
    service: ClientService, token: str, catalogue: dict[str, Any]
) -> None:
    payload = {**stock_key(), "quantity": 1}
    post(service, token, "inventory.set_initial_stock", payload)
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "inventory.set_initial_stock", payload)
    assert excinfo.value.code == grpc.StatusCode.ALREADY_EXISTS


def test_servicer_missing_stock_is_not_found(
    service: ClientService, token: str, catalogue: dict[str, Any]
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        get(service, token, "inventory.stock", stock_key())
    assert excinfo.value.code == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize(
    "resource",
    [
        "inventory.set_initial_stock",
        "inventory.stock",
        "inventory.warehouse",
        "inventory.low_stock",
    ],
)
def test_inventory_actions_require_authentication(
    service: ClientService, resource: str
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        if resource == "inventory.set_initial_stock":
            post(service, "bogus-token", resource, {**stock_key(), "quantity": 1})
        else:
            get(service, "bogus-token", resource, stock_key())
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


def test_earlier_steps_still_work(
    service: ClientService, token: str, catalogue: dict[str, Any]
) -> None:
    """Steps 23 and 24 must keep working alongside step 25."""
    products = get(service, token, "products", {})
    warehouses = get(service, token, "warehouses", {})
    assert [i.data["sku"] for i in products.items] == ["SKU-1", "SKU-2"]
    assert [i.data["code"] for i in warehouses.items] == ["WH-1", "WH-2"]

    post(
        service,
        token,
        "product.create",
        {"sku": "SKU-3", "name": "Third", "unit_price": 1.5},
    )
    assert len(get(service, token, "products", {}).items) == 3


def test_stock_row_survives_product_price_update(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    """Catalogue edits must not disturb stock levels."""
    set_stock(session_factory, quantity=12)
    with session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-1", "unit_price": 99.5})

    assert read(session_factory).quantity == 12
    with session_factory() as db:
        assert inventory.get_product(db, sku="SKU-1").unit_price == Decimal("99.50")


def test_products_and_warehouses_are_distinct_entities(
    session_factory: sessionmaker[DbSession], catalogue: dict[str, Any]
) -> None:
    with session_factory() as db:
        assert isinstance(inventory.get_product(db, sku="SKU-1"), Product)
        assert isinstance(inventory.get_warehouse(db, code="WH-1"), Warehouse)
