"""Order placement (Phase F step 27) -- the correctness-critical path.

Every test here is ultimately about one of three guarantees:

* an order lands whole or not at all;
* stock never goes negative;
* two callers cannot both consume the same units.

Several tests deliberately inject failures partway through the workflow to
prove the rollback covers work already done, rather than only the line that
failed.
"""

from __future__ import annotations

import inspect
import threading
from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth, inventory, orders
from app.storage import repositories as repository_module
from app.storage.database import session_scope
from app.storage.models import ORDER_COMPLETED, Order, User, ensure_utc, utc_now
from app.storage.repositories import (
    InventoryRepository,
    OrderItemRepository,
    OrderRepository,
    SalesHistoryRepository,
)
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

STOCK = {"SKU-1": 100, "SKU-2": 5, "SKU-3": 0}
PRICES = {"SKU-1": "10.00", "SKU-2": "2.50", "SKU-3": "7.25"}


# --- fixtures --------------------------------------------------------------
@pytest.fixture
def world(session_factory: sessionmaker[DbSession], auth_user: User) -> dict[str, Any]:
    """One warehouse, three products, known stock levels.

    SKU-3 deliberately has a stock row holding zero, and WH-2 deliberately
    has no stock rows at all.
    """
    with session_scope(session_factory) as db:
        for code, name in [("WH-1", "North"), ("WH-2", "South")]:
            inventory.create_warehouse(
                db, {"code": code, "name": name, "location": "IN"}
            )
        for sku, price in PRICES.items():
            inventory.create_product(
                db, {"sku": sku, "name": f"Item {sku}", "unit_price": price}
            )
        for sku, quantity in STOCK.items():
            inventory.set_initial_stock(
                db, {"warehouse_code": "WH-1", "sku": sku, "quantity": quantity}
            )
    return {"user_id": auth_user.id}


def place(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    items: list[dict[str, Any]],
    warehouse: str = "WH-1",
) -> Order:
    with session_scope(session_factory) as db:
        actor = db.get(User, world["user_id"])
        order = orders.place_order(
            db, actor, {"warehouse_code": warehouse, "items": items}
        )
        db.flush()
        return order


def stock_levels(session_factory: sessionmaker[DbSession]) -> dict[str, int]:
    with session_factory() as db:
        return {
            record.product.sku: record.quantity
            for record in InventoryRepository(db).list_for_warehouse(1)
        }


def counts(session_factory: sessionmaker[DbSession]) -> dict[str, int]:
    with session_factory() as db:
        return {
            "orders": len(OrderRepository(db).list()),
            "items": len(OrderItemRepository(db).list()),
            "sales": len(SalesHistoryRepository(db).list()),
        }


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


# =========================================================================
# 1-5. the happy paths
# =========================================================================
def test_successful_single_item_order(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(session_factory, world, [{"sku": "SKU-1", "quantity": 4}])

    assert order.order_number.startswith("ORD-")
    assert order.status == ORDER_COMPLETED
    assert order.total_amount == Decimal("40.00")
    assert counts(session_factory) == {"orders": 1, "items": 1, "sales": 1}


def test_successful_multi_item_order(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-2", "quantity": 3}],
    )

    # 2 * 10.00 + 3 * 2.50
    assert order.total_amount == Decimal("27.50")
    assert counts(session_factory) == {"orders": 1, "items": 2, "sales": 2}


def test_inventory_decreases_by_exact_requested_amount(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    before = stock_levels(session_factory)
    place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 7}, {"sku": "SKU-2", "quantity": 1}],
    )
    after = stock_levels(session_factory)

    assert after["SKU-1"] == before["SKU-1"] - 7
    assert after["SKU-2"] == before["SKU-2"] - 1
    assert after["SKU-3"] == before["SKU-3"]  # untouched product unchanged


def test_order_record_created_with_expected_fields(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])

    with session_factory() as db:
        stored = OrderRepository(db).get_by_order_number(order.order_number)
        assert stored is not None
        assert stored.user_id == world["user_id"]
        assert stored.warehouse_id == 1
        assert stored.status == ORDER_COMPLETED
        assert stored.completed_at is not None
        assert stored.total_amount == Decimal("10.00")


def test_correct_order_item_records_created(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 3}, {"sku": "SKU-2", "quantity": 2}],
    )

    with session_factory() as db:
        items = OrderItemRepository(db).list_for_order(order.id)
        rows = [
            (i.product.sku, i.quantity, str(i.unit_price), str(i.subtotal))
            for i in items
        ]
    assert rows == [
        ("SKU-1", 3, "10.00", "30.00"),
        ("SKU-2", 2, "2.50", "5.00"),
    ]


def test_sales_history_rows_use_server_timestamps(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """sold_at is the server's clock, and a client cannot supply one."""
    before = utc_now()
    order = place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])

    with session_factory() as db:
        sales = SalesHistoryRepository(db).list_for_order(order.id)
    assert len(sales) == 1
    assert ensure_utc(sales[0].sold_at) >= before
    assert sales[0].unit_price == Decimal("10.00")


def test_client_supplied_sold_at_is_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {
                "warehouse_code": "WH-1",
                "items": [{"sku": "SKU-1", "quantity": 1}],
                "sold_at": "1999-01-01T00:00:00+00:00",
            },
        )


@pytest.mark.parametrize("forbidden", ["status", "total_amount", "order_number"])
def test_client_cannot_dictate_server_owned_fields(
    session_factory: sessionmaker[DbSession], world: dict[str, Any], forbidden: str
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {
                "warehouse_code": "WH-1",
                "items": [{"sku": "SKU-1", "quantity": 1}],
                forbidden: "tampered",
            },
        )


# =========================================================================
# 6. duplicate product lines
# =========================================================================
def test_duplicate_requested_products_are_consolidated(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Two lines for one product become one line with the summed quantity.

    order_items is unique per (order, product), so writing both would
    violate the schema; summing is what the request means.
    """
    before = stock_levels(session_factory)
    order = place(
        session_factory,
        world,
        [
            {"sku": "SKU-1", "quantity": 3},
            {"sku": "SKU-2", "quantity": 1},
            {"sku": "SKU-1", "quantity": 2},
        ],
    )

    with session_factory() as db:
        items = OrderItemRepository(db).list_for_order(order.id)
        rows = {i.product.sku: i.quantity for i in items}

    assert rows == {"SKU-1": 5, "SKU-2": 1}
    assert len(items) == 2  # not three
    assert stock_levels(session_factory)["SKU-1"] == before["SKU-1"] - 5
    assert counts(session_factory)["sales"] == 2


def test_duplicate_lines_are_summed_before_the_stock_check(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """3 + 3 exceeds the 5 units of SKU-2, even though neither line does."""
    with pytest.raises(orders.InsufficientStockError) as excinfo:
        place(
            session_factory,
            world,
            [{"sku": "SKU-2", "quantity": 3}, {"sku": "SKU-2", "quantity": 3}],
        )
    assert excinfo.value.requested == 6
    assert stock_levels(session_factory)["SKU-2"] == 5


def test_duplicate_lines_identified_by_id_and_sku_are_merged(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The same product named two different ways is still one product."""
    order = place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 2}, {"product_id": 1, "quantity": 3}],
    )
    with session_factory() as db:
        items = OrderItemRepository(db).list_for_order(order.id)
    assert len(items) == 1
    assert items[0].quantity == 5


# =========================================================================
# 7-11. request validation
# =========================================================================
@pytest.mark.parametrize("quantity", [0, -1, -100])
def test_non_positive_quantity_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any], quantity: int
) -> None:
    before = stock_levels(session_factory)
    with pytest.raises(inventory.ValidationError):
        place(session_factory, world, [{"sku": "SKU-1", "quantity": quantity}])

    assert stock_levels(session_factory) == before
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ([], "empty list"),
        ("not a list", "not a list"),
        ([{"sku": "SKU-1"}], "missing quantity"),
        ([{"quantity": 1}], "missing product"),
        ([{"sku": "SKU-1", "quantity": 1.5}], "fractional quantity"),
        ([{"sku": "SKU-1", "quantity": True}], "bool quantity"),
        ([{"sku": "SKU-1", "quantity": "two"}], "non-numeric quantity"),
        ([{"sku": "SKU-1", "quantity": 1, "price": 0}], "unknown line field"),
        (["not a mapping"], "line is not a mapping"),
    ],
)
def test_malformed_request_rejected(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    items: Any,
    reason: str,
) -> None:
    with pytest.raises(inventory.ValidationError):
        place(session_factory, world, items)
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_missing_items_key_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        actor = db.get(User, world["user_id"])
        orders.place_order(db, actor, {"warehouse_code": "WH-1"})


def test_missing_warehouse_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with pytest.raises(inventory.NotFoundError):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 1}],
            warehouse="GHOST",
        )
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_missing_product_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    before = stock_levels(session_factory)
    with pytest.raises(inventory.NotFoundError):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 1}, {"sku": "GHOST", "quantity": 1}],
        )

    assert stock_levels(session_factory) == before
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_missing_inventory_record_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """WH-2 exists but holds no stock rows at all."""
    with pytest.raises(inventory.NotFoundError):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 1}],
            warehouse="WH-2",
        )
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


# =========================================================================
# 12-14. insufficient stock and multi-item rollback
# =========================================================================
def test_insufficient_stock_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with pytest.raises(orders.InsufficientStockError) as excinfo:
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 6}])

    error = excinfo.value
    assert error.sku == "SKU-2"
    assert error.requested == 6
    assert error.available == 5
    assert "SKU-2" in str(error)


def test_ordering_from_a_zero_stock_row_is_rejected(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with pytest.raises(orders.InsufficientStockError):
        place(session_factory, world, [{"sku": "SKU-3", "quantity": 1}])
    assert stock_levels(session_factory)["SKU-3"] == 0


def test_insufficient_stock_leaves_quantity_unchanged(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    before = stock_levels(session_factory)
    with pytest.raises(orders.InsufficientStockError):
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 999}])

    assert stock_levels(session_factory) == before
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_exact_stock_is_allowed(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Taking the last unit is legal; the guard is `>=`, not `>`."""
    place(session_factory, world, [{"sku": "SKU-2", "quantity": 5}])
    assert stock_levels(session_factory)["SKU-2"] == 0


def test_first_item_succeeds_second_fails_rolls_everything_back(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The mandated multi-item rollback case.

    SKU-1 has plenty and is decremented first; SKU-2 then fails. The
    already-applied decrement must be undone along with everything else.
    """
    before = stock_levels(session_factory)

    with pytest.raises(orders.InsufficientStockError) as excinfo:
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 10}, {"sku": "SKU-2", "quantity": 500}],
        )
    assert excinfo.value.sku == "SKU-2"

    after = stock_levels(session_factory)
    assert after["SKU-1"] == before["SKU-1"], "first product was not rolled back"
    assert after["SKU-2"] == before["SKU-2"], "second product changed"
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


# =========================================================================
# 15-16. rollback on an injected exception partway through
# =========================================================================
def test_rollback_when_exception_occurs_after_order_creation(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header exists in the session, then a line write explodes."""
    before = stock_levels(session_factory)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected failure after the order header")

    monkeypatch.setattr(orders.OrderItemRepository, "create", boom)

    with pytest.raises(RuntimeError, match="injected failure"):
        place(session_factory, world, [{"sku": "SKU-1", "quantity": 2}])

    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}
    assert stock_levels(session_factory) == before


def test_rollback_when_exception_occurs_after_one_decrement(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail on the second line, after the first has already decremented."""
    before = stock_levels(session_factory)
    calls = {"n": 0}
    original = orders.SalesHistoryRepository.create

    def fail_on_second(self: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure after one decrement")
        return original(self, **kwargs)

    monkeypatch.setattr(orders.SalesHistoryRepository, "create", fail_on_second)

    with pytest.raises(RuntimeError, match="injected failure"):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 4}, {"sku": "SKU-2", "quantity": 1}],
        )

    assert calls["n"] == 2, "the first line should have been written before the failure"
    assert stock_levels(session_factory) == before
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_rollback_when_caller_raises_after_a_successful_order(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The transaction belongs to the caller, so its failure undoes ours."""
    before = stock_levels(session_factory)

    with pytest.raises(RuntimeError), session_scope(session_factory) as db:
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        raise RuntimeError("caller failed after placing the order")

    assert stock_levels(session_factory) == before
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


# =========================================================================
# 17. stock can never go negative
# =========================================================================
def test_stock_can_never_become_negative(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Hammer a small stock row with oversized requests."""
    for quantity in (6, 10, 99, 5000):
        with pytest.raises(orders.InsufficientStockError):
            place(session_factory, world, [{"sku": "SKU-2", "quantity": quantity}])
        assert stock_levels(session_factory)["SKU-2"] == 5

    # Drain it exactly, then try once more.
    place(session_factory, world, [{"sku": "SKU-2", "quantity": 5}])
    assert stock_levels(session_factory)["SKU-2"] == 0
    with pytest.raises(orders.InsufficientStockError):
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])
    assert stock_levels(session_factory)["SKU-2"] == 0


def test_repeated_orders_drain_stock_without_crossing_zero(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    for _ in range(5):
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])
    assert stock_levels(session_factory)["SKU-2"] == 0

    with pytest.raises(orders.InsufficientStockError):
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])
    assert stock_levels(session_factory)["SKU-2"] == 0


def test_guarded_decrement_refuses_to_go_below_zero(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The repository primitive itself is the guard, tested directly."""
    with session_scope(session_factory) as db:
        repository = InventoryRepository(db)
        assert (
            repository.decrement_quantity(warehouse_id=1, product_id=2, amount=6) == 0
        )  # 5 in stock: no row matches
        assert (
            repository.decrement_quantity(warehouse_id=1, product_id=2, amount=5) == 1
        )  # exactly enough
    assert stock_levels(session_factory)["SKU-2"] == 0


# =========================================================================
# 18. concurrency
# =========================================================================
def test_two_orders_cannot_both_consume_the_same_last_unit(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Exactly one of two racing orders may take the final unit.

    Runs against the Phase D SQLite configuration (WAL, busy_timeout), so
    what is under test is the real deployment behaviour, not a substitute
    database.
    """
    # Bring SKU-2 down to a single unit.
    place(session_factory, world, [{"sku": "SKU-2", "quantity": 4}])
    assert stock_levels(session_factory)["SKU-2"] == 1

    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait(timeout=10)
        try:
            place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])
            outcome = "ok"
        except orders.InsufficientStockError:
            outcome = "insufficient"
        # Deliberately broad: a worker thread must not swallow an
        # unexpected failure, it has to reach the assertion below.
        except Exception as exc:  # noqa: BLE001
            outcome = f"unexpected:{type(exc).__name__}"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(results) == ["insufficient", "ok"], results
    assert stock_levels(session_factory)["SKU-2"] == 0
    assert counts(session_factory)["orders"] == 2  # the drain order plus the winner


def test_concurrent_orders_never_oversell(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Ten threads chase 5 units of SKU-2; at most five may win."""
    successes: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])
            won = True
        except orders.InsufficientStockError:
            won = False
        # Deliberately broad, as above.
        except Exception:  # noqa: BLE001
            won = False
        with lock:
            successes.append(won)

    threads = [threading.Thread(target=attempt) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sum(successes) == 5
    assert stock_levels(session_factory)["SKU-2"] == 0


# =========================================================================
# transaction ownership
# =========================================================================
def test_repositories_used_by_order_placement_never_commit() -> None:
    """No repository may end the transaction out from under the caller."""
    suspects = [
        repository_module.OrderRepository,
        repository_module.OrderItemRepository,
        repository_module.SalesHistoryRepository,
        repository_module.InventoryRepository,
        repository_module.BaseRepository,
    ]
    for repository in suspects:
        source = inspect.getsource(repository)
        assert ".commit()" not in source, f"{repository.__name__} commits"


def test_order_domain_never_commits() -> None:
    source = inspect.getsource(orders)
    assert ".commit()" not in source
    assert "session.commit" not in source


def test_place_order_leaves_the_transaction_open_for_the_caller(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Nothing is visible to another session until the caller commits."""
    with session_factory() as db:
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        # Still uncommitted: a separate connection must not see it.
        assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}
        db.rollback()

    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}
    assert stock_levels(session_factory)["SKU-1"] == STOCK["SKU-1"]


def test_one_commit_produces_the_whole_order(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with session_factory() as db:
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {
                "warehouse_code": "WH-1",
                "items": [
                    {"sku": "SKU-1", "quantity": 1},
                    {"sku": "SKU-2", "quantity": 1},
                ],
            },
        )
        db.commit()  # the single commit

    assert counts(session_factory) == {"orders": 1, "items": 2, "sales": 2}


# =========================================================================
# serialization and the API surface
# =========================================================================
def test_serialize_order_is_struct_safe(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(session_factory, world, [{"sku": "SKU-1", "quantity": 2}])
    with session_factory() as db:
        stored = OrderRepository(db).get_by_order_number(order.order_number)
        payload = orders.serialize_order(stored)

    assert payload["total_amount"] == "20.00"  # exact string, never a double
    assert payload["status"] == ORDER_COMPLETED
    assert payload["items"][0]["subtotal"] == "20.00"

    item = client_pb2.DataItem(id="1")
    item.data.update(payload)
    assert item.data["order_number"] == order.order_number


def test_servicer_places_an_order(
    service: ClientService, token: str, world: dict[str, Any]
) -> None:
    response = post(
        service,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 2}]},
    )
    assert response.success is True
    assert "ORD-" in response.message
    assert "20.00" in response.message
    assert counts(service._session_factory) == {"orders": 1, "items": 1, "sales": 1}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 0}]},
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            {"warehouse_code": "GHOST", "items": [{"sku": "SKU-1", "quantity": 1}]},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            {"warehouse_code": "WH-1", "items": [{"sku": "GHOST", "quantity": 1}]},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            {"warehouse_code": "WH-2", "items": [{"sku": "SKU-1", "quantity": 1}]},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-2", "quantity": 999}]},
            grpc.StatusCode.FAILED_PRECONDITION,
        ),
    ],
)
def test_servicer_status_mapping(
    service: ClientService,
    token: str,
    world: dict[str, Any],
    payload: dict[str, Any],
    expected: grpc.StatusCode,
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "order.place", payload)
    assert excinfo.value.code == expected


def test_insufficient_stock_message_identifies_the_product(
    service: ClientService, token: str, world: dict[str, Any]
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        post(
            service,
            token,
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-2", "quantity": 42}]},
        )
    details = excinfo.value.details
    assert "SKU-2" in details
    assert "42" in details
    for leak in ("sqlalchemy", "traceback", "select", "sqlite"):
        assert leak not in details.lower()


def test_order_placement_requires_authentication(
    service: ClientService, world: dict[str, Any]
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        post(
            service,
            "bogus-token",
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED
    assert counts(service._session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_order_is_attributed_to_the_authenticated_user(
    service: ClientService, token: str, world: dict[str, Any], auth_user: User
) -> None:
    """The order's user_id comes from the session, not from the payload."""
    post(
        service,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
    )
    with service._session_factory() as db:
        assert OrderRepository(db).list()[0].user_id == auth_user.id


def test_earlier_steps_still_work(
    service: ClientService, token: str, world: dict[str, Any]
) -> None:
    """Steps 23-26 must keep working alongside step 27."""
    request = client_pb2.GetRequest(session_token=token, resource="products")
    assert len(service.get(request, FakeContext()).items) == 3

    # Buying the whole stock of SKU-1 drops it to 0, which is at its
    # reorder point, so step 25's low-stock report must now include it.
    post(
        service,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 100}]},
    )
    low = client_pb2.GetRequest(session_token=token, resource="inventory.low_stock")
    low.params.update({"warehouse_code": "WH-1"})
    skus = {item.data["sku"] for item in service.get(low, FakeContext()).items}
    assert skus == {"SKU-1", "SKU-3"}  # SKU-2 still has 5 units


# =========================================================================
# Step 28: negative-inventory protection audit
#
# The two scenarios below are stated explicitly rather than folded into the
# broader rollback tests, because they are the exact boundary the whole
# system is built around: 5 units cannot satisfy a request for 7, and 5
# units exactly satisfy a request for 5. Zero stock is a valid state;
# negative stock is not reachable by any implemented operation.
# =========================================================================
AUDIT_SKU = "SKU-2"  # the `world` fixture stocks this with exactly 5


def test_audit_stock_five_request_seven_is_rejected_completely(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """stock = 5, requested = 7 -> rejected, nothing changes.

    Asserts every outcome the audit requires in one place: the order is
    refused, the stock is untouched at exactly 5, and no Order, OrderItem
    or SalesHistory row is committed.
    """
    assert stock_levels(session_factory)[AUDIT_SKU] == 5

    with pytest.raises(orders.InsufficientStockError) as excinfo:
        place(session_factory, world, [{"sku": AUDIT_SKU, "quantity": 7}])

    error = excinfo.value
    assert error.sku == AUDIT_SKU
    assert error.requested == 7
    assert error.available == 5

    assert stock_levels(session_factory)[AUDIT_SKU] == 5, "stock must remain exactly 5"
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_audit_stock_five_request_seven_maps_to_failed_precondition(
    service: ClientService, token: str, world: dict[str, Any]
) -> None:
    """The same 5-vs-7 case as seen by a client: FAILED_PRECONDITION."""
    with pytest.raises(_Aborted) as excinfo:
        post(
            service,
            token,
            "order.place",
            {
                "warehouse_code": "WH-1",
                "items": [{"sku": AUDIT_SKU, "quantity": 7}],
            },
        )

    assert excinfo.value.code == grpc.StatusCode.FAILED_PRECONDITION
    assert AUDIT_SKU in excinfo.value.details
    assert counts(service._session_factory) == {"orders": 0, "items": 0, "sales": 0}
    assert stock_levels(service._session_factory)[AUDIT_SKU] == 5


def test_audit_stock_five_request_five_succeeds_and_reaches_zero(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """stock = 5, requested = 5 -> success, stock becomes exactly 0.

    Zero is a valid resting state; the guard is `>=`, not `>`.
    """
    assert stock_levels(session_factory)[AUDIT_SKU] == 5

    order = place(session_factory, world, [{"sku": AUDIT_SKU, "quantity": 5}])

    assert order.status == ORDER_COMPLETED
    assert stock_levels(session_factory)[AUDIT_SKU] == 0, "stock must be exactly 0"
    assert counts(session_factory) == {"orders": 1, "items": 1, "sales": 1}


def test_audit_zero_stock_is_a_valid_state_not_an_error(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Draining to zero is normal; the row survives and reads back cleanly."""
    place(session_factory, world, [{"sku": AUDIT_SKU, "quantity": 5}])

    with session_factory() as db:
        record = inventory.get_stock(db, {"warehouse_code": "WH-1", "sku": AUDIT_SKU})
        assert record.quantity == 0
        assert inventory.is_low_stock(record) is True


def test_audit_every_write_path_to_quantity_is_guarded(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The three write paths, exercised against their boundaries.

    These are the only ways `inventory.quantity` can change: INSERT on
    creation, guarded increment, guarded decrement.
    """
    with session_scope(session_factory) as db:
        repository = InventoryRepository(db)

        # 1. INSERT: negative rejected by the domain before it is attempted.
        with pytest.raises(inventory.ValidationError):
            inventory.set_initial_stock(
                db,
                {"warehouse_code": "WH-2", "sku": "SKU-1", "quantity": -1},
            )

        # 2. Increment: only ever adds, so it cannot approach zero.
        with pytest.raises(inventory.ValidationError):
            inventory.add_stock(
                db, {"warehouse_code": "WH-1", "sku": AUDIT_SKU, "amount": -3}
            )

        # 3. Decrement: refuses to match a row that holds too little.
        assert (
            repository.decrement_quantity(warehouse_id=1, product_id=2, amount=6) == 0
        )

    assert stock_levels(session_factory)[AUDIT_SKU] == 5


def test_audit_multi_item_rollback_preserves_every_inventory_row(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Audit point 7: every row, not just the failing one, is restored."""
    before = stock_levels(session_factory)

    with pytest.raises(orders.InsufficientStockError):
        place(
            session_factory,
            world,
            [
                {"sku": "SKU-1", "quantity": 50},  # plenty, decremented first
                {"sku": "SKU-3", "quantity": 1},  # holds 0, fails
                {"sku": AUDIT_SKU, "quantity": 1},  # never reached
            ],
        )

    after = stock_levels(session_factory)
    assert after == before, "every inventory row must be unchanged"
    assert all(quantity >= 0 for quantity in after.values())
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


def test_audit_database_constraint_is_the_final_backstop(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Even bypassing the domain and the guarded UPDATE, the schema refuses.

    Uses the generic repository writer with a hand-mutated row, which is
    the closest an accidental future caller could get to negative stock.
    """
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError), session_factory() as db:
        record = InventoryRepository(db).get_by_warehouse_and_product(1, 2)
        record.quantity = -1  # deliberately illegal
        InventoryRepository(db).save(record)

    assert stock_levels(session_factory)[AUDIT_SKU] == 5


def test_audit_no_sequence_of_operations_can_go_negative(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Interleave adds and orders; quantity stays within [0, inf) throughout."""
    observed: list[int] = []

    for amount, order_quantity in [(5, 8), (0, 2), (3, 10), (1, 1)]:
        if amount:
            with session_scope(session_factory) as db:
                inventory.add_stock(
                    db,
                    {"warehouse_code": "WH-1", "sku": AUDIT_SKU, "amount": amount},
                )
        try:
            place(
                session_factory, world, [{"sku": AUDIT_SKU, "quantity": order_quantity}]
            )
        except orders.InsufficientStockError:
            pass
        observed.append(stock_levels(session_factory)[AUDIT_SKU])

    assert all(quantity >= 0 for quantity in observed), observed
