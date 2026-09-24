"""Sales history (Phase F step 29).

Sales history is not a separate operation -- it is written as part of the
atomic order transaction from step 27. These tests therefore prove two
things: that a successful order produces exactly the right history rows,
and that a *failed* order produces none at all. A false sales row would
corrupt the demand forecasting built on this table later.
"""

from __future__ import annotations

import threading
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import inventory, orders
from app.storage.database import session_scope
from app.storage.models import User, ensure_utc, utc_now
from app.storage.repositories import (
    InventoryRepository,
    OrderItemRepository,
    OrderRepository,
    SalesHistoryRepository,
)

STOCK = {"SKU-1": 100, "SKU-2": 20, "SKU-3": 0}
PRICES = {"SKU-1": "10.00", "SKU-2": "2.50", "SKU-3": "7.25"}


# --- fixtures --------------------------------------------------------------
@pytest.fixture
def world(session_factory: sessionmaker[DbSession], auth_user: User) -> dict[str, Any]:
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
):
    with session_scope(session_factory) as db:
        actor = db.get(User, world["user_id"])
        order = orders.place_order(
            db, actor, {"warehouse_code": warehouse, "items": items}
        )
        db.flush()
        return order


def history(session_factory: sessionmaker[DbSession]) -> list[dict[str, Any]]:
    with session_factory() as db:
        return [
            {
                "order_id": row.order_id,
                "warehouse_id": row.warehouse_id,
                "product_id": row.product_id,
                "quantity": row.quantity,
                "unit_price": row.unit_price,
                "sold_at": row.sold_at,
            }
            for row in SalesHistoryRepository(db).list()
        ]


def counts(session_factory: sessionmaker[DbSession]) -> dict[str, int]:
    with session_factory() as db:
        return {
            "orders": len(OrderRepository(db).list()),
            "items": len(OrderItemRepository(db).list()),
            "sales": len(SalesHistoryRepository(db).list()),
        }


# --- a successful single-item order ---------------------------------------
def test_successful_one_item_order_writes_sales_history(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(session_factory, world, [{"sku": "SKU-1", "quantity": 3}])

    rows = history(session_factory)
    assert len(rows) == 1
    assert rows[0]["order_id"] == order.id


def test_sales_history_quantity_is_correct(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 7}])
    assert history(session_factory)[0]["quantity"] == 7


def test_sales_history_product_is_correct(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    place(session_factory, world, [{"sku": "SKU-2", "quantity": 1}])

    with session_factory() as db:
        row = SalesHistoryRepository(db).list()[0]
        assert row.product.sku == "SKU-2"
        assert row.product_id == 2


def test_sales_history_warehouse_is_correct(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])

    with session_factory() as db:
        row = SalesHistoryRepository(db).list()[0]
        assert row.warehouse.code == "WH-1"
        assert row.warehouse_id == 1


def test_sold_at_is_populated_and_server_generated(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The timestamp comes from the server clock, near the moment of sale."""
    before = utc_now()
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])
    after = utc_now()

    sold_at = ensure_utc(history(session_factory)[0]["sold_at"])
    assert before <= sold_at <= after


def test_sales_history_records_the_price_charged(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The unit price is captured at sale time, not looked up later."""
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 2}])
    assert history(session_factory)[0]["unit_price"] == Decimal("10.00")

    # A later catalogue price change must not rewrite recorded history.
    with session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-1", "unit_price": 99.00})

    assert history(session_factory)[0]["unit_price"] == Decimal("10.00")


def test_client_cannot_supply_sold_at(
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
                "sold_at": "1970-01-01T00:00:00+00:00",
            },
        )
    assert history(session_factory) == []


# --- a successful multi-item order ----------------------------------------
def test_multi_item_order_writes_one_row_per_product(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    order = place(
        session_factory,
        world,
        [
            {"sku": "SKU-1", "quantity": 4},
            {"sku": "SKU-2", "quantity": 6},
        ],
    )

    with session_factory() as db:
        rows = SalesHistoryRepository(db).list_for_order(order.id)
        by_sku = {row.product.sku: row.quantity for row in rows}

    assert len(rows) == 2
    assert by_sku == {"SKU-1": 4, "SKU-2": 6}


def test_multi_item_quantities_match_each_product(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Audit point 8: verify the quantity for every product individually."""
    requested = {"SKU-1": 11, "SKU-2": 3}
    place(
        session_factory,
        world,
        [{"sku": sku, "quantity": q} for sku, q in requested.items()],
    )

    with session_factory() as db:
        recorded = {
            row.product.sku: row.quantity for row in SalesHistoryRepository(db).list()
        }
    assert recorded == requested


def test_sales_rows_in_one_order_share_a_timestamp(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """One order is one moment in time, not one moment per line."""
    order = place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 1}, {"sku": "SKU-2", "quantity": 1}],
    )

    with session_factory() as db:
        rows = SalesHistoryRepository(db).list_for_order(order.id)
        stamps = {row.sold_at for row in rows}
        assert len(stamps) == 1
        # Normalised on both sides: the reloaded row is naive (SQLite drops
        # tzinfo) while the in-session Order still holds the aware value.
        assert ensure_utc(rows[0].sold_at) == ensure_utc(order.completed_at)


def test_duplicate_lines_produce_one_consolidated_sales_row(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-1", "quantity": 3}],
    )

    rows = history(session_factory)
    assert len(rows) == 1
    assert rows[0]["quantity"] == 5


# --- failed orders write nothing ------------------------------------------
def test_insufficient_stock_order_writes_zero_sales_rows(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """A rejected sale must never appear as a completed one."""
    with pytest.raises(orders.InsufficientStockError):
        place(session_factory, world, [{"sku": "SKU-2", "quantity": 999}])

    assert history(session_factory) == []
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


@pytest.mark.parametrize(
    ("items", "warehouse", "expected_error"),
    [
        ([{"sku": "SKU-1", "quantity": 0}], "WH-1", inventory.ValidationError),
        ([{"sku": "SKU-1", "quantity": -5}], "WH-1", inventory.ValidationError),
        ([{"sku": "GHOST", "quantity": 1}], "WH-1", inventory.NotFoundError),
        ([{"sku": "SKU-1", "quantity": 1}], "GHOST", inventory.NotFoundError),
        ([{"sku": "SKU-1", "quantity": 1}], "WH-2", inventory.NotFoundError),
        ([{"sku": "SKU-3", "quantity": 1}], "WH-1", orders.InsufficientStockError),
    ],
)
def test_no_failure_mode_writes_sales_history(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    items: list[dict[str, Any]],
    warehouse: str,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        place(session_factory, world, items, warehouse=warehouse)
    assert history(session_factory) == []


def test_rollback_after_partial_workflow_leaves_no_new_history(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """A prior successful order's history survives; the failed one adds none."""
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])
    baseline = history(session_factory)
    assert len(baseline) == 1

    # SKU-1 succeeds and is decremented, then SKU-2 fails.
    with pytest.raises(orders.InsufficientStockError):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 5}, {"sku": "SKU-2", "quantity": 5000}],
        )

    assert history(session_factory) == baseline, "a failed order added history"
    assert counts(session_factory) == {"orders": 1, "items": 1, "sales": 1}


def test_injected_failure_after_first_sales_row_rolls_back(
    session_factory: sessionmaker[DbSession],
    world: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail after one sales row is written; it must not survive."""
    calls = {"n": 0}
    original = orders.SalesHistoryRepository.create

    def fail_on_second(self: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure after the first sales row")
        return original(self, **kwargs)

    monkeypatch.setattr(orders.SalesHistoryRepository, "create", fail_on_second)

    with pytest.raises(RuntimeError, match="injected failure"):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 1}, {"sku": "SKU-2", "quantity": 1}],
        )

    assert calls["n"] == 2
    assert history(session_factory) == []
    assert counts(session_factory) == {"orders": 0, "items": 0, "sales": 0}


# --- transactional consistency across all four tables ----------------------
def test_all_four_tables_stay_consistent_after_success(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Order, OrderItem, SalesHistory and Inventory agree with each other."""
    order = place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 9}, {"sku": "SKU-2", "quantity": 4}],
    )

    with session_factory() as db:
        items = {
            i.product_id: i for i in OrderItemRepository(db).list_for_order(order.id)
        }
        sales = {
            s.product_id: s for s in SalesHistoryRepository(db).list_for_order(order.id)
        }
        stock = {
            r.product_id: r.quantity
            for r in InventoryRepository(db).list_for_warehouse(order.warehouse_id)
        }

    assert set(items) == set(sales), "an item without history, or vice versa"
    for product_id, item in items.items():
        sale = sales[product_id]
        # Every line agrees on quantity, price and warehouse.
        assert sale.quantity == item.quantity
        assert sale.unit_price == item.unit_price
        assert sale.warehouse_id == order.warehouse_id
        assert sale.order_id == order.id
        # And the stock actually moved by that amount.
        sku = "SKU-1" if product_id == 1 else "SKU-2"
        assert stock[product_id] == STOCK[sku] - item.quantity


def test_all_four_tables_stay_consistent_after_failure(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    with pytest.raises(orders.InsufficientStockError):
        place(
            session_factory,
            world,
            [{"sku": "SKU-1", "quantity": 1}, {"sku": "SKU-3", "quantity": 1}],
        )

    with session_factory() as db:
        assert OrderRepository(db).list() == []
        assert OrderItemRepository(db).list() == []
        assert SalesHistoryRepository(db).list() == []
        stock = {
            r.product.sku: r.quantity
            for r in InventoryRepository(db).list_for_warehouse(1)
        }
    assert stock == STOCK


def test_sales_history_is_not_committed_before_the_caller_commits(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Point 5: the repository must not end the transaction itself."""
    with session_factory() as db:
        actor = db.get(User, world["user_id"])
        orders.place_order(
            db,
            actor,
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-1", "quantity": 1}]},
        )
        # A separate connection must not see the history yet.
        assert history(session_factory) == []
        db.rollback()

    assert history(session_factory) == []


def test_sales_history_repository_never_commits() -> None:
    import inspect

    source = inspect.getsource(SalesHistoryRepository)
    assert ".commit()" not in source


# --- retrieval helpers -----------------------------------------------------
def test_history_ordering_is_deterministic(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """Ordered by (sold_at, id), so repeated reads agree."""
    for _ in range(3):
        place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])

    with session_factory() as db:
        repository = SalesHistoryRepository(db)
        first = [r.id for r in repository.list()]
        second = [r.id for r in repository.list()]

    assert first == second == sorted(first)


def test_list_for_product_filters_and_orders(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """The query shape the forecasting layer will need."""
    place(
        session_factory,
        world,
        [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-2", "quantity": 3}],
    )
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 4}])

    with session_factory() as db:
        rows = SalesHistoryRepository(db).list_for_product(1)
        assert [r.quantity for r in rows] == [2, 4]
        assert all(r.product_id == 1 for r in rows)

        scoped = SalesHistoryRepository(db).list_for_product(1, warehouse_id=1)
        assert len(scoped) == 2
        assert SalesHistoryRepository(db).list_for_product(1, warehouse_id=99) == []


def test_list_for_product_since_window(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """A cutoff excludes older rows, whatever timezone it is expressed in."""
    place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])

    with session_factory() as db:
        repository = SalesHistoryRepository(db)
        assert (
            len(repository.list_for_product(1, since=utc_now() - timedelta(days=1)))
            == 1
        )
        assert repository.list_for_product(1, since=utc_now() + timedelta(days=1)) == []


def test_list_for_order_scopes_to_one_order(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    first = place(session_factory, world, [{"sku": "SKU-1", "quantity": 1}])
    second = place(session_factory, world, [{"sku": "SKU-2", "quantity": 2}])

    with session_factory() as db:
        repository = SalesHistoryRepository(db)
        assert [r.quantity for r in repository.list_for_order(first.id)] == [1]
        assert [r.quantity for r in repository.list_for_order(second.id)] == [2]


# --- concurrency -----------------------------------------------------------
def test_concurrent_orders_write_exactly_one_history_row_each(
    session_factory: sessionmaker[DbSession], world: dict[str, Any]
) -> None:
    """History must not gain phantom rows from losing transactions."""
    successes: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        try:
            place(session_factory, world, [{"sku": "SKU-2", "quantity": 4}])
            won = True
        except orders.InsufficientStockError:
            won = False
        # Deliberately broad: a worker thread must surface, not swallow.
        except Exception:  # noqa: BLE001
            won = False
        with lock:
            successes.append(won)

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    won = sum(successes)
    assert won == 5  # 20 units / 4 per order
    rows = history(session_factory)
    assert len(rows) == won, "one history row per successful order, no more"
    assert sum(r["quantity"] for r in rows) == 20
