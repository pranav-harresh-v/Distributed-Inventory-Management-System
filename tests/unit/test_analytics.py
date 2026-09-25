"""The inventory summary context (Phase G).

The summary is the one context that can grow without bound, so most of these
tests are about what happens when it does: every collection is capped, and
every cap reports the true total beside the sample. A model shown ten
low-stock rows out of thirty must be told there are thirty, or the summary it
writes will be confidently wrong.

The rest covers the arithmetic a management report rests on -- totals,
per-warehouse rollups, valuation in exact Decimals -- and proves that
building a report writes nothing.
"""

from __future__ import annotations

import inspect
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session as DbSession

from app.domain import analytics
from app.domain.analytics import build_inventory_summary_context
from app.domain.inventory import NotFoundError, ValidationError
from app.storage.models import (
    ORDER_COMPLETED,
    Inventory,
    Order,
    Product,
    SalesHistory,
    Supplier,
    SupplierProduct,
    User,
    Warehouse,
    utc_now,
)


@pytest.fixture
def estate(session: DbSession) -> dict[str, Any]:
    """Two warehouses, three products, and a mix of stock positions."""
    user = User(
        username="ops", email="ops@example.com", password_hash="x", role="manager"
    )
    widget = Product(sku="SKU-1", name="Widget", unit_price=Decimal("10.00"))
    gizmo = Product(sku="SKU-2", name="Gizmo", unit_price=Decimal("2.50"))
    doohickey = Product(sku="SKU-3", name="Doohickey", unit_price=Decimal("7.25"))
    main = Warehouse(code="WH-1", name="Main", location="Bengaluru")
    depot = Warehouse(code="WH-2", name="Depot", location="Pune")
    session.add_all([user, widget, gizmo, doohickey, main, depot])
    session.flush()

    session.add_all(
        [
            # healthy
            Inventory(
                warehouse_id=main.id, product_id=widget.id, quantity=100, reorder_point=25
            ),
            # low
            Inventory(
                warehouse_id=main.id, product_id=gizmo.id, quantity=5, reorder_point=20
            ),
            # out of stock (and therefore also low)
            Inventory(
                warehouse_id=main.id,
                product_id=doohickey.id,
                quantity=0,
                reorder_point=10,
            ),
            # healthy, other warehouse
            Inventory(
                warehouse_id=depot.id, product_id=widget.id, quantity=40, reorder_point=5
            ),
        ]
    )
    order = Order(
        order_number="ORD-TEST",
        user_id=user.id,
        warehouse_id=main.id,
        status=ORDER_COMPLETED,
        total_amount=Decimal("0.00"),
    )
    session.add(order)
    session.flush()

    return {
        "widget": widget,
        "gizmo": gizmo,
        "doohickey": doohickey,
        "main": main,
        "depot": depot,
        "order": order,
    }


def sell(
    session: DbSession,
    estate: dict[str, Any],
    product: Product,
    *,
    units: int,
    days_ago: int = 1,
    warehouse: Warehouse | None = None,
) -> None:
    session.add(
        SalesHistory(
            order_id=estate["order"].id,
            warehouse_id=(warehouse or estate["main"]).id,
            product_id=product.id,
            quantity=units,
            unit_price=product.unit_price,
            sold_at=utc_now() - timedelta(days=days_ago),
        )
    )
    session.flush()


# --- totals across warehouses ----------------------------------------------
def test_the_summary_spans_every_warehouse(
    session: DbSession, estate: dict[str, Any]
) -> None:
    context = build_inventory_summary_context(session)

    totals = context["totals"]
    assert totals["warehouses"] == 2
    assert totals["warehouses_included"] == 2
    assert totals["products_in_catalogue"] == 3
    assert totals["stock_lines"] == 4
    assert totals["total_units"] == 145
    assert context["scope"] == "all warehouses"


def test_stock_is_valued_exactly(session: DbSession, estate: dict[str, Any]) -> None:
    """100x10.00 + 5x2.50 + 0 + 40x10.00 = 1412.50, to the cent."""
    context = build_inventory_summary_context(session)

    assert context["totals"]["stock_value"] == "1412.50"


def test_each_warehouse_is_rolled_up(
    session: DbSession, estate: dict[str, Any]
) -> None:
    context = build_inventory_summary_context(session)

    by_code = {item["code"]: item for item in context["warehouses"]["items"]}
    assert by_code["WH-1"]["total_units"] == 105
    assert by_code["WH-1"]["low_stock_lines"] == 2
    assert by_code["WH-1"]["out_of_stock_lines"] == 1
    assert by_code["WH-2"]["total_units"] == 40
    assert by_code["WH-2"]["low_stock_lines"] == 0


def test_the_summary_can_be_scoped_to_one_warehouse(
    session: DbSession, estate: dict[str, Any]
) -> None:
    context = build_inventory_summary_context(session, {"warehouse_code": "WH-2"})

    assert context["scope"] == "WH-2"
    assert context["totals"]["warehouses_included"] == 1
    assert context["totals"]["total_units"] == 40
    assert context["totals"]["low_stock_lines"] == 0


# --- low and out of stock --------------------------------------------------
def test_low_stock_items_are_reported(
    session: DbSession, estate: dict[str, Any]
) -> None:
    context = build_inventory_summary_context(session)

    low = context["low_stock_items"]
    assert low["count"] == 2
    skus = {item["sku"] for item in low["items"]}
    assert skus == {"SKU-2", "SKU-3"}


def test_the_worst_shortfall_comes_first(
    session: DbSession, estate: dict[str, Any]
) -> None:
    """An operator reads the top of the list, so it must be the urgent end."""
    context = build_inventory_summary_context(session)

    first = context["low_stock_items"]["items"][0]
    assert first["sku"] == "SKU-2"  # 20 - 5 = 15 short, against 10 - 0 = 10


def test_out_of_stock_items_are_reported_separately(
    session: DbSession, estate: dict[str, Any]
) -> None:
    context = build_inventory_summary_context(session)

    out = context["out_of_stock_items"]
    assert out["count"] == 1
    assert out["items"][0]["sku"] == "SKU-3"


def test_nothing_low_is_stated_rather_than_left_blank(
    session: DbSession, estate: dict[str, Any]
) -> None:
    """A count of zero is a finding; an empty list would read as no data."""
    context = build_inventory_summary_context(session, {"warehouse_code": "WH-2"})

    assert context["low_stock_items"]["count"] == 0
    assert context["low_stock_items"]["items"] == []
    assert context["out_of_stock_items"]["count"] == 0


def test_a_long_low_stock_list_is_capped_but_counted(
    session: DbSession, estate: dict[str, Any]
) -> None:
    """The cap must not let the model believe it saw everything."""
    for index in range(30):
        product = Product(
            sku=f"BULK-{index:02d}", name=f"Bulk {index}", unit_price=Decimal("1.00")
        )
        session.add(product)
        session.flush()
        session.add(
            Inventory(
                warehouse_id=estate["main"].id,
                product_id=product.id,
                quantity=0,
                reorder_point=5,
            )
        )
    session.flush()

    context = build_inventory_summary_context(session)

    low = context["low_stock_items"]
    assert low["count"] == 32
    assert low["shown"] == analytics.MAX_LOW_STOCK_ITEMS
    assert len(low["items"]) == analytics.MAX_LOW_STOCK_ITEMS


# --- sales -----------------------------------------------------------------
def test_recent_sales_are_summarised(
    session: DbSession, estate: dict[str, Any]
) -> None:
    sell(session, estate, estate["widget"], units=12)
    sell(session, estate, estate["gizmo"], units=3)

    metrics = build_inventory_summary_context(session)["sales_metrics"]

    assert metrics["units_sold"] == 15
    assert metrics["products_sold"] == 2
    assert metrics["window_days"] == analytics.DEFAULT_SALES_WINDOW_DAYS


def test_the_best_seller_leads_the_top_products(
    session: DbSession, estate: dict[str, Any]
) -> None:
    sell(session, estate, estate["gizmo"], units=3)
    sell(session, estate, estate["widget"], units=12)

    top = build_inventory_summary_context(session)["sales_metrics"]["top_products"]

    assert [item["sku"] for item in top] == ["SKU-1", "SKU-2"]
    assert top[0]["units"] == 12


def test_sales_outside_the_window_are_excluded(
    session: DbSession, estate: dict[str, Any]
) -> None:
    sell(session, estate, estate["widget"], units=12, days_ago=2)
    sell(session, estate, estate["widget"], units=999, days_ago=90)

    metrics = build_inventory_summary_context(session)["sales_metrics"]

    assert metrics["units_sold"] == 12


def test_the_sales_window_is_configurable(
    session: DbSession, estate: dict[str, Any]
) -> None:
    sell(session, estate, estate["widget"], units=12, days_ago=45)

    metrics = build_inventory_summary_context(session, {"sales_window_days": 60})[
        "sales_metrics"
    ]

    assert metrics["window_days"] == 60
    assert metrics["units_sold"] == 12


def test_no_sales_reports_zero_rather_than_nothing(
    session: DbSession, estate: dict[str, Any]
) -> None:
    metrics = build_inventory_summary_context(session)["sales_metrics"]

    assert metrics["units_sold"] == 0
    assert metrics["top_products"] == []


# --- suppliers -------------------------------------------------------------
def test_supplier_coverage_is_summarised(
    session: DbSession, estate: dict[str, Any]
) -> None:
    """Coverage, not a directory: the list itself would dwarf the summary."""
    supplier = Supplier(name="Acme")
    session.add(supplier)
    session.flush()
    session.add(
        SupplierProduct(
            supplier_id=supplier.id,
            product_id=estate["widget"].id,
            unit_cost=Decimal("4.00"),
            lead_time_days=7,
        )
    )
    session.flush()

    suppliers = build_inventory_summary_context(session)["suppliers"]

    assert suppliers["supplier_count"] == 1
    assert suppliers["products_with_supplier"] == 1
    assert suppliers["supplier_product_links"] == 1


def test_no_suppliers_is_reported_as_zero(
    session: DbSession, estate: dict[str, Any]
) -> None:
    suppliers = build_inventory_summary_context(session)["suppliers"]

    assert suppliers["supplier_count"] == 0
    assert suppliers["products_with_supplier"] == 0


# --- validation ------------------------------------------------------------
def test_an_unknown_warehouse_is_not_found(
    session: DbSession, estate: dict[str, Any]
) -> None:
    with pytest.raises(NotFoundError):
        build_inventory_summary_context(session, {"warehouse_code": "NOPE"})


def test_an_unknown_parameter_is_rejected(
    session: DbSession, estate: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="Unknown"):
        build_inventory_summary_context(session, {"everything": True})


@pytest.mark.parametrize("window", [0, -1, 400])
def test_an_unusable_window_is_rejected(
    session: DbSession, estate: dict[str, Any], window: int
) -> None:
    with pytest.raises(ValidationError):
        build_inventory_summary_context(session, {"sales_window_days": window})


def test_an_empty_node_still_produces_a_summary(session: DbSession) -> None:
    """A fresh node has no data; that is a report, not an error."""
    context = build_inventory_summary_context(session)

    assert context["totals"]["stock_lines"] == 0
    assert context["totals"]["total_units"] == 0
    assert context["totals"]["stock_value"] == "0.00"
    assert context["warehouses"]["count"] == 0


# --- no writes -------------------------------------------------------------
def test_building_a_summary_changes_nothing(
    session: DbSession, estate: dict[str, Any]
) -> None:
    sell(session, estate, estate["widget"], units=12)
    session.commit()

    before = {
        model: session.query(model).count()
        for model in (Inventory, SalesHistory, Product, Warehouse, Order)
    }

    build_inventory_summary_context(session)

    assert not session.new
    assert not session.dirty
    assert not session.deleted
    assert {
        model: session.query(model).count()
        for model in (Inventory, SalesHistory, Product, Warehouse, Order)
    } == before


def test_the_module_never_commits() -> None:
    source = inspect.getsource(analytics)

    assert ".commit()" not in source
    assert ".rollback()" not in source


def test_the_module_creates_nothing() -> None:
    source = inspect.getsource(analytics)

    for forbidden in (".create(", ".add(", ".save(", "session.delete"):
        assert forbidden not in source
