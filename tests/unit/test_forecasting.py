"""Demand statistics and the two demand-driven contexts (Phase G).

These builders decide what a model is allowed to believe, so the tests care
most about the boundary between measured and invented. A product that has
never sold must produce a baseline of ``None`` and say so; one that sold
twice in a month must keep its real numbers and be labelled sparse. Either
gap filled with a plausible estimate would reach an operator looking exactly
like a real forecast.

Every test runs against a temporary SQLite database through the existing
fixtures, and the last group asserts that none of these read paths writes.
"""

from __future__ import annotations

import inspect
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.orm import Session as DbSession

from app.domain import forecasting
from app.domain.forecasting import (
    DEFAULT_HISTORY_DAYS,
    baseline_forecast,
    build_demand_forecast_context,
    build_reorder_context,
    demand_statistics,
)
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
def scenario(session: DbSession) -> dict[str, Any]:
    """One product in one warehouse, with a user to hang orders from."""
    user = User(
        username="ops", email="ops@example.com", password_hash="x", role="manager"
    )
    product = Product(sku="SKU-1", name="Widget", unit_price=Decimal("24.99"))
    other = Product(sku="SKU-2", name="Gizmo", unit_price=Decimal("5.00"))
    warehouse = Warehouse(code="WH-1", name="Main", location="Bengaluru")
    session.add_all([user, product, other, warehouse])
    session.flush()

    record = Inventory(
        warehouse_id=warehouse.id,
        product_id=product.id,
        quantity=100,
        reorder_point=25,
        safety_stock=10,
    )
    order = Order(
        order_number="ORD-TEST",
        user_id=user.id,
        warehouse_id=warehouse.id,
        status=ORDER_COMPLETED,
        total_amount=Decimal("0.00"),
    )
    session.add_all([record, order])
    session.flush()

    return {
        "product": product,
        "other": other,
        "warehouse": warehouse,
        "record": record,
        "order": order,
    }


def sell(
    session: DbSession,
    scenario: dict[str, Any],
    *,
    days_ago: int,
    units: int,
    product: Product | None = None,
) -> None:
    """Record a sale that happened ``days_ago``, as place_order would have."""
    session.add(
        SalesHistory(
            order_id=scenario["order"].id,
            warehouse_id=scenario["warehouse"].id,
            product_id=(product or scenario["product"]).id,
            quantity=units,
            unit_price=Decimal("24.99"),
            sold_at=utc_now() - timedelta(days=days_ago),
        )
    )
    session.flush()


def add_supplier(
    session: DbSession,
    scenario: dict[str, Any],
    *,
    lead_time_days: int = 7,
    preferred: bool = True,
    name: str = "Acme",
) -> SupplierProduct:
    supplier = Supplier(name=name)
    session.add(supplier)
    session.flush()
    link = SupplierProduct(
        supplier_id=supplier.id,
        product_id=scenario["product"].id,
        unit_cost=Decimal("11.50"),
        lead_time_days=lead_time_days,
        minimum_order_qty=1,
        is_preferred=preferred,
    )
    session.add(link)
    session.flush()
    return link


# --- demand statistics -----------------------------------------------------
def test_sales_are_grouped_by_day(session: DbSession, scenario: dict[str, Any]) -> None:
    sell(session, scenario, days_ago=1, units=3)
    sell(session, scenario, days_ago=1, units=2)
    sell(session, scenario, days_ago=2, units=4)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.total_units == 9
    assert stats.days_with_sales == 2
    assert dict(stats.daily_units)[
        (utc_now() - timedelta(days=1)).date().isoformat()
    ] == 5


def test_the_average_spreads_over_the_whole_window(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """Ten units on one day of thirty is a third of a unit a day, not ten."""
    sell(session, scenario, days_ago=1, units=10)

    stats = demand_statistics(
        session, scenario["product"], scenario["warehouse"], history_days=30
    )

    assert stats.average_daily == round(10 / 30, 3)


def test_sales_outside_the_window_are_ignored(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    sell(session, scenario, days_ago=2, units=5)
    sell(session, scenario, days_ago=90, units=500)

    stats = demand_statistics(
        session, scenario["product"], scenario["warehouse"], history_days=30
    )

    assert stats.total_units == 5


def test_another_product_does_not_leak_in(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    sell(session, scenario, days_ago=1, units=5)
    sell(session, scenario, days_ago=1, units=99, product=scenario["other"])

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.total_units == 5


def test_a_rising_trend_is_detected(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    sell(session, scenario, days_ago=25, units=2)
    sell(session, scenario, days_ago=3, units=20)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.trend == "rising"


def test_a_falling_trend_is_detected(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    sell(session, scenario, days_ago=25, units=20)
    sell(session, scenario, days_ago=3, units=2)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.trend == "falling"


def test_steady_sales_are_not_called_a_trend(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    # Balanced either side of the 15-day midpoint: two sales of five units in
    # each half, so there is nothing for the comparison to find.
    for day in (3, 9, 21, 27):
        sell(session, scenario, days_ago=day, units=5)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.trend == "steady"
    assert stats.change_percent == 0.0


@pytest.mark.parametrize(
    ("sale_days", "expected"),
    [([], "none"), ([2], "sparse"), ([2, 5, 9, 14], "adequate")],
)
def test_data_quality_reflects_how_much_history_exists(
    session: DbSession, scenario: dict[str, Any], sale_days: list[int], expected: str
) -> None:
    for day in sale_days:
        sell(session, scenario, days_ago=day, units=4)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.data_quality == expected


# --- baseline forecast -----------------------------------------------------
def test_the_baseline_projects_the_observed_rate(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    for day in range(1, 11):
        sell(session, scenario, days_ago=day, units=3)

    stats = demand_statistics(
        session, scenario["product"], scenario["warehouse"], history_days=30
    )

    assert baseline_forecast(stats, 30) == 30


def test_no_history_gives_no_baseline(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """Zero is a prediction; None is an admission. They are not the same."""
    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert baseline_forecast(stats, 14) is None


def test_sparse_history_still_produces_a_flagged_baseline(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    sell(session, scenario, days_ago=2, units=30)

    stats = demand_statistics(session, scenario["product"], scenario["warehouse"])

    assert stats.data_quality == "sparse"
    assert baseline_forecast(stats, 30) == 30


# --- demand forecast context -----------------------------------------------
def test_demand_context_with_history(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    for day in range(1, 8):
        sell(session, scenario, days_ago=day, units=2)

    context = build_demand_forecast_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1", "horizon_days": 7}
    )

    assert context["product"]["sku"] == "SKU-1"
    assert context["warehouse"]["code"] == "WH-1"
    assert context["forecast_horizon_days"] == 7
    assert context["recent_sales_total"] == 14
    assert context["historical_daily_sales"]["days_with_sales"] == 7
    assert len(context["historical_daily_sales"]["days"]) == 7
    assert context["baseline_forecast"] is not None
    assert context["data_quality"] == "adequate"


def test_demand_context_without_any_sales(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """The absence is stated; nothing is invented to fill it."""
    context = build_demand_forecast_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["recent_sales_total"] == 0
    assert context["baseline_forecast"] is None
    assert context["data_quality"] == "none"
    assert context["trend"]["direction"] == "no_sales"
    assert context["historical_daily_sales"]["days"] == []
    # The series reports itself as empty rather than being missing, so the
    # prompt cannot describe a history that was read as one that was not.
    assert context["historical_daily_sales"]["days_with_sales"] == 0


def test_the_horizon_defaults_when_not_given(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    context = build_demand_forecast_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["forecast_horizon_days"] == forecasting.DEFAULT_HORIZON_DAYS
    assert context["history_window_days"] == DEFAULT_HISTORY_DAYS


def test_the_daily_series_is_capped(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """A year of history must not become a prompt nobody can read."""
    for day in range(1, 200):
        sell(session, scenario, days_ago=day, units=1)

    context = build_demand_forecast_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1", "history_days": 365}
    )

    series = context["historical_daily_sales"]
    assert len(series["days"]) == forecasting.MAX_DAILY_POINTS
    # The true count still travels, so the cap cannot mislead the reader.
    assert series["days_with_sales"] == 199


# --- validation ------------------------------------------------------------
def test_an_unknown_product_is_not_found(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    with pytest.raises(NotFoundError, match="product"):
        build_demand_forecast_context(
            session, {"sku": "NOPE", "warehouse_code": "WH-1"}
        )


def test_an_unknown_warehouse_is_not_found(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    with pytest.raises(NotFoundError, match="warehouse"):
        build_demand_forecast_context(session, {"sku": "SKU-1", "warehouse_code": "NO"})


@pytest.mark.parametrize(
    "payload",
    [
        {"warehouse_code": "WH-1"},
        {"sku": "SKU-1"},
        {"sku": "SKU-1", "product_id": 1, "warehouse_code": "WH-1"},
    ],
)
def test_an_ambiguous_or_incomplete_identifier_is_rejected(
    session: DbSession, scenario: dict[str, Any], payload: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        build_demand_forecast_context(session, payload)


def test_an_unknown_field_is_rejected(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="Unknown"):
        build_demand_forecast_context(
            session, {"sku": "SKU-1", "warehouse_code": "WH-1", "magic": 1}
        )


@pytest.mark.parametrize("horizon", [0, -5, 400])
def test_an_unusable_horizon_is_rejected(
    session: DbSession, scenario: dict[str, Any], horizon: int
) -> None:
    with pytest.raises(ValidationError):
        build_demand_forecast_context(
            session,
            {"sku": "SKU-1", "warehouse_code": "WH-1", "horizon_days": horizon},
        )


# --- reorder context -------------------------------------------------------
def test_reorder_context_when_stock_is_low(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    scenario["record"].quantity = 5
    session.flush()
    for day in range(1, 11):
        sell(session, scenario, days_ago=day, units=3)
    add_supplier(session, scenario, lead_time_days=7)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["current_quantity"] == 5
    assert context["reorder_point"] == 25
    assert context["below_reorder_point"] is True
    assert context["lead_time_days"] == 7
    assert context["supplier"]["supplier_name"] == "Acme"
    assert context["baseline_reorder_quantity"] > 0
    assert "lead time" in context["baseline_rationale"]


def test_reorder_context_when_stock_is_adequate(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    for day in range(1, 11):
        sell(session, scenario, days_ago=day, units=3)
    add_supplier(session, scenario, lead_time_days=7)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["current_quantity"] == 100
    assert context["below_reorder_point"] is False
    # Cover needed is well under what is on hand, so the baseline floors at
    # zero rather than going negative.
    assert context["baseline_reorder_quantity"] == 0


def test_the_baseline_covers_lead_time_and_safety_stock(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """ceil(1 unit/day x 10 days) + 10 safety - 4 on hand = 16."""
    scenario["record"].quantity = 4
    session.flush()
    for day in range(1, 31):
        sell(session, scenario, days_ago=day, units=1)
    add_supplier(session, scenario, lead_time_days=10)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["baseline_reorder_quantity"] == 16


def test_a_missing_supplier_gives_no_baseline(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """No lead time on file means no defensible quantity, so none is offered."""
    scenario["record"].quantity = 5
    session.flush()
    for day in range(1, 11):
        sell(session, scenario, days_ago=day, units=3)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["supplier"] is None
    assert context["lead_time_days"] is None
    assert context["baseline_reorder_quantity"] is None
    assert "lead time" in context["baseline_rationale"]
    assert context["supplier_options"]["count"] == 0
    # Still reports the shortfall, so the advice is not blocked by the gap.
    assert context["below_reorder_point"] is True


def test_no_sales_history_gives_no_baseline(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    scenario["record"].quantity = 5
    session.flush()
    add_supplier(session, scenario, lead_time_days=7)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["baseline_reorder_quantity"] is None
    assert "no sales history" in context["baseline_rationale"]


def test_the_preferred_supplier_leads(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """Reuses the repository's preferred-then-cheapest ordering."""
    add_supplier(session, scenario, lead_time_days=21, preferred=False, name="Slow")
    add_supplier(session, scenario, lead_time_days=3, preferred=True, name="Fast")

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["supplier"]["supplier_name"] == "Fast"
    assert context["lead_time_days"] == 3
    assert context["supplier_options"]["count"] == 2


def test_a_product_with_no_stock_record_is_not_found(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    with pytest.raises(NotFoundError, match="stock record"):
        build_reorder_context(session, {"sku": "SKU-2", "warehouse_code": "WH-1"})


def test_money_reaches_the_context_as_an_exact_string(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    add_supplier(session, scenario)

    context = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    assert context["supplier"]["unit_cost"] == "11.50"


def test_supplier_terms_are_trimmed_for_the_model(
    session: DbSession, scenario: dict[str, Any]
) -> None:
    """Surrogate keys and audit timestamps are noise in a prompt.

    Phi-3 Mini has a small useful context; row ids and created_at crowd out
    the lead time and cost the advice actually rests on.
    """
    add_supplier(session, scenario)

    supplier = build_reorder_context(
        session, {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )["supplier"]

    assert set(supplier) == {
        "supplier_name",
        "unit_cost",
        "lead_time_days",
        "minimum_order_qty",
        "is_preferred",
    }


# --- no writes -------------------------------------------------------------
@pytest.mark.parametrize(
    ("builder", "payload"),
    [
        (build_demand_forecast_context, {"sku": "SKU-1", "warehouse_code": "WH-1"}),
        (build_reorder_context, {"sku": "SKU-1", "warehouse_code": "WH-1"}),
    ],
)
def test_building_a_context_changes_nothing(
    session: DbSession, scenario: dict[str, Any], builder: Any, payload: dict[str, Any]
) -> None:
    """A read path that writes would be a Raft hazard as well as a bug."""
    sell(session, scenario, days_ago=1, units=5)
    add_supplier(session, scenario)
    session.commit()

    before = {
        model: session.query(model).count()
        for model in (Inventory, SalesHistory, Order, Product, SupplierProduct)
    }
    quantity_before = scenario["record"].quantity

    builder(session, payload)

    assert not session.new
    assert not session.dirty
    assert not session.deleted
    assert {
        model: session.query(model).count()
        for model in (Inventory, SalesHistory, Order, Product, SupplierProduct)
    } == before
    assert scenario["record"].quantity == quantity_before


def test_the_module_never_commits() -> None:
    """The caller owns the transaction, exactly as in Phase F."""
    source = inspect.getsource(forecasting)

    assert ".commit()" not in source
    assert ".rollback()" not in source


def test_the_module_creates_nothing() -> None:
    """No repository write method is reachable from a context builder."""
    source = inspect.getsource(forecasting)

    for forbidden in (".create(", ".add(", ".save(", "session.delete"):
        assert forbidden not in source
