"""Demand statistics and the context sent to the LLM for advice.

Two analyses live here -- demand forecasting and reorder recommendation --
because both answer the same underlying question, "how fast does this move",
and sharing :func:`demand_statistics` is what keeps them from drifting into
two different opinions about the same sales history.

The division of labour with the LLM is the point of the module, and is worth
being explicit about:

* **Every number is computed here, deterministically.** The average daily
  rate, the trend, the baseline forecast and the baseline reorder quantity
  are plain arithmetic over rows from this node's database. Run twice on the
  same data they produce identical output, which is what makes them safe to
  show an operator and, in Milestone 2, safe to replay on another replica.
* **The model explains; it does not calculate.** It receives these figures as
  evidence and writes the advice. Nothing it returns is parsed back into a
  number the system acts on, so a hallucinated quantity cannot reach stock.

Nothing here writes. Every function opens no transaction of its own, calls
only read methods, and leaves the session exactly as it found it.

Sparse data is reported rather than smoothed. A product with two sales in
thirty days gets ``data_quality`` of ``"sparse"`` and keeps its real numbers;
a product with none gets a baseline of ``None`` and an explicit note. Filling
either gap with a plausible-looking estimate would be indistinguishable, to
the reader, from a real forecast.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from sqlalchemy.orm import Session as DbSession

from app.domain.inventory import (
    PRODUCT_KEYS,
    WAREHOUSE_KEYS,
    NotFoundError,
    ValidationError,
    coerce_int,
    is_low_stock,
    reject_unknown_fields,
    require_mapping,
    resolve_product,
    resolve_warehouse,
)
from app.storage.models import (
    Inventory,
    Product,
    SupplierProduct,
    Warehouse,
    ensure_utc,
    utc_now,
)
from app.storage.repositories import (
    InventoryRepository,
    SalesHistoryRepository,
    SupplierProductRepository,
)

__all__ = [
    "DEFAULT_HISTORY_DAYS",
    "DEFAULT_HORIZON_DAYS",
    "DemandStatistics",
    "baseline_forecast",
    "build_demand_forecast_context",
    "build_reorder_context",
    "demand_statistics",
]

#: How far back sales are read when the caller does not say. Thirty days is
#: long enough for a weekly pattern to show and short enough that a product
#: whose demand shifted last month is not judged on the old rate.
DEFAULT_HISTORY_DAYS: Final[int] = 30
MAX_HISTORY_DAYS: Final[int] = 365

DEFAULT_HORIZON_DAYS: Final[int] = 14
MAX_HORIZON_DAYS: Final[int] = 365

#: Daily points are capped so a year of history cannot turn into a prompt
#: nobody can read and a small model cannot hold. The most recent days
#: survive, because they are the ones that predict.
MAX_DAILY_POINTS: Final[int] = 60

#: Below this many days with a sale, a rate is arithmetic rather than
#: evidence, and the context says so.
SPARSE_DAY_THRESHOLD: Final[int] = 3

#: How far a half-over-half change must move before it is called a direction
#: rather than noise.
TREND_SENSITIVITY: Final[float] = 0.10

#: Supplier options included in a reorder context. Ordered preferred-first
#: then cheapest by the repository, so the ones that matter come first.
MAX_SUPPLIER_OPTIONS: Final[int] = 3

_FORECAST_FIELDS: Final[frozenset[str]] = (
    PRODUCT_KEYS | WAREHOUSE_KEYS | frozenset({"horizon_days", "history_days"})
)
_REORDER_FIELDS: Final[frozenset[str]] = (
    PRODUCT_KEYS | WAREHOUSE_KEYS | frozenset({"history_days"})
)


@dataclass(frozen=True)
class DemandStatistics:
    """What the sales history says, before anyone interprets it."""

    window_days: int
    since: datetime
    total_units: int
    days_with_sales: int
    daily_units: tuple[tuple[str, int], ...]
    average_daily: float
    trend: str
    change_percent: float | None
    data_quality: str

    @property
    def has_history(self) -> bool:
        return self.total_units > 0


def _coerce_window(data: Any, field: str, default: int, maximum: int) -> int:
    value = coerce_int(data[field], field, minimum=1) if field in data else default
    if value > maximum:
        raise ValidationError(f"'{field}' must not exceed {maximum}.")
    return value


def _group_by_day(rows: Sequence[Any]) -> dict[str, int]:
    """Total units per calendar day, in chronological order.

    ``ensure_utc`` first: SQLite returns a naive datetime, and taking
    ``.date()`` off an unnormalised value would bucket a sale into the wrong
    day for anyone east or west of UTC.
    """
    daily: dict[str, int] = {}
    for row in rows:
        day = ensure_utc(row.sold_at).date().isoformat()
        daily[day] = daily.get(day, 0) + row.quantity
    return dict(sorted(daily.items()))


def _trend(daily: dict[str, int], since: datetime, window_days: int) -> tuple[str, float | None]:
    """Compare the window's second half against its first.

    Halves rather than a regression: it is explainable in one sentence at a
    viva, needs no library, and answers the only question the advice depends
    on -- is this speeding up or slowing down.
    """
    if not daily:
        return "no_sales", None

    midpoint = (since + timedelta(days=window_days / 2)).date().isoformat()
    first = sum(units for day, units in daily.items() if day < midpoint)
    second = sum(units for day, units in daily.items() if day >= midpoint)

    if first == 0 and second == 0:
        return "no_sales", None
    if first == 0:
        return "rising", None

    change = (second - first) / first
    if change > TREND_SENSITIVITY:
        direction = "rising"
    elif change < -TREND_SENSITIVITY:
        direction = "falling"
    else:
        direction = "steady"
    return direction, round(change * 100, 1)


def demand_statistics(
    db: DbSession,
    product: Product,
    warehouse: Warehouse | None = None,
    *,
    history_days: int = DEFAULT_HISTORY_DAYS,
) -> DemandStatistics:
    """Summarise recent sales of one product, optionally in one warehouse.

    Reads through :meth:`SalesHistoryRepository.list_for_product`, which the
    Phase D schema's ``(product_id, sold_at)`` index exists to serve.
    """
    since = utc_now() - timedelta(days=history_days)
    rows = SalesHistoryRepository(db).list_for_product(
        product.id,
        warehouse_id=warehouse.id if warehouse is not None else None,
        since=since,
    )

    daily = _group_by_day(rows)
    total_units = sum(daily.values())
    days_with_sales = len(daily)

    # Averaged over the whole window, not over the days that had a sale:
    # a product selling ten units on one day of thirty moves at a third of a
    # unit a day, and pretending otherwise would overstate every forecast
    # built on it.
    average_daily = round(total_units / history_days, 3) if history_days else 0.0

    if total_units == 0:
        quality = "none"
    elif days_with_sales < SPARSE_DAY_THRESHOLD:
        quality = "sparse"
    else:
        quality = "adequate"

    direction, change_percent = _trend(daily, since, history_days)
    points = tuple(daily.items())[-MAX_DAILY_POINTS:]

    return DemandStatistics(
        window_days=history_days,
        since=since,
        total_units=total_units,
        days_with_sales=days_with_sales,
        daily_units=points,
        average_daily=average_daily,
        trend=direction,
        change_percent=change_percent,
        data_quality=quality,
    )


def baseline_forecast(stats: DemandStatistics, horizon_days: int) -> int | None:
    """Expected units over the horizon at the observed rate.

    ``None`` when there is no history at all: zero sales support no forecast,
    and returning 0 would read as "we predict none" rather than "we cannot
    say". A sparse history still produces a number, flagged by
    ``data_quality`` so the model can temper the advice built on it.
    """
    if not stats.has_history:
        return None
    projected = Decimal(str(stats.average_daily)) * horizon_days
    return int(projected.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _serialize_product(product: Product) -> dict[str, Any]:
    """Identity only. A forecast does not need a description or timestamps."""
    return {"id": product.id, "sku": product.sku, "name": product.name}


def _serialize_warehouse(warehouse: Warehouse) -> dict[str, Any]:
    return {"id": warehouse.id, "code": warehouse.code, "name": warehouse.name}


def _serialize_supplier_terms(link: SupplierProduct) -> dict[str, Any]:
    """The commercial terms, and nothing else.

    Not :func:`~app.domain.inventory.serialize_supplier_product`, which is
    the client-facing projection and carries surrogate keys and audit
    timestamps. Those are meaningless to a model and, repeated across every
    supplier option, would crowd out the figures the advice depends on. The
    cost stays an exact decimal string.
    """
    return {
        "supplier_name": link.supplier.name,
        "unit_cost": str(link.unit_cost),
        "lead_time_days": link.lead_time_days,
        "minimum_order_qty": link.minimum_order_qty,
        "is_preferred": link.is_preferred,
    }


def _serialize_history(stats: DemandStatistics) -> dict[str, Any]:
    """Render the series so that "no sales" is a statement, not an absence.

    An empty list would be reported to the model as missing data, when in
    fact the history was read and found empty. The count travels with the
    series to keep the two apart.
    """
    return {
        "window_days": stats.window_days,
        "days_with_sales": stats.days_with_sales,
        "days_shown": len(stats.daily_units),
        "days": [{"date": day, "units": units} for day, units in stats.daily_units],
    }


def build_demand_forecast_context(db: DbSession, payload: Any) -> dict[str, Any]:
    """Assemble the DEMAND_FORECAST context for one product and warehouse.

    Keys match the fields :mod:`llm.prompts` labels, so the prompt renders
    them under readable headings; anything the model is not given appears
    there as explicitly unavailable.
    """
    data = require_mapping(payload, "payload")
    reject_unknown_fields(data, _FORECAST_FIELDS)

    product = resolve_product(db, data)
    warehouse = resolve_warehouse(db, data)

    horizon_days = _coerce_window(
        data, "horizon_days", DEFAULT_HORIZON_DAYS, MAX_HORIZON_DAYS
    )
    history_days = _coerce_window(
        data, "history_days", DEFAULT_HISTORY_DAYS, MAX_HISTORY_DAYS
    )

    stats = demand_statistics(db, product, warehouse, history_days=history_days)

    return {
        "product": _serialize_product(product),
        "warehouse": _serialize_warehouse(warehouse),
        "forecast_horizon_days": horizon_days,
        "historical_daily_sales": _serialize_history(stats),
        "recent_sales_total": stats.total_units,
        "recent_daily_average": stats.average_daily,
        "trend": {"direction": stats.trend, "change_percent": stats.change_percent},
        "baseline_forecast": baseline_forecast(stats, horizon_days),
        "data_quality": stats.data_quality,
        "history_window_days": history_days,
    }


def _baseline_reorder_quantity(
    record: Inventory, stats: DemandStatistics, lead_time_days: int | None
) -> tuple[int | None, str]:
    """Units to order to cover the lead time, plus why that number.

    ``ceil(average daily x lead time) + safety stock - on hand``, floored at
    zero. Rounded up rather than to nearest: under-ordering re-runs the
    stock-out this calculation exists to prevent, and one spare unit is the
    cheaper error.

    Returns ``None`` when an input is missing, with the reason, rather than
    substituting a default lead time -- a fabricated seven days would produce
    a real-looking quantity nobody could justify.
    """
    if not stats.has_history:
        return None, "no sales history in the window, so demand is unknown"
    if lead_time_days is None:
        return None, "no supplier lead time is recorded for this product"

    cover = math.ceil(stats.average_daily * lead_time_days) + record.safety_stock
    return max(0, cover - record.quantity), (
        f"{stats.average_daily} units/day over a {lead_time_days}-day lead time, "
        f"plus {record.safety_stock} safety stock, less {record.quantity} on hand"
    )


def build_reorder_context(db: DbSession, payload: Any) -> dict[str, Any]:
    """Assemble the REORDER_RECOMMENDATION context.

    Reads stock, sales and supplier terms; writes nothing. In particular it
    never creates a restock order -- the recommendation that comes back is
    advice for an operator, and acting on it stays a separate, deliberate
    write through the existing Phase F paths.
    """
    data = require_mapping(payload, "payload")
    reject_unknown_fields(data, _REORDER_FIELDS)

    product = resolve_product(db, data)
    warehouse = resolve_warehouse(db, data)
    history_days = _coerce_window(
        data, "history_days", DEFAULT_HISTORY_DAYS, MAX_HISTORY_DAYS
    )

    record = InventoryRepository(db).get_by_warehouse_and_product(
        warehouse.id, product.id
    )
    if record is None:
        raise NotFoundError(
            f"No stock record for product '{product.sku}' in warehouse "
            f"'{warehouse.code}'."
        )

    stats = demand_statistics(db, product, warehouse, history_days=history_days)

    # Preferred first, then cheapest -- the repository's own ordering, reused
    # rather than restated so both callers agree on what "the" supplier is.
    links = SupplierProductRepository(db).list_for_product(product.id)
    preferred = links[0] if links else None
    lead_time_days = preferred.lead_time_days if preferred is not None else None

    quantity, rationale = _baseline_reorder_quantity(record, stats, lead_time_days)

    return {
        "product": _serialize_product(product),
        "warehouse": _serialize_warehouse(warehouse),
        "current_quantity": record.quantity,
        "reorder_point": record.reorder_point,
        "safety_stock": record.safety_stock,
        # The Phase F rule, reused rather than reimplemented, so the advice
        # and the low-stock report can never disagree about the same row.
        "below_reorder_point": is_low_stock(record),
        "recent_demand": {
            "window_days": stats.window_days,
            "units": stats.total_units,
            "daily_average": stats.average_daily,
            "trend": stats.trend,
            "data_quality": stats.data_quality,
        },
        "forecast_demand": baseline_forecast(stats, lead_time_days)
        if lead_time_days is not None
        else None,
        "lead_time_days": lead_time_days,
        "supplier": _serialize_supplier_terms(preferred)
        if preferred is not None
        else None,
        "supplier_options": {
            "count": len(links),
            "items": [
                _serialize_supplier_terms(link)
                for link in links[:MAX_SUPPLIER_OPTIONS]
            ],
        },
        "baseline_reorder_quantity": quantity,
        "baseline_rationale": rationale,
    }
