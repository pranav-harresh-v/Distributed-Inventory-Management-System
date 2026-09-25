"""Inventory analytics: the context behind an INVENTORY_SUMMARY.

Builds a compact picture of what this node holds -- totals, per-warehouse
levels, what is low, what has run out, what is moving -- for a model to turn
into a management summary.

**Compact is a requirement, not a preference.** Phi-3 Mini reads a few
thousand tokens usefully; a node with a thousand stock rows would bury the
three that matter in the other nine hundred and ninety-seven. So every
collection here is capped and, wherever a cap can bite, the true count
travels beside the sample: ``{"count": 37, "items": [...10 items...]}`` tells
the model there are thirty-seven low-stock products and shows the worst ten,
where a bare list of ten would have implied that was all of them.

The ordering of every sample is total -- the sort keys always end in a
tie-breaker -- so the same database produces the same summary twice running.

Nothing here writes, and nothing opens a transaction of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any, Final

from sqlalchemy.orm import Session as DbSession

from app.domain.inventory import (
    WAREHOUSE_KEYS,
    ValidationError,
    coerce_int,
    is_low_stock,
    reject_unknown_fields,
    require_mapping,
    resolve_warehouse,
)
from app.storage.models import Inventory, Warehouse, utc_now
from app.storage.repositories import (
    InventoryRepository,
    ProductRepository,
    SalesHistoryRepository,
    SupplierProductRepository,
    SupplierRepository,
    WarehouseRepository,
)

__all__ = [
    "DEFAULT_SALES_WINDOW_DAYS",
    "build_inventory_summary_context",
]

DEFAULT_SALES_WINDOW_DAYS: Final[int] = 30
MAX_SALES_WINDOW_DAYS: Final[int] = 365

#: Caps. Generous enough to be useful in a demo, small enough that the prompt
#: stays readable and the model stays attentive.
MAX_WAREHOUSES: Final[int] = 25
MAX_LOW_STOCK_ITEMS: Final[int] = 10
MAX_OUT_OF_STOCK_ITEMS: Final[int] = 10
MAX_TOP_PRODUCTS: Final[int] = 5

_SUMMARY_FIELDS: Final[frozenset[str]] = WAREHOUSE_KEYS | frozenset({"sales_window_days"})


def _stock_line(record: Inventory) -> dict[str, Any]:
    """One stock row, trimmed to what a summary reader needs."""
    return {
        "sku": record.product.sku,
        "name": record.product.name,
        "warehouse_code": record.warehouse.code,
        "quantity": record.quantity,
        "reorder_point": record.reorder_point,
    }


def _shortfall(record: Inventory) -> int:
    return record.reorder_point - record.quantity


def _capped(
    records: Sequence[Inventory], limit: int, key: Any = None
) -> dict[str, Any]:
    """A sample plus the true total, so a cap cannot mislead.

    The count is what stops a model reporting "three products are low" when
    it was shown three of thirty.
    """
    ordered = sorted(records, key=key) if key is not None else list(records)
    return {
        "count": len(records),
        "shown": min(len(records), limit),
        "items": [_stock_line(record) for record in ordered[:limit]],
    }


def _valuation(records: Sequence[Inventory]) -> Decimal:
    """Stock value, totalled in Decimal.

    Computed here over rows already loaded rather than as a SQL ``SUM``:
    ``quantity * unit_price`` through the database would come back as a
    float, and this project keeps currency exact by never letting it become
    one. It is returned as a Decimal and rendered as a string.
    """
    total = Decimal("0.00")
    for record in records:
        total += record.product.unit_price * record.quantity
    return total


def _warehouse_rollup(warehouse: Warehouse, records: Sequence[Inventory]) -> dict[str, Any]:
    return {
        "code": warehouse.code,
        "name": warehouse.name,
        "location": warehouse.location,
        "product_lines": len(records),
        "total_units": sum(record.quantity for record in records),
        "stock_value": str(_valuation(records)),
        "low_stock_lines": sum(1 for record in records if is_low_stock(record)),
        "out_of_stock_lines": sum(1 for record in records if record.quantity == 0),
    }


def _recent_sales(
    db: DbSession, window_days: int, warehouse: Warehouse | None
) -> dict[str, Any]:
    """Units sold in the window, with the products that moved most.

    One grouped query rather than a scan of every sale: the database returns
    a row per product, and the top few are taken from that.
    """
    since = utc_now() - timedelta(days=window_days)
    sold = SalesHistoryRepository(db).units_by_product_since(
        since, warehouse_id=warehouse.id if warehouse is not None else None
    )

    products = ProductRepository(db)
    top = []
    for product_id, units in sold[:MAX_TOP_PRODUCTS]:
        product = products.get(product_id)
        if product is not None:
            top.append({"sku": product.sku, "name": product.name, "units": units})

    return {
        "window_days": window_days,
        "units_sold": sum(units for _, units in sold),
        "products_sold": len(sold),
        "top_products": top,
    }


def _supplier_overview(db: DbSession) -> dict[str, Any]:
    """Coverage rather than a directory.

    Which products have a supplier on file matters to a restocking decision;
    the supplier list itself does not, and would be the largest thing in the
    prompt.
    """
    links = SupplierProductRepository(db).list(limit=None)
    return {
        "supplier_count": SupplierRepository(db).count(),
        "products_with_supplier": len({link.product_id for link in links}),
        "supplier_product_links": len(links),
    }


def build_inventory_summary_context(db: DbSession, params: Any = None) -> dict[str, Any]:
    """Assemble the INVENTORY_SUMMARY context.

    Scoped to one warehouse when the caller names one, otherwise covering
    every warehouse on this node. Keys line up with the labels
    :mod:`llm.prompts` renders.
    """
    data = require_mapping(params or {}, "params")
    reject_unknown_fields(data, _SUMMARY_FIELDS)

    window_days = (
        coerce_int(data["sales_window_days"], "sales_window_days", minimum=1)
        if "sales_window_days" in data
        else DEFAULT_SALES_WINDOW_DAYS
    )
    if window_days > MAX_SALES_WINDOW_DAYS:
        raise ValidationError(
            f"'sales_window_days' must not exceed {MAX_SALES_WINDOW_DAYS}."
        )

    scoped = resolve_warehouse(db, data) if WAREHOUSE_KEYS & set(data) else None

    warehouses = (
        [scoped]
        if scoped is not None
        else list(WarehouseRepository(db).list(limit=MAX_WAREHOUSES))
    )

    inventory = InventoryRepository(db)
    rollups: list[dict[str, Any]] = []
    all_records: list[Inventory] = []

    # One eager-loaded read per warehouse. Warehouses are few and capped, and
    # `list_for_warehouse` pulls each row's product and warehouse in the same
    # trip, so this is bounded rather than a row-by-row lazy load.
    for warehouse in warehouses:
        records = inventory.list_for_warehouse(warehouse.id)
        all_records.extend(records)
        rollups.append(_warehouse_rollup(warehouse, records))

    low_stock = [record for record in all_records if is_low_stock(record)]
    out_of_stock = [record for record in all_records if record.quantity == 0]

    return {
        "totals": {
            "products_in_catalogue": ProductRepository(db).count(),
            "warehouses": WarehouseRepository(db).count(),
            "warehouses_included": len(warehouses),
            "stock_lines": len(all_records),
            "total_units": sum(record.quantity for record in all_records),
            "stock_value": str(_valuation(all_records)),
            "low_stock_lines": len(low_stock),
            "out_of_stock_lines": len(out_of_stock),
        },
        "warehouses": {"count": len(rollups), "items": rollups},
        # Worst shortfall first: the rows an operator should act on soonest.
        "low_stock_items": _capped(
            low_stock,
            MAX_LOW_STOCK_ITEMS,
            key=lambda record: (-_shortfall(record), record.product.sku),
        ),
        "out_of_stock_items": _capped(
            out_of_stock,
            MAX_OUT_OF_STOCK_ITEMS,
            key=lambda record: record.product.sku,
        ),
        "sales_metrics": _recent_sales(db, window_days, scoped),
        "suppliers": _supplier_overview(db),
        "scope": scoped.code if scoped is not None else "all warehouses",
    }
