"""Atomic order placement.

This is the correctness-critical path of the whole system, and the one the
project's stated invariants are about: stock must never go negative, order
placement must be atomic, and a failed order must not partially modify
inventory.

How those are achieved, in one place:

**Transaction boundary.** :func:`place_order` never commits. It runs inside
the caller's :func:`~app.storage.database.session_scope`, which issues a
single commit when the function returns cleanly and a rollback if anything
raises. Repositories flush only. So an order either lands whole -- header,
lines, stock decrements and sales rows -- or leaves no trace at all.

**Stock decrement.** Every decrement is a guarded conditional UPDATE
(``... WHERE quantity >= :amount``) performed by
:meth:`~app.storage.repositories.InventoryRepository.decrement_quantity`.
The availability test and the write are a single statement, so two
concurrent orders cannot both pass a check and then both write. A row count
of zero means the units were not there, and the order is rejected.

**Determinism.** Lines are consolidated and processed in a fixed order
(ascending product id) so the same request always touches rows in the same
sequence. That matters for Milestone 2, where this operation becomes a
replicated Raft command that every node must apply identically.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession

from app.domain.inventory import (
    CatalogError,
    NotFoundError,
    ValidationError,
    coerce_int,
    isoformat,
    reject_unknown_fields,
    require_mapping,
    resolve_product,
    resolve_warehouse,
    translate_integrity_error,
)
from app.storage.models import (
    CURRENCY_EXPONENT,
    ORDER_COMPLETED,
    Order,
    Product,
    User,
    utc_now,
)
from app.storage.repositories import (
    InventoryRepository,
    OrderItemRepository,
    OrderRepository,
    SalesHistoryRepository,
)

__all__ = [
    "InsufficientStockError",
    "OrderError",
    "place_order",
    "serialize_order",
]

#: Fields accepted on an order request. `sold_at`, `status`, `total_amount`
#: and `order_number` are deliberately absent: those are the server's to
#: decide, and a client must not be able to backdate a sale or declare its
#: own total.
ORDER_REQUEST_FIELDS: Final[frozenset[str]] = frozenset(
    {"warehouse_id", "warehouse_code", "items"}
)

ORDER_LINE_FIELDS: Final[frozenset[str]] = frozenset({"product_id", "sku", "quantity"})

MAX_ORDER_LINES: Final[int] = 100

#: ``ORD-`` plus 12 hex characters. Generated server-side so a client
#: cannot choose or collide with an order number.
_ORDER_NUMBER_PREFIX: Final[str] = "ORD-"


class OrderError(CatalogError):
    """Base class for order failures, so they map through the same table."""


class InsufficientStockError(OrderError):
    """A requested quantity exceeds what the warehouse holds.

    Carries the product and the numbers involved so the API layer can
    report *which* line failed without inventing a message format.
    """

    def __init__(self, sku: str, requested: int, available: int) -> None:
        super().__init__(
            f"Insufficient stock for '{sku}': requested {requested}, "
            f"available {available}."
        )
        self.sku = sku
        self.requested = requested
        self.available = available


@dataclass(frozen=True)
class _OrderLine:
    """One consolidated request line, resolved against the catalogue."""

    product: Product
    quantity: int


def _consolidate_lines(db: DbSession, raw_items: Any) -> list[_OrderLine]:
    """Validate the requested lines and merge duplicate products.

    Two lines for the same product are summed rather than rejected or
    written twice: ``order_items`` is unique per (order, product), so a
    duplicate would otherwise violate the schema, and summing is what a
    caller adding the same item twice means.

    The result is sorted by product id, which makes the row-touch order
    deterministic for a given request.
    """
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise ValidationError("'items' must be a list of order lines.")
    if not raw_items:
        raise ValidationError("'items' must contain at least one line.")
    if len(raw_items) > MAX_ORDER_LINES:
        raise ValidationError(f"An order may not exceed {MAX_ORDER_LINES} lines.")

    totals: dict[int, int] = {}
    products: dict[int, Product] = {}

    for index, raw in enumerate(raw_items):
        line = require_mapping(raw, f"items[{index}]")
        reject_unknown_fields(line, ORDER_LINE_FIELDS)
        if "quantity" not in line:
            raise ValidationError(f"items[{index}]: 'quantity' is required.")

        # minimum=1 enforces the order_items CHECK (quantity > 0) up front,
        # so zero and negative requests never reach the database.
        quantity = coerce_int(line["quantity"], f"items[{index}].quantity", minimum=1)
        product = resolve_product(db, line)

        products[product.id] = product
        totals[product.id] = totals.get(product.id, 0) + quantity

    for product_id, quantity in totals.items():
        if quantity > MAX_ORDER_LINES * 1_000_000:  # pragma: no cover - guard
            raise ValidationError("Requested quantity is implausibly large.")
        totals[product_id] = quantity

    return [
        _OrderLine(product=products[product_id], quantity=totals[product_id])
        for product_id in sorted(totals)
    ]


def _generate_order_number() -> str:
    return f"{_ORDER_NUMBER_PREFIX}{uuid.uuid4().hex[:12].upper()}"


def serialize_order(order: Order) -> dict[str, Any]:
    """Project an Order onto protobuf-Struct-safe primitives."""
    return {
        "id": order.id,
        "order_number": order.order_number,
        "user_id": order.user_id,
        "warehouse_id": order.warehouse_id,
        "status": order.status,
        "total_amount": str(order.total_amount),
        "created_at": isoformat(order.created_at),
        "completed_at": isoformat(order.completed_at),
        "items": [
            {
                "product_id": item.product_id,
                "quantity": item.quantity,
                "unit_price": str(item.unit_price),
                "subtotal": str(item.subtotal),
            }
            for item in order.items
        ],
    }


def place_order(db: DbSession, actor: User, payload: Any) -> Order:
    """Place one order atomically.

    Caller contract: run inside a transaction scope that commits once on
    success and rolls back on any exception. This function flushes so that
    constraint violations surface here, but never commits.
    """
    data = require_mapping(payload, "payload")
    reject_unknown_fields(data, ORDER_REQUEST_FIELDS)
    if "items" not in data:
        raise ValidationError("'items' is required.")

    # 1. Resolve and validate everything before touching any stock, so an
    #    obviously bad request never starts a partial mutation.
    warehouse = resolve_warehouse(db, data)
    lines = _consolidate_lines(db, data["items"])

    inventory_repository = InventoryRepository(db)
    order_repository = OrderRepository(db)
    item_repository = OrderItemRepository(db)
    sales_repository = SalesHistoryRepository(db)

    # 2. Create the header. If any line below fails, the caller's rollback
    #    removes this row too -- which is exactly why no partial order can
    #    survive.
    order = order_repository.create(
        order_number=_generate_order_number(),
        user_id=actor.id,
        warehouse_id=warehouse.id,
        status=ORDER_COMPLETED,
        total_amount=Decimal("0.00"),
    )

    sold_at = utc_now()  # one server timestamp for the whole order
    total = Decimal("0.00")

    for line in lines:
        product = line.product

        # 3. The stock row must exist; "no record" is a different failure
        #    from "not enough", and the client is told which.
        record = inventory_repository.get_by_warehouse_and_product(
            warehouse.id, product.id
        )
        if record is None:
            raise NotFoundError(
                f"No stock record for product '{product.sku}' in warehouse "
                f"'{warehouse.code}'."
            )

        # 4. Guarded decrement. Zero rows updated means the units were not
        #    available at the moment the statement ran.
        updated = inventory_repository.decrement_quantity(
            warehouse_id=warehouse.id, product_id=product.id, amount=line.quantity
        )
        if updated == 0:
            available = inventory_repository.get_by_warehouse_and_product(
                warehouse.id, product.id
            ).quantity
            raise InsufficientStockError(
                sku=product.sku, requested=line.quantity, available=available
            )

        unit_price = product.unit_price
        subtotal = (unit_price * line.quantity).quantize(CURRENCY_EXPONENT)
        total += subtotal

        item_repository.create(
            order_id=order.id,
            product_id=product.id,
            quantity=line.quantity,
            unit_price=unit_price,
            subtotal=subtotal,
        )
        sales_repository.create(
            order_id=order.id,
            warehouse_id=warehouse.id,
            product_id=product.id,
            quantity=line.quantity,
            unit_price=unit_price,
            sold_at=sold_at,
        )

    # 5. Finalise the header now that every line succeeded.
    order.total_amount = total
    order.completed_at = sold_at
    try:
        order_repository.save(order)
    except IntegrityError as exc:
        raise translate_integrity_error(
            exc, entity="order", conflict=f"number '{order.order_number}'"
        ) from exc

    return order
