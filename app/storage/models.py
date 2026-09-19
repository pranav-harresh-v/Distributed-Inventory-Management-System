"""ORM schema for one inventory node's local SQLite database.

Conventions used throughout:

* Money is ``Numeric(12, 2)`` wrapped in :class:`Money` so values move in
  and out of the database as :class:`~decimal.Decimal`. Floats are never
  accepted for stored currency.
* Timestamps are timezone-aware UTC. SQLite has no native timezone type,
  so values are written as UTC and read back naive; always compare against
  :func:`utc_now`.
* Invariants that must hold regardless of which layer writes the row
  (quantities, prices, uniqueness) are enforced by named database
  constraints, not only in Python.
* Child collections load lazily (``lazy="select"``) so that fetching one
  product never drags its whole sales history into memory. The one
  exception is ``Order.items``, which is bounded and always needed with
  its order.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from app.storage.database import Base

__all__ = [
    "ADMIN_ROLE",
    "MANAGER_ROLE",
    "ORDER_STATUSES",
    "RESTOCK_STATUSES",
    "STAFF_ROLE",
    "USER_ROLES",
    "Inventory",
    "Money",
    "Order",
    "OrderItem",
    "Product",
    "RestockOrder",
    "SalesHistory",
    "Session",
    "Supplier",
    "SupplierProduct",
    "User",
    "Warehouse",
    "utc_now",
]

# --------------------------------------------------------------------------
# Vocabularies
#
# Kept as Python constants rather than database CHECK constraints: the roles
# and statuses below are the *initial* set and are expected to grow, and a
# schema migration is a poor reason to add an order status.
# --------------------------------------------------------------------------
ADMIN_ROLE: Final[str] = "admin"
MANAGER_ROLE: Final[str] = "manager"
STAFF_ROLE: Final[str] = "staff"
USER_ROLES: Final[frozenset[str]] = frozenset({ADMIN_ROLE, MANAGER_ROLE, STAFF_ROLE})

ORDER_PENDING: Final[str] = "PENDING"
ORDER_COMPLETED: Final[str] = "COMPLETED"
ORDER_CANCELLED: Final[str] = "CANCELLED"
ORDER_FAILED: Final[str] = "FAILED"
ORDER_STATUSES: Final[frozenset[str]] = frozenset(
    {ORDER_PENDING, ORDER_COMPLETED, ORDER_CANCELLED, ORDER_FAILED}
)

RESTOCK_PENDING: Final[str] = "PENDING"
RESTOCK_ORDERED: Final[str] = "ORDERED"
RESTOCK_RECEIVED: Final[str] = "RECEIVED"
RESTOCK_CANCELLED: Final[str] = "CANCELLED"
RESTOCK_STATUSES: Final[frozenset[str]] = frozenset(
    {RESTOCK_PENDING, RESTOCK_ORDERED, RESTOCK_RECEIVED, RESTOCK_CANCELLED}
)

CURRENCY_EXPONENT: Final[Decimal] = Decimal("0.01")
ZERO_MONEY: Final[Decimal] = Decimal("0.00")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Money(TypeDecorator[Decimal]):
    """``Numeric(12, 2)`` that always round-trips exact 2dp Decimals.

    Rejecting :class:`float` on the way in is deliberate: binary floats
    cannot represent currency exactly, and a silent rounding error in an
    order total is far harder to find later than a TypeError here.
    """

    impl = Numeric(12, 2)
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, float):
            raise TypeError(
                "Monetary values must be Decimal, int or str -- not float; "
                f"got {value!r}"
            )
        return Decimal(value).quantize(CURRENCY_EXPONENT, rounding=ROUND_HALF_UP)

    def process_result_value(self, value: Any, dialect: Any) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value)).quantize(CURRENCY_EXPONENT)


def _utc_column(**kwargs: Any) -> Any:
    """Timezone-aware UTC ``DateTime`` column."""
    return mapped_column(DateTime(timezone=True), **kwargs)


class TimestampMixin:
    """``created_at`` / ``updated_at`` pair shared by the master tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


# --------------------------------------------------------------------------
# Identity and sessions
# --------------------------------------------------------------------------
class User(TimestampMixin, Base):
    """An operator of the system. Only the password *hash* is ever stored."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default=STAFF_ROLE, server_default=text("'staff'")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("1")
    )

    sessions: Mapped[list[Session]] = relationship(
        back_populates="user",
        lazy="select",
        # Sessions are ephemeral auth artefacts with no business value of
        # their own, so they follow their user rather than blocking a delete.
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    orders: Mapped[list[Order]] = relationship(back_populates="user", lazy="select")


class Session(Base):
    """An issued login session. Stores a token *hash*, never a raw token."""

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE", name="fk_sessions_user_id"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    created_at: Mapped[datetime] = _utc_column(nullable=False, default=utc_now)
    expires_at: Mapped[datetime] = _utc_column(nullable=False)
    revoked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )

    user: Mapped[User] = relationship(back_populates="sessions")


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
class Product(TimestampMixin, Base):
    """A sellable item, identified by its SKU."""

    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("unit_price >= 0", name="ck_products_unit_price_non_negative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sku: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("1")
    )

    inventory_records: Mapped[list[Inventory]] = relationship(
        back_populates="product", lazy="select"
    )
    supplier_links: Mapped[list[SupplierProduct]] = relationship(
        back_populates="product", lazy="select"
    )


class Warehouse(TimestampMixin, Base):
    """A stock-holding location. One node owns one warehouse in Milestone 2."""

    __tablename__ = "warehouses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(
        String(32), nullable=False, unique=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("1")
    )

    inventory_records: Mapped[list[Inventory]] = relationship(
        back_populates="warehouse", lazy="select"
    )
    orders: Mapped[list[Order]] = relationship(
        back_populates="warehouse", lazy="select"
    )


class Inventory(Base):
    """Stock level of one product in one warehouse.

    ``quantity >= 0`` is the system's central safety invariant: even a buggy
    caller cannot drive stock negative, because SQLite rejects the row.
    """

    __tablename__ = "inventory"
    __table_args__ = (
        UniqueConstraint(
            "warehouse_id", "product_id", name="uq_inventory_warehouse_product"
        ),
        CheckConstraint("quantity >= 0", name="ck_inventory_quantity_non_negative"),
        CheckConstraint(
            "reorder_point >= 0", name="ck_inventory_reorder_point_non_negative"
        ),
        CheckConstraint(
            "safety_stock >= 0", name="ck_inventory_safety_stock_non_negative"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # warehouse_id is also the leading column of the unique index above; the
    # explicit index keeps warehouse-wide scans fast if that key ever changes.
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", name="fk_inventory_warehouse_id"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", name="fk_inventory_product_id"),
        nullable=False,
        index=True,
    )
    quantity: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    reorder_point: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    safety_stock: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    updated_at: Mapped[datetime] = _utc_column(
        nullable=False, default=utc_now, onupdate=utc_now
    )

    warehouse: Mapped[Warehouse] = relationship(back_populates="inventory_records")
    product: Mapped[Product] = relationship(back_populates="inventory_records")


# --------------------------------------------------------------------------
# Suppliers
# --------------------------------------------------------------------------
class Supplier(TimestampMixin, Base):
    """An external party that can replenish stock."""

    __tablename__ = "suppliers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("1")
    )

    product_links: Mapped[list[SupplierProduct]] = relationship(
        back_populates="supplier", lazy="select"
    )


class SupplierProduct(TimestampMixin, Base):
    """Commercial terms offered by one supplier for one product."""

    __tablename__ = "supplier_products"
    __table_args__ = (
        UniqueConstraint(
            "supplier_id", "product_id", name="uq_supplier_products_supplier_product"
        ),
        CheckConstraint(
            "unit_cost >= 0", name="ck_supplier_products_unit_cost_non_negative"
        ),
        CheckConstraint(
            "lead_time_days >= 0",
            name="ck_supplier_products_lead_time_days_non_negative",
        ),
        CheckConstraint(
            "minimum_order_qty > 0",
            name="ck_supplier_products_minimum_order_qty_positive",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    supplier_id: Mapped[int] = mapped_column(
        ForeignKey("suppliers.id", name="fk_supplier_products_supplier_id"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", name="fk_supplier_products_product_id"),
        nullable=False,
        index=True,
    )
    supplier_sku: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit_cost: Mapped[Decimal] = mapped_column(Money, nullable=False)
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    minimum_order_qty: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    is_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )

    supplier: Mapped[Supplier] = relationship(back_populates="product_links")
    product: Mapped[Product] = relationship(back_populates="supplier_links")


# --------------------------------------------------------------------------
# Orders
# --------------------------------------------------------------------------
class Order(Base):
    """A customer order raised against one warehouse.

    Only the schema exists in Milestone 1; the atomic placement workflow is
    implemented in the domain layer later.
    """

    __tablename__ = "orders"
    __table_args__ = (
        CheckConstraint(
            "total_amount >= 0", name="ck_orders_total_amount_non_negative"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_number: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", name="fk_orders_user_id"), nullable=False, index=True
    )
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", name="fk_orders_warehouse_id"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ORDER_PENDING,
        server_default=text("'PENDING'"),
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Money, nullable=False, default=ZERO_MONEY, server_default=text("0")
    )
    created_at: Mapped[datetime] = _utc_column(nullable=False, default=utc_now)
    completed_at: Mapped[datetime | None] = _utc_column(nullable=True)

    user: Mapped[User] = relationship(back_populates="orders")
    warehouse: Mapped[Warehouse] = relationship(back_populates="orders")
    # An order line has no meaning without its order, so it is the one place
    # a delete cascade is correct: removing an order must not orphan items.
    items: Mapped[list[OrderItem]] = relationship(
        back_populates="order",
        lazy="selectin",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class OrderItem(Base):
    """One product line of an order."""

    __tablename__ = "order_items"
    __table_args__ = (
        UniqueConstraint("order_id", "product_id", name="uq_order_items_order_product"),
        CheckConstraint("quantity > 0", name="ck_order_items_quantity_positive"),
        CheckConstraint(
            "unit_price >= 0", name="ck_order_items_unit_price_non_negative"
        ),
        CheckConstraint("subtotal >= 0", name="ck_order_items_subtotal_non_negative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE", name="fk_order_items_order_id"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", name="fk_order_items_product_id"),
        nullable=False,
        index=True,
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    subtotal: Mapped[Decimal] = mapped_column(Money, nullable=False)

    order: Mapped[Order] = relationship(back_populates="items")
    product: Mapped[Product] = relationship(lazy="select")


# --------------------------------------------------------------------------
# Restocking and analytics
# --------------------------------------------------------------------------
class RestockOrder(Base):
    """One replenishment request for a single product, from one supplier."""

    __tablename__ = "restock_orders"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_restock_orders_quantity_positive"),
        CheckConstraint(
            "unit_cost >= 0", name="ck_restock_orders_unit_cost_non_negative"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    restock_number: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", name="fk_restock_orders_warehouse_id"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", name="fk_restock_orders_product_id"),
        nullable=False,
        index=True,
    )
    supplier_id: Mapped[int] = mapped_column(
        ForeignKey("suppliers.id", name="fk_restock_orders_supplier_id"),
        nullable=False,
        index=True,
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_cost: Mapped[Decimal] = mapped_column(Money, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RESTOCK_PENDING,
        server_default=text("'PENDING'"),
    )
    expected_date: Mapped[datetime | None] = _utc_column(nullable=True)
    created_at: Mapped[datetime] = _utc_column(nullable=False, default=utc_now)
    received_at: Mapped[datetime | None] = _utc_column(nullable=True)

    warehouse: Mapped[Warehouse] = relationship(lazy="select")
    product: Mapped[Product] = relationship(lazy="select")
    supplier: Mapped[Supplier] = relationship(lazy="select")


class SalesHistory(Base):
    """Append-only record of goods sold, used later for demand forecasting.

    Forecasting reads "sales of product P (in warehouse W) over the last N
    days", so the indexes lead with the entity and end with ``sold_at``;
    each also serves lookups on its leading column alone.
    """

    __tablename__ = "sales_history"
    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_sales_history_quantity_positive"),
        CheckConstraint(
            "unit_price >= 0", name="ck_sales_history_unit_price_non_negative"
        ),
        Index("ix_sales_history_product_sold_at", "product_id", "sold_at"),
        Index("ix_sales_history_warehouse_sold_at", "warehouse_id", "sold_at"),
        Index("ix_sales_history_sold_at", "sold_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(
        ForeignKey("orders.id", name="fk_sales_history_order_id"),
        nullable=False,
        index=True,
    )
    warehouse_id: Mapped[int] = mapped_column(
        ForeignKey("warehouses.id", name="fk_sales_history_warehouse_id"),
        nullable=False,
    )
    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id", name="fk_sales_history_product_id"), nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(Money, nullable=False)
    sold_at: Mapped[datetime] = _utc_column(nullable=False, default=utc_now)

    order: Mapped[Order] = relationship(lazy="select")
    warehouse: Mapped[Warehouse] = relationship(lazy="select")
    product: Mapped[Product] = relationship(lazy="select")
