"""Thin data-access helpers over the ORM models.

Scope note: these are *storage* helpers only -- lookups, inserts and
listings. Business rules (order placement, stock reservation, restocking,
authentication) belong in the domain layer and are added in later phases.

Transaction policy: repositories ``flush()`` but never ``commit()``. The
caller owns the transaction boundary (normally
:func:`app.storage.database.session_scope`), which is what lets a whole
business operation succeed or fail atomically. Nothing here catches
``SQLAlchemyError``: a constraint violation must reach the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Generic, TypeVar

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.storage.database import Base
from app.storage.models import (
    STAFF_ROLE,
    Inventory,
    Order,
    OrderItem,
    Product,
    SalesHistory,
    Supplier,
    SupplierProduct,
    User,
    Warehouse,
    ensure_utc,
    utc_now,
)
from app.storage.models import (
    Session as SessionRecord,
)

ModelT = TypeVar("ModelT", bound=Base)

__all__ = [
    "BaseRepository",
    "InventoryRepository",
    "OrderItemRepository",
    "OrderRepository",
    "ProductRepository",
    "SalesHistoryRepository",
    "SessionRepository",
    "SupplierProductRepository",
    "SupplierRepository",
    "UserRepository",
    "WarehouseRepository",
]


class BaseRepository(Generic[ModelT]):
    """Shared ``get``/``list``/``add`` behaviour for a single model."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity_id: int) -> ModelT | None:
        """Fetch one row by primary key, or ``None``."""
        return self.session.get(self.model, entity_id)

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[ModelT]:
        """List rows, optionally hiding deactivated ones."""
        stmt = select(self.model)
        if active_only:
            stmt = stmt.where(self.model.is_active.is_(True))
        stmt = stmt.order_by(self.model.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new row and flush so it receives its primary key.

        Flushing surfaces constraint violations here rather than at an
        unrelated commit further down the call stack.
        """
        self.session.add(entity)
        self.session.flush()
        return entity

    def save(self, entity: ModelT) -> ModelT:
        """Flush pending changes to an entity already tracked by the session.

        The counterpart to :meth:`add` for updates: the caller has mutated a
        loaded row, and this pushes it to the database now so a constraint
        violation surfaces here rather than at commit time. The caller still
        owns the transaction.
        """
        self.session.flush()
        return entity


class UserRepository(BaseRepository[User]):
    """Lookup helpers for user accounts."""

    model = User

    def get_by_username(self, username: str) -> User | None:
        return self.session.scalars(
            select(User).where(User.username == username)
        ).one_or_none()

    def get_by_email(self, email: str) -> User | None:
        return self.session.scalars(
            select(User).where(User.email == email)
        ).one_or_none()

    def create(
        self,
        *,
        username: str,
        email: str,
        password_hash: str,
        role: str = STAFF_ROLE,
        is_active: bool = True,
    ) -> User:
        """Insert a user. The caller supplies an already-hashed password."""
        return self.add(
            User(
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                is_active=is_active,
            )
        )


class SessionRepository(BaseRepository[SessionRecord]):
    """Login-session access.

    Only ever handles the *hash* of a session token; the raw token exists
    solely in the login response. ``sessions`` has no ``is_active`` column,
    so :meth:`list` drops the filter it would otherwise inherit.
    """

    model = SessionRecord

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[SessionRecord]:
        stmt = select(SessionRecord).order_by(SessionRecord.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def get_by_token_hash(self, token_hash: str) -> SessionRecord | None:
        """Look up a session by token hash.

        ``sessions.token_hash`` is unique and indexed, so this is a single
        index probe -- which is why session tokens are hashed with a fast
        deterministic digest rather than a salted one.
        """
        return self.session.scalars(
            select(SessionRecord).where(SessionRecord.token_hash == token_hash)
        ).one_or_none()

    def list_for_user(self, user_id: int) -> Sequence[SessionRecord]:
        return self.session.scalars(
            select(SessionRecord)
            .where(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.id)
        ).all()

    def create(
        self,
        *,
        user_id: int,
        token_hash: str,
        expires_at: datetime,
    ) -> SessionRecord:
        """Insert a new session. ``revoked`` defaults to False."""
        return self.add(
            SessionRecord(
                user_id=user_id,
                token_hash=token_hash,
                expires_at=expires_at,
            )
        )

    def revoke(self, record: SessionRecord) -> SessionRecord:
        """Mark one session revoked, leaving every other session alone."""
        record.revoked = True
        self.session.flush()
        return record


class ProductRepository(BaseRepository[Product]):
    """Catalogue access."""

    model = Product

    def get_by_sku(self, sku: str) -> Product | None:
        return self.session.scalars(
            select(Product).where(Product.sku == sku)
        ).one_or_none()

    def create(
        self,
        *,
        sku: str,
        name: str,
        unit_price: Decimal,
        description: str | None = None,
        is_active: bool = True,
    ) -> Product:
        return self.add(
            Product(
                sku=sku,
                name=name,
                unit_price=unit_price,
                description=description,
                is_active=is_active,
            )
        )


class WarehouseRepository(BaseRepository[Warehouse]):
    """Warehouse access."""

    model = Warehouse

    def get_by_code(self, code: str) -> Warehouse | None:
        return self.session.scalars(
            select(Warehouse).where(Warehouse.code == code)
        ).one_or_none()

    def create(
        self,
        *,
        code: str,
        name: str,
        location: str,
        is_active: bool = True,
    ) -> Warehouse:
        return self.add(
            Warehouse(code=code, name=name, location=location, is_active=is_active)
        )


class InventoryRepository(BaseRepository[Inventory]):
    """Stock-level access.

    ``Inventory`` has no ``is_active`` flag, so :meth:`list` is overridden to
    drop that filter rather than inherit a query that cannot run.
    """

    model = Inventory

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[Inventory]:
        stmt = select(Inventory).order_by(Inventory.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def get_by_warehouse_and_product(
        self, warehouse_id: int, product_id: int
    ) -> Inventory | None:
        """Fetch the single stock row for a (warehouse, product) pair.

        The product and warehouse are loaded eagerly so callers can
        serialize the row after the session closes without tripping a lazy
        load on a detached instance.
        """
        return self.session.scalars(
            select(Inventory)
            .where(
                Inventory.warehouse_id == warehouse_id,
                Inventory.product_id == product_id,
            )
            .options(selectinload(Inventory.product), selectinload(Inventory.warehouse))
        ).one_or_none()

    def list_for_warehouse(self, warehouse_id: int) -> Sequence[Inventory]:
        """Every stock row in one warehouse, ordered by product.

        ``selectinload`` pulls the related product and warehouse in one
        extra query rather than one per row, which matters as soon as a
        caller serializes the whole list.
        """
        return self.session.scalars(
            select(Inventory)
            .where(Inventory.warehouse_id == warehouse_id)
            .options(selectinload(Inventory.product), selectinload(Inventory.warehouse))
            .order_by(Inventory.product_id)
        ).all()

    def list_low_stock(self, warehouse_id: int | None = None) -> Sequence[Inventory]:
        """Rows at or below their reorder point.

        The comparison runs in SQL so the database does the filtering, and
        it compares two columns of the same row -- the threshold is per
        (warehouse, product), not a global constant.
        """
        stmt = (
            select(Inventory)
            .where(Inventory.quantity <= Inventory.reorder_point)
            .options(selectinload(Inventory.product), selectinload(Inventory.warehouse))
            .order_by(Inventory.warehouse_id, Inventory.product_id)
        )
        if warehouse_id is not None:
            stmt = stmt.where(Inventory.warehouse_id == warehouse_id)
        return self.session.scalars(stmt).all()

    def increment_quantity(
        self, *, warehouse_id: int, product_id: int, amount: int
    ) -> int:
        """Add ``amount`` to a stock row inside the database itself.

        Issues ``SET quantity = quantity + :amount`` rather than reading the
        value into Python, adding, and writing it back. The difference
        matters under concurrency: with read-modify-write, two callers can
        both read 10, both write 15, and one increment is silently lost.
        Letting the database compute the new value makes the update atomic
        against the row, so both increments land.

        Returns the number of rows updated -- zero means no such stock row.
        """
        result = self.session.execute(
            update(Inventory)
            .where(
                Inventory.warehouse_id == warehouse_id,
                Inventory.product_id == product_id,
            )
            .values(quantity=Inventory.quantity + amount, updated_at=utc_now())
            # Keeps any already-loaded Inventory object in this session from
            # reporting the pre-update quantity.
            .execution_options(synchronize_session="fetch")
        )
        return result.rowcount

    def decrement_quantity(
        self, *, warehouse_id: int, product_id: int, amount: int
    ) -> int:
        """Remove ``amount`` from a stock row, but only if it is there.

        The availability check lives in the WHERE clause rather than in
        Python::

            UPDATE inventory SET quantity = quantity - :amount
             WHERE warehouse_id = :w AND product_id = :p
               AND quantity >= :amount

        Checking first and writing afterwards leaves a window in which
        another transaction can consume the same units; here the test and
        the write are one statement, so the database decides. A return of
        0 means the update matched nothing: either the row does not exist
        or it held less than ``amount``. The caller distinguishes those.

        This is also why stock cannot go negative through this path -- the
        guard makes the decrement impossible, rather than relying on the
        CHECK constraint to reject it after the fact.
        """
        result = self.session.execute(
            update(Inventory)
            .where(
                Inventory.warehouse_id == warehouse_id,
                Inventory.product_id == product_id,
                Inventory.quantity >= amount,
            )
            .values(quantity=Inventory.quantity - amount, updated_at=utc_now())
            .execution_options(synchronize_session="fetch")
        )
        return result.rowcount

    def create(
        self,
        *,
        warehouse_id: int,
        product_id: int,
        quantity: int = 0,
        reorder_point: int = 0,
        safety_stock: int = 0,
    ) -> Inventory:
        return self.add(
            Inventory(
                warehouse_id=warehouse_id,
                product_id=product_id,
                quantity=quantity,
                reorder_point=reorder_point,
                safety_stock=safety_stock,
            )
        )


class SupplierRepository(BaseRepository[Supplier]):
    """Supplier access."""

    model = Supplier

    def get_by_name(self, name: str) -> Supplier | None:
        return self.session.scalars(
            select(Supplier).where(Supplier.name == name)
        ).one_or_none()

    def create(
        self,
        *,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        address: str | None = None,
        is_active: bool = True,
    ) -> Supplier:
        return self.add(
            Supplier(
                name=name,
                email=email,
                phone=phone,
                address=address,
                is_active=is_active,
            )
        )


class SupplierProductRepository(BaseRepository[SupplierProduct]):
    """Supplier/product terms.

    Like ``Inventory`` this is a link table without ``is_active``, so
    :meth:`list` ignores the ``active_only`` filter.
    """

    model = SupplierProduct

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[SupplierProduct]:
        stmt = select(SupplierProduct).order_by(SupplierProduct.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def get_by_supplier_and_product(
        self, supplier_id: int, product_id: int
    ) -> SupplierProduct | None:
        """Fetch the terms one supplier offers for one product.

        Supplier and product load eagerly so the row can be serialized
        after the session closes, as with ``InventoryRepository``.
        """
        return self.session.scalars(
            select(SupplierProduct)
            .where(
                SupplierProduct.supplier_id == supplier_id,
                SupplierProduct.product_id == product_id,
            )
            .options(
                selectinload(SupplierProduct.supplier),
                selectinload(SupplierProduct.product),
            )
        ).one_or_none()

    def list_for_product(self, product_id: int) -> Sequence[SupplierProduct]:
        """Every supplier offering one product, preferred and cheapest first."""
        return self.session.scalars(
            select(SupplierProduct)
            .where(SupplierProduct.product_id == product_id)
            .options(
                selectinload(SupplierProduct.supplier),
                selectinload(SupplierProduct.product),
            )
            .order_by(SupplierProduct.is_preferred.desc(), SupplierProduct.unit_cost)
        ).all()

    def create(
        self,
        *,
        supplier_id: int,
        product_id: int,
        unit_cost: Decimal,
        lead_time_days: int,
        supplier_sku: str | None = None,
        minimum_order_qty: int = 1,
        is_preferred: bool = False,
    ) -> SupplierProduct:
        return self.add(
            SupplierProduct(
                supplier_id=supplier_id,
                product_id=product_id,
                unit_cost=unit_cost,
                lead_time_days=lead_time_days,
                supplier_sku=supplier_sku,
                minimum_order_qty=minimum_order_qty,
                is_preferred=is_preferred,
            )
        )


class OrderRepository(BaseRepository[Order]):
    """Order headers.

    Like every repository here it flushes but never commits: order
    placement spans several tables and the caller's transaction is what
    makes the whole thing atomic.
    """

    model = Order

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[Order]:
        stmt = select(Order).order_by(Order.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def get_by_order_number(self, order_number: str) -> Order | None:
        return self.session.scalars(
            select(Order)
            .where(Order.order_number == order_number)
            .options(selectinload(Order.items), selectinload(Order.warehouse))
        ).one_or_none()

    def create(
        self,
        *,
        order_number: str,
        user_id: int,
        warehouse_id: int,
        status: str,
        total_amount: Decimal,
    ) -> Order:
        return self.add(
            Order(
                order_number=order_number,
                user_id=user_id,
                warehouse_id=warehouse_id,
                status=status,
                total_amount=total_amount,
            )
        )


class OrderItemRepository(BaseRepository[OrderItem]):
    """Order lines. One row per (order, product) -- the schema enforces it."""

    model = OrderItem

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[OrderItem]:
        stmt = select(OrderItem).order_by(OrderItem.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def list_for_order(self, order_id: int) -> Sequence[OrderItem]:
        return self.session.scalars(
            select(OrderItem)
            .where(OrderItem.order_id == order_id)
            .options(selectinload(OrderItem.product))
            .order_by(OrderItem.product_id)
        ).all()

    def create(
        self,
        *,
        order_id: int,
        product_id: int,
        quantity: int,
        unit_price: Decimal,
        subtotal: Decimal,
    ) -> OrderItem:
        return self.add(
            OrderItem(
                order_id=order_id,
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price,
                subtotal=subtotal,
            )
        )


class SalesHistoryRepository(BaseRepository[SalesHistory]):
    """Append-only record of goods sold, for later demand forecasting.

    Every query here orders by ``(sold_at, id)``. Chronological order is
    what a forecasting caller wants, and the primary key breaks ties so
    rows written within one order -- which share a single timestamp --
    still come back in a stable, repeatable sequence.
    """

    model = SalesHistory

    @staticmethod
    def _as_stored(moment: datetime) -> datetime:
        """Normalise a cutoff to the naive-UTC form SQLite stores.

        SQLAlchemy's SQLite DATETIME writes wall-clock digits and drops the
        offset, so comparing a non-UTC aware value directly would skew the
        window by that offset. Converting to UTC first makes the comparison
        correct whatever timezone the caller passes.
        """
        return ensure_utc(moment).replace(tzinfo=None)

    def list(
        self,
        *,
        active_only: bool = False,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Sequence[SalesHistory]:
        stmt = select(SalesHistory).order_by(SalesHistory.sold_at, SalesHistory.id)
        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)
        return self.session.scalars(stmt).all()

    def list_for_order(self, order_id: int) -> Sequence[SalesHistory]:
        return self.session.scalars(
            select(SalesHistory)
            .where(SalesHistory.order_id == order_id)
            .order_by(SalesHistory.sold_at, SalesHistory.id)
        ).all()

    def list_for_product(
        self,
        product_id: int,
        *,
        warehouse_id: int | None = None,
        since: datetime | None = None,
    ) -> Sequence[SalesHistory]:
        """Sales of one product, optionally scoped by warehouse and window.

        This is the shape the demand-forecasting layer needs ("units of
        product P sold in warehouse W over the last N days"), and it is
        what the schema's ``(product_id, sold_at)`` and
        ``(warehouse_id, sold_at)`` indexes were created to serve.
        """
        stmt = select(SalesHistory).where(SalesHistory.product_id == product_id)
        if warehouse_id is not None:
            stmt = stmt.where(SalesHistory.warehouse_id == warehouse_id)
        if since is not None:
            stmt = stmt.where(SalesHistory.sold_at >= self._as_stored(since))
        return self.session.scalars(
            stmt.order_by(SalesHistory.sold_at, SalesHistory.id)
        ).all()

    def create(
        self,
        *,
        order_id: int,
        warehouse_id: int,
        product_id: int,
        quantity: int,
        unit_price: Decimal,
        sold_at: datetime,
    ) -> SalesHistory:
        """Record one sold line.

        ``sold_at`` is supplied by the server, never by a client.
        """
        return self.add(
            SalesHistory(
                order_id=order_id,
                warehouse_id=warehouse_id,
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price,
                sold_at=sold_at,
            )
        )
