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

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.storage.database import Base
from app.storage.models import (
    STAFF_ROLE,
    Inventory,
    Product,
    Supplier,
    SupplierProduct,
    User,
    Warehouse,
)
from app.storage.models import (
    Session as SessionRecord,
)

ModelT = TypeVar("ModelT", bound=Base)

__all__ = [
    "BaseRepository",
    "InventoryRepository",
    "ProductRepository",
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
        """Fetch the single stock row for a (warehouse, product) pair."""
        return self.session.scalars(
            select(Inventory).where(
                Inventory.warehouse_id == warehouse_id,
                Inventory.product_id == product_id,
            )
        ).one_or_none()

    def list_for_warehouse(self, warehouse_id: int) -> Sequence[Inventory]:
        return self.session.scalars(
            select(Inventory)
            .where(Inventory.warehouse_id == warehouse_id)
            .order_by(Inventory.product_id)
        ).all()

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
        return self.session.scalars(
            select(SupplierProduct).where(
                SupplierProduct.supplier_id == supplier_id,
                SupplierProduct.product_id == product_id,
            )
        ).one_or_none()

    def list_for_product(self, product_id: int) -> Sequence[SupplierProduct]:
        return self.session.scalars(
            select(SupplierProduct)
            .where(SupplierProduct.product_id == product_id)
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
