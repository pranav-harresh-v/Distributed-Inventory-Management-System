"""Schema-level tests: every invariant is enforced by the database itself.

These tests deliberately bypass any future domain-layer validation. They
assert that SQLite rejects bad rows on its own, so a bug elsewhere in the
system can never leave the node in an impossible state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session as OrmSession

from app.storage.models import (
    Inventory,
    Order,
    OrderItem,
    Product,
    RestockOrder,
    SalesHistory,
    Session,
    Supplier,
    SupplierProduct,
    User,
    Warehouse,
    utc_now,
)


def expect_integrity_error(session: OrmSession) -> None:
    """Flush and require the database to reject the pending rows."""
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def make_supplier(session: OrmSession, name: str = "Acme") -> Supplier:
    supplier = Supplier(name=name)
    session.add(supplier)
    session.flush()
    return supplier


def make_order(session: OrmSession, user: User, warehouse: Warehouse) -> Order:
    order = Order(
        order_number=f"ORD-{user.id}-{warehouse.id}",
        user_id=user.id,
        warehouse_id=warehouse.id,
    )
    session.add(order)
    session.flush()
    return order


# --- uniqueness ------------------------------------------------------------
def test_duplicate_username_rejected(session: OrmSession, user: User) -> None:
    """Requirement 3."""
    session.add(User(username="alice", email="other@example.com", password_hash="hash"))
    expect_integrity_error(session)


def test_duplicate_email_rejected(session: OrmSession, user: User) -> None:
    session.add(User(username="bob", email="alice@example.com", password_hash="hash"))
    expect_integrity_error(session)


def test_duplicate_product_sku_rejected(session: OrmSession, product: Product) -> None:
    """Requirement 4."""
    session.add(Product(sku="SKU-1", name="Other", unit_price=Decimal("1.00")))
    expect_integrity_error(session)


def test_duplicate_warehouse_code_rejected(
    session: OrmSession, warehouse: Warehouse
) -> None:
    """Requirement 5."""
    session.add(Warehouse(code="WH-1", name="Other", location="Chennai"))
    expect_integrity_error(session)


def test_duplicate_inventory_pair_rejected(
    session: OrmSession, warehouse: Warehouse, product: Product
) -> None:
    """Requirement 6: one stock row per (warehouse, product)."""
    session.add(Inventory(warehouse_id=warehouse.id, product_id=product.id, quantity=5))
    session.flush()
    session.add(Inventory(warehouse_id=warehouse.id, product_id=product.id, quantity=7))
    expect_integrity_error(session)


def test_duplicate_supplier_product_rejected(
    session: OrmSession, product: Product
) -> None:
    """Requirement 10."""
    supplier = make_supplier(session)
    for _ in range(2):
        session.add(
            SupplierProduct(
                supplier_id=supplier.id,
                product_id=product.id,
                unit_cost=Decimal("3.00"),
                lead_time_days=4,
            )
        )
    expect_integrity_error(session)


def test_duplicate_session_token_hash_rejected(session: OrmSession, user: User) -> None:
    expires = utc_now() + timedelta(hours=1)
    for _ in range(2):
        session.add(
            Session(user_id=user.id, token_hash="same-hash", expires_at=expires)
        )
    expect_integrity_error(session)


def test_duplicate_order_number_rejected(
    session: OrmSession, user: User, warehouse: Warehouse
) -> None:
    for _ in range(2):
        session.add(
            Order(
                order_number="ORD-DUP",
                user_id=user.id,
                warehouse_id=warehouse.id,
            )
        )
    expect_integrity_error(session)


# --- inventory CHECK constraints ------------------------------------------
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quantity", -1),  # requirement 7
        ("reorder_point", -1),  # requirement 8
        ("safety_stock", -1),  # requirement 9
    ],
)
def test_negative_inventory_values_rejected(
    session: OrmSession, warehouse: Warehouse, product: Product, field: str, value: int
) -> None:
    session.add(
        Inventory(
            warehouse_id=warehouse.id,
            product_id=product.id,
            **{field: value},
        )
    )
    expect_integrity_error(session)


def test_zero_inventory_quantity_allowed(
    session: OrmSession, warehouse: Warehouse, product: Product
) -> None:
    """Stock may reach zero -- only negative stock is impossible."""
    session.add(Inventory(warehouse_id=warehouse.id, product_id=product.id, quantity=0))
    session.flush()
    record = session.scalars(select(Inventory)).one()
    assert record.quantity == 0


# --- order / restock / sales CHECK constraints ----------------------------
@pytest.mark.parametrize("quantity", [0, -3])
def test_order_item_quantity_must_be_positive(
    session: OrmSession,
    user: User,
    warehouse: Warehouse,
    product: Product,
    quantity: int,
) -> None:
    """Requirement 11."""
    order = make_order(session, user, warehouse)
    session.add(
        OrderItem(
            order_id=order.id,
            product_id=product.id,
            quantity=quantity,
            unit_price=Decimal("9.99"),
            subtotal=Decimal("0.00"),
        )
    )
    expect_integrity_error(session)


@pytest.mark.parametrize("quantity", [0, -5])
def test_restock_quantity_must_be_positive(
    session: OrmSession, warehouse: Warehouse, product: Product, quantity: int
) -> None:
    """Requirement 12."""
    supplier = make_supplier(session, name="Restock Co")
    session.add(
        RestockOrder(
            restock_number="RST-1",
            warehouse_id=warehouse.id,
            product_id=product.id,
            supplier_id=supplier.id,
            quantity=quantity,
            unit_cost=Decimal("5.00"),
        )
    )
    expect_integrity_error(session)


@pytest.mark.parametrize("quantity", [0, -2])
def test_sales_history_quantity_must_be_positive(
    session: OrmSession,
    user: User,
    warehouse: Warehouse,
    product: Product,
    quantity: int,
) -> None:
    """Requirement 13."""
    order = make_order(session, user, warehouse)
    session.add(
        SalesHistory(
            order_id=order.id,
            warehouse_id=warehouse.id,
            product_id=product.id,
            quantity=quantity,
            unit_price=Decimal("9.99"),
            sold_at=utc_now(),
        )
    )
    expect_integrity_error(session)


def test_negative_money_values_rejected(session: OrmSession) -> None:
    session.add(Product(sku="SKU-NEG", name="Bad", unit_price=Decimal("-0.01")))
    expect_integrity_error(session)


def test_negative_order_total_rejected(
    session: OrmSession, user: User, warehouse: Warehouse
) -> None:
    session.add(
        Order(
            order_number="ORD-NEG",
            user_id=user.id,
            warehouse_id=warehouse.id,
            total_amount=Decimal("-1.00"),
        )
    )
    expect_integrity_error(session)


def test_supplier_product_minimum_order_qty_must_be_positive(
    session: OrmSession, product: Product
) -> None:
    supplier = make_supplier(session, name="Zero MOQ")
    session.add(
        SupplierProduct(
            supplier_id=supplier.id,
            product_id=product.id,
            unit_cost=Decimal("1.00"),
            lead_time_days=1,
            minimum_order_qty=0,
        )
    )
    expect_integrity_error(session)


# --- foreign keys ----------------------------------------------------------
def test_inventory_rejects_unknown_foreign_keys(
    session: OrmSession, warehouse: Warehouse
) -> None:
    """Requirement 14."""
    session.add(Inventory(warehouse_id=warehouse.id, product_id=9999, quantity=1))
    expect_integrity_error(session)


def test_session_rejects_unknown_user(session: OrmSession) -> None:
    session.add(
        Session(
            user_id=4242,
            token_hash="orphan-token",
            expires_at=utc_now() + timedelta(hours=1),
        )
    )
    expect_integrity_error(session)


def test_order_rejects_unknown_warehouse(session: OrmSession, user: User) -> None:
    session.add(Order(order_number="ORD-FK", user_id=user.id, warehouse_id=777))
    expect_integrity_error(session)


# --- relationships and cascade --------------------------------------------
def test_relationships_navigate_both_ways(
    session: OrmSession, warehouse: Warehouse, product: Product
) -> None:
    record = Inventory(warehouse_id=warehouse.id, product_id=product.id, quantity=12)
    session.add(record)
    session.flush()

    assert record.warehouse.code == "WH-1"
    assert record.product.sku == "SKU-1"
    assert record in warehouse.inventory_records
    assert record in product.inventory_records


def test_deleting_an_order_removes_its_items(
    session: OrmSession, user: User, warehouse: Warehouse, product: Product
) -> None:
    """Order items must never be orphaned when their order goes away."""
    order = make_order(session, user, warehouse)
    order.items.append(
        OrderItem(
            product_id=product.id,
            quantity=2,
            unit_price=Decimal("9.99"),
            subtotal=Decimal("19.98"),
        )
    )
    session.flush()
    assert session.scalars(select(OrderItem)).all() != []

    session.delete(order)
    session.flush()
    assert session.scalars(select(OrderItem)).all() == []


def test_user_sessions_relationship(session: OrmSession, user: User) -> None:
    login = Session(
        user_id=user.id,
        token_hash="hashed-token",
        expires_at=utc_now() + timedelta(hours=8),
    )
    session.add(login)
    session.flush()

    assert login.user.username == "alice"
    assert login in user.sessions


# --- column semantics ------------------------------------------------------
def test_defaults_are_applied(session: OrmSession, user: User) -> None:
    assert user.role == "staff"
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None


def test_order_defaults(session: OrmSession, user: User, warehouse: Warehouse) -> None:
    order = make_order(session, user, warehouse)
    assert order.status == "PENDING"
    assert order.total_amount == Decimal("0.00")
    assert order.completed_at is None


def test_money_round_trips_as_decimal(session: OrmSession) -> None:
    session.add(Product(sku="SKU-M", name="Money", unit_price=Decimal("1234.56")))
    session.commit()
    session.expire_all()

    stored = session.scalars(select(Product).where(Product.sku == "SKU-M")).one()
    assert isinstance(stored.unit_price, Decimal)
    assert stored.unit_price == Decimal("1234.56")


def test_float_money_is_rejected(session: OrmSession) -> None:
    """Currency must never be stored as a binary float."""
    session.add(Product(sku="SKU-F", name="Floaty", unit_price=1234.56))  # type: ignore[arg-type]
    with pytest.raises(StatementError):
        session.flush()
    session.rollback()


def test_timestamps_are_utc(session: OrmSession, user: User) -> None:
    """``utc_now`` is timezone-aware; stored values track UTC."""
    assert utc_now().tzinfo is timezone.utc
    created = user.created_at
    reference = datetime.now(timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    assert abs((reference - created).total_seconds()) < 60


def test_password_hash_is_not_plaintext(session: OrmSession) -> None:
    """The schema has no plaintext password column at all."""
    columns = {column.name for column in User.__table__.columns}
    assert "password_hash" in columns
    assert not {"password", "plaintext_password"} & columns


def test_sessions_store_only_token_hashes() -> None:
    columns = {column.name for column in Session.__table__.columns}
    assert "token_hash" in columns
    assert "token" not in columns
