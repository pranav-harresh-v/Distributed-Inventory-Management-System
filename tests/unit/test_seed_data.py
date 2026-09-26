"""Seeding behaviour for the two development accounts.

Every test runs against a throwaway SQLite session from ``tests/conftest.py``
-- ``scripts/seed_data.py``'s own ``main()`` is never called here, because
that would open ``data/node1.db``. The seeding *functions* take a session, so
they can be driven against a temporary database directly.

What is worth pinning down is idempotency. The script's contract is that
running it twice produces exactly the same rows as running it once, and the
subtle half of that is the password: bcrypt salts differ on every call, so a
seed that re-hashed unconditionally would rewrite the row each run and
silently replace a password a developer had changed by hand.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.domain import auth
from app.storage.models import ADMIN_ROLE, STAFF_ROLE, USER_ROLES
from app.storage.repositories import UserRepository
from scripts.seed_data import (
    DEV_ADMIN_EMAIL,
    DEV_ADMIN_PASSWORD,
    DEV_ADMIN_USERNAME,
    DEV_USER_EMAIL,
    DEV_USER_PASSWORD,
    DEV_USER_USERNAME,
    INVENTORY,
    PRODUCTS,
    SALES_HISTORY,
    SALES_WAREHOUSE,
    SEED_ORDER_PREFIX,
    SUPPLIER_PRODUCTS,
    seed_admin_user,
    seed_demo_user,
    seed_inventory,
    seed_products,
    seed_sales_history,
    seed_supplier_products,
    seed_suppliers,
    seed_warehouses,
)


def user(session: Session, username: str) -> Any:
    return UserRepository(session).get_by_username(username)


# --- the demo user ---------------------------------------------------------
def test_the_demo_user_is_created_on_a_fresh_database(session: Session) -> None:
    assert seed_demo_user(session) is True
    session.flush()

    created = user(session, DEV_USER_USERNAME)
    assert created is not None
    assert created.username == "user1"
    assert created.email == DEV_USER_EMAIL
    assert created.role == STAFF_ROLE
    assert created.is_active is True


def test_the_demo_user_gets_a_role_the_schema_already_defines(
    session: Session,
) -> None:
    """No new role is invented; `staff` is one of the three that existed."""
    seed_demo_user(session)
    session.flush()

    assert user(session, DEV_USER_USERNAME).role in USER_ROLES
    assert STAFF_ROLE == "staff"


def test_the_demo_password_is_hashed_never_stored_in_the_clear(
    session: Session,
) -> None:
    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert stored != DEV_USER_PASSWORD
    assert DEV_USER_PASSWORD not in stored
    assert stored.startswith("$2")  # bcrypt


def test_the_demo_user_can_actually_log_in(session: Session) -> None:
    """Hashed by the same code that verifies it, so the account really works."""
    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert auth.verify_password(DEV_USER_PASSWORD, stored) is True
    assert auth.verify_password("not-the-password", stored) is False


def test_seeding_the_demo_user_twice_adds_one_row(session: Session) -> None:
    assert seed_demo_user(session) is True
    session.flush()
    assert seed_demo_user(session) is False
    session.flush()

    assert len(UserRepository(session).list()) == 1


def test_a_rerun_does_not_rewrite_an_existing_demo_password(
    session: Session,
) -> None:
    """The subtle half of idempotency.

    bcrypt salts differ per call, so an unconditional re-hash would change
    the stored value on every run -- and would clobber a password the
    developer had changed by hand.
    """
    seed_demo_user(session)
    session.flush()
    before = user(session, DEV_USER_USERNAME).password_hash

    seed_demo_user(session)
    session.flush()

    assert user(session, DEV_USER_USERNAME).password_hash == before


def test_an_environment_override_sets_the_demo_password(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEED_USER_PASSWORD", "Another-Dev-Only-456")

    seed_demo_user(session)
    session.flush()

    stored = user(session, DEV_USER_USERNAME).password_hash
    assert auth.verify_password("Another-Dev-Only-456", stored) is True
    assert auth.verify_password(DEV_USER_PASSWORD, stored) is False


# --- the admin account, unchanged -----------------------------------------
def test_the_admin_account_is_unchanged(session: Session) -> None:
    """The constants the rest of the tooling imports must not have moved."""
    assert DEV_ADMIN_USERNAME == "admin"
    assert DEV_ADMIN_EMAIL == "admin@example.com"
    assert DEV_ADMIN_PASSWORD == "ChangeMe-Dev-Only-123"

    assert seed_admin_user(session) is True
    session.flush()

    admin = user(session, DEV_ADMIN_USERNAME)
    assert admin.role == ADMIN_ROLE
    assert auth.verify_password(DEV_ADMIN_PASSWORD, admin.password_hash) is True


def test_the_admin_seed_is_still_idempotent(session: Session) -> None:
    assert seed_admin_user(session) is True
    session.flush()
    assert seed_admin_user(session) is False
    session.flush()

    assert len(UserRepository(session).list()) == 1


def test_seeding_the_demo_user_leaves_the_admin_alone(session: Session) -> None:
    """Adding the second account must not disturb the first."""
    seed_admin_user(session)
    session.flush()
    before = user(session, DEV_ADMIN_USERNAME).password_hash

    seed_demo_user(session)
    session.flush()

    admin = user(session, DEV_ADMIN_USERNAME)
    assert admin.password_hash == before
    assert admin.role == ADMIN_ROLE
    assert admin.email == DEV_ADMIN_EMAIL


# --- both together ---------------------------------------------------------
def test_both_accounts_coexist_with_distinct_identities(session: Session) -> None:
    """`users.username` and `users.email` are both unique in the schema."""
    seed_admin_user(session)
    seed_demo_user(session)
    session.flush()

    rows = UserRepository(session).list()
    assert len(rows) == 2
    assert {row.username for row in rows} == {"admin", "user1"}
    assert len({row.email for row in rows}) == 2
    assert {row.role for row in rows} == {ADMIN_ROLE, STAFF_ROLE}


def test_a_full_reseed_adds_nothing_the_second_time(session: Session) -> None:
    """What `python scripts/seed_data.py` twice must do to the user rows."""
    first = (int(seed_admin_user(session)), int(seed_demo_user(session)))
    session.flush()
    second = (int(seed_admin_user(session)), int(seed_demo_user(session)))
    session.flush()

    assert first == (1, 1)
    assert second == (0, 0)
    assert len(UserRepository(session).list()) == 2


# --- the demo states the catalogue has to be able to show -----------------
#
# These assert the *data*, not the code that reads it. The seed's job is to
# put the system into states worth demonstrating, and a table of numbers can
# drift out of those states silently.
def test_the_seed_contains_a_low_stock_row() -> None:
    """Low stock is exactly quantity <= reorder_point.

    Before this was added the seed contained no such row at all, so the
    dashboard low-stock panel, `inventory.low_stock` and any reorder
    recommendation had nothing to show.
    """
    low = [(c, s, q, r) for c, s, q, r, _ in INVENTORY if q <= r]

    assert low, "no seeded stock row is at or below its reorder point"


def test_the_seed_contains_a_row_exactly_at_the_reorder_point() -> None:
    """The boundary, which is the case an off-by-one would get wrong."""
    assert any(q == r for _, _, q, r, _ in INVENTORY)


def test_the_seed_contains_normal_and_near_threshold_rows() -> None:
    normal = [(c, s) for c, s, q, r, _ in INVENTORY if q > r * 1.5]
    near = [(c, s) for c, s, q, r, _ in INVENTORY if r < q <= r * 1.5]

    assert normal, "every row is low -- nothing shows a healthy warehouse"
    assert near, "no row sits just above its reorder point"


def test_no_seeded_quantity_is_negative() -> None:
    """The schema forbids it; the seed must not be the thing that tries."""
    for _code, _sku, quantity, reorder, safety in INVENTORY:
        assert quantity >= 0
        assert reorder >= 0
        assert safety >= 0


def test_every_seeded_stock_row_names_a_seeded_product() -> None:
    skus = {str(spec["sku"]) for spec in PRODUCTS}
    for _code, sku, *_rest in INVENTORY:
        assert sku in skus, sku


def test_the_low_stock_product_has_supplier_terms() -> None:
    """A reorder recommendation needs a cost and a lead time to reason about."""
    low_skus = {sku for _c, sku, q, r, _s in INVENTORY if q <= r}
    supplied = {sku for _name, sku, *_rest in SUPPLIER_PRODUCTS}

    assert low_skus, "precondition: there is a low-stock product"
    assert low_skus <= supplied


# --- historical sales -----------------------------------------------------
def test_the_history_is_dense_enough_to_forecast_from() -> None:
    """Below SPARSE_DAY_THRESHOLD a rate is arithmetic, not evidence."""
    from app.domain.forecasting import DEFAULT_HISTORY_DAYS, SPARSE_DAY_THRESHOLD

    days = {days_ago for days_ago, _lines in SALES_HISTORY}

    assert len(days) > SPARSE_DAY_THRESHOLD
    assert max(days) <= DEFAULT_HISTORY_DAYS, "history starts outside the window"


def test_the_history_carries_a_trend_worth_showing() -> None:
    """Flat history makes a forecast demonstration say nothing."""
    from app.domain.forecasting import TREND_SENSITIVITY

    by_age = {
        days_ago: sum(q for sku, q in lines if sku == "SKU-1001")
        for days_ago, lines in SALES_HISTORY
    }
    oldest_first = [by_age[key] for key in sorted(by_age, reverse=True)]
    half = len(oldest_first) // 2
    older, newer = sum(oldest_first[:half]), sum(oldest_first[half:])

    assert newer > older * (1 + TREND_SENSITIVITY)


def test_every_seeded_sale_is_a_positive_quantity() -> None:
    """ck_sales_history_quantity_positive: the schema rejects zero."""
    for _days_ago, lines in SALES_HISTORY:
        assert lines, "an order with no lines"
        for _sku, quantity in lines:
            assert quantity > 0


def test_seeded_sales_name_real_products_and_a_real_warehouse() -> None:
    skus = {str(spec["sku"]) for spec in PRODUCTS}
    warehouses = {code for code, _sku, *_rest in INVENTORY}

    assert SALES_WAREHOUSE in warehouses
    for _days_ago, lines in SALES_HISTORY:
        for sku, _quantity in lines:
            assert sku in skus, sku


def seed_everything(session: Session) -> None:
    seed_admin_user(session)
    seed_demo_user(session)
    seed_products(session)
    seed_warehouses(session)
    seed_inventory(session)
    seed_suppliers(session)
    seed_supplier_products(session)
    session.flush()


def test_sales_history_is_written_with_its_orders(session: Session) -> None:
    """sales_history.order_id is NOT NULL, so history needs orders."""
    from app.storage.repositories import (
        OrderItemRepository,
        OrderRepository,
        SalesHistoryRepository,
    )

    seed_everything(session)
    added = seed_sales_history(session)
    session.flush()

    orders = OrderRepository(session).list()
    sales = SalesHistoryRepository(session).list()
    items = OrderItemRepository(session).list()

    assert added == sum(len(lines) for _days, lines in SALES_HISTORY)
    assert len(orders) == len(SALES_HISTORY)
    assert len(sales) == added
    assert len(items) == added
    assert all(row.order_id is not None for row in sales)


def test_seeded_orders_are_distinguishable_from_real_ones(
    session: Session,
) -> None:
    """A live demo order gets an ORD- number; these must not look like one."""
    from app.storage.repositories import OrderRepository

    seed_everything(session)
    seed_sales_history(session)
    session.flush()

    for order in OrderRepository(session).list():
        assert order.order_number.startswith(SEED_ORDER_PREFIX)


def test_seeded_orders_total_what_their_lines_come_to(session: Session) -> None:
    """The same arithmetic place_order does, so the past is a plausible one."""
    from decimal import Decimal

    from app.storage.repositories import OrderItemRepository, OrderRepository

    seed_everything(session)
    seed_sales_history(session)
    session.flush()

    lines_by_order: dict[int, Decimal] = {}
    for item in OrderItemRepository(session).list():
        lines_by_order[item.order_id] = (
            lines_by_order.get(item.order_id, Decimal("0.00")) + item.subtotal
        )

    for order in OrderRepository(session).list():
        assert order.total_amount == lines_by_order[order.id]
        assert order.completed_at is not None


def test_seeding_sales_history_twice_adds_nothing(session: Session) -> None:
    from app.storage.repositories import OrderRepository, SalesHistoryRepository

    seed_everything(session)
    first = seed_sales_history(session)
    session.flush()
    second = seed_sales_history(session)
    session.flush()

    assert first > 0
    assert second == 0
    assert len(OrderRepository(session).list()) == len(SALES_HISTORY)
    assert len(SalesHistoryRepository(session).list()) == first


def test_seeded_history_does_not_move_stock(session: Session) -> None:
    """Seeded quantities describe stock on hand today, not stock before sales.

    Decrementing here would invent a second, contradictory history.
    """
    from app.storage.repositories import InventoryRepository

    seed_everything(session)
    before = {row.id: row.quantity for row in InventoryRepository(session).list()}

    seed_sales_history(session)
    session.flush()

    after = {row.id: row.quantity for row in InventoryRepository(session).list()}
    assert after == before
    assert all(quantity >= 0 for quantity in after.values())
