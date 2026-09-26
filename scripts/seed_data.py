"""Insert deterministic demo data into this node's local database.

Usage (from anywhere)::

    python scripts/seed_data.py

Idempotent: every row is looked up by its natural key (username, SKU,
warehouse code, supplier name, or the composite keys of inventory and
supplier_products) and only inserted when absent. Running the script twice
produces exactly the same rows as running it once.

Development credentials
-----------------------
Two accounts are seeded, both throwaway values for local development only:

===========  ==========  =========================  ==========================
Account      Role        Password constant          Environment override
===========  ==========  =========================  ==========================
``admin``    ``admin``   ``DEV_ADMIN_PASSWORD``     ``SEED_ADMIN_PASSWORD``
``user1``    ``staff``   ``DEV_USER_PASSWORD``      ``SEED_USER_PASSWORD``
===========  ==========  =========================  ==========================

``admin`` drives the admin client (``client/app.py``); ``user1`` is an
ordinary account for the user client (``client/user_app.py``). Both are
bcrypt-hashed before they reach the database; no plaintext password is ever
stored. Never reuse either anywhere real.

The roles differ only in what they *say*: role-based authorization is not
implemented, so ``user1`` is not currently restricted by the server. The
account exists so the user client can be driven by something other than the
administrator.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

# Make the project importable when this file is run as a script rather than
# as a module (``python scripts/seed_data.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.orm import Session

from app.domain import auth
from app.storage.database import get_database_path, init_db, session_scope
from app.storage.models import (
    ADMIN_ROLE,
    CURRENCY_EXPONENT,
    ORDER_COMPLETED,
    STAFF_ROLE,
    utc_now,
)
from app.storage.repositories import (
    InventoryRepository,
    OrderItemRepository,
    OrderRepository,
    ProductRepository,
    SalesHistoryRepository,
    SupplierProductRepository,
    SupplierRepository,
    UserRepository,
    WarehouseRepository,
)

# --- Development-only demo credentials -------------------------------------
DEV_ADMIN_USERNAME = "admin"
DEV_ADMIN_EMAIL = "admin@example.com"
DEV_ADMIN_PASSWORD = "ChangeMe-Dev-Only-123"

DEV_USER_USERNAME = "user1"
DEV_USER_EMAIL = "user1@example.com"
DEV_USER_PASSWORD = "User1-Dev-Only-123"

# --- Deterministic demo catalogue ------------------------------------------
PRODUCTS: tuple[dict[str, object], ...] = (
    {
        "sku": "SKU-1001",
        "name": "Wireless Mouse",
        "description": "Bluetooth optical mouse",
        "unit_price": Decimal("24.99"),
    },
    {
        "sku": "SKU-1002",
        "name": "Mechanical Keyboard",
        "description": "87-key tenkeyless keyboard",
        "unit_price": Decimal("79.50"),
    },
    {
        "sku": "SKU-1003",
        "name": "27in Monitor",
        "description": "27-inch 1440p IPS display",
        "unit_price": Decimal("249.00"),
    },
    {
        "sku": "SKU-1004",
        "name": "USB-C Dock",
        "description": "8-port USB-C docking station",
        "unit_price": Decimal("129.95"),
    },
    # Added so the demo has something genuinely below its reorder point.
    # Introduced as a new product rather than by lowering an existing
    # quantity: the seed skips rows that already exist, so editing a number
    # would only reach a database created after the edit, while a new
    # product and its two stock rows land in an already-seeded one too.
    {
        "sku": "SKU-1005",
        "name": "Laptop Stand",
        "description": "Aluminium adjustable laptop stand",
        "unit_price": Decimal("39.95"),
    },
)

WAREHOUSES: tuple[dict[str, str], ...] = (
    {"code": "WH-001", "name": "North Depot", "location": "Bengaluru, IN"},
    {"code": "WH-002", "name": "South Depot", "location": "Chennai, IN"},
)

# (warehouse code, sku) -> stock levels
INVENTORY: tuple[tuple[str, str, int, int, int], ...] = (
    ("WH-001", "SKU-1001", 150, 40, 20),
    ("WH-001", "SKU-1002", 80, 25, 10),
    ("WH-001", "SKU-1003", 30, 10, 5),
    ("WH-001", "SKU-1004", 60, 20, 10),
    ("WH-002", "SKU-1001", 90, 30, 15),
    ("WH-002", "SKU-1002", 45, 20, 10),
    ("WH-002", "SKU-1003", 12, 10, 5),
    ("WH-002", "SKU-1004", 25, 15, 8),
    # Low stock is exactly `quantity <= reorder_point` (app.domain.inventory
    # .is_low_stock). Before these two rows the seed contained none at all,
    # so the dashboard's low-stock panel, `inventory.low_stock` and any
    # reorder recommendation had nothing to show.
    ("WH-001", "SKU-1005", 6, 15, 5),  # clearly below the reorder point
    ("WH-002", "SKU-1005", 15, 15, 5),  # exactly at it -- the boundary case
)

SUPPLIERS: tuple[dict[str, str], ...] = (
    {
        "name": "Nimbus Components",
        "email": "sales@nimbus.example.com",
        "phone": "+91-80-5550-0101",
        "address": "12 Industrial Layout, Bengaluru",
    },
    {
        "name": "Vertex Peripherals",
        "email": "orders@vertex.example.com",
        "phone": "+91-44-5550-0202",
        "address": "88 Export Road, Chennai",
    },
)

# (supplier name, sku, supplier sku, unit cost, lead time, MOQ, preferred)
SUPPLIER_PRODUCTS: tuple[tuple[str, str, str, Decimal, int, int, bool], ...] = (
    ("Nimbus Components", "SKU-1001", "NC-MOU-01", Decimal("14.20"), 5, 25, True),
    ("Nimbus Components", "SKU-1002", "NC-KEY-02", Decimal("48.00"), 7, 10, True),
    ("Nimbus Components", "SKU-1003", "NC-MON-03", Decimal("182.40"), 14, 5, False),
    ("Vertex Peripherals", "SKU-1003", "VP-MON-27", Decimal("176.75"), 10, 5, True),
    ("Vertex Peripherals", "SKU-1004", "VP-DOCK-8", Decimal("86.30"), 9, 10, True),
    # The low-stock product needs supplier terms, or a reorder
    # recommendation for it has no cost or lead time to reason about.
    ("Nimbus Components", "SKU-1005", "NC-STD-05", Decimal("21.75"), 6, 20, True),
    ("Vertex Peripherals", "SKU-1005", "VP-STD-AL", Decimal("23.40"), 4, 15, False),
)


# --- Historical sales, for the forecasting and reorder demonstrations ------
#
# `sales_history.order_id` is a NOT NULL foreign key, so history cannot exist
# without orders to hang it on. These orders are therefore seeded *because*
# the forecast needs them, which is the "specific demonstration purpose" that
# justifies them; they are not decoration. Their numbers carry a `SEED-ORD-`
# prefix so they are distinguishable at a glance from the `ORD-` numbers
# `place_order` generates during a live demo.
#
# Dates are relative to now, not absolute. `demand_statistics` looks back a
# window of days from the moment it is asked, so a fixed calendar date would
# quietly slide out of that window and leave the forecast with no data --
# the demo would break by doing nothing.
#
# The shape is deliberate: SKU-1001 rises from 3/day to 7/day across the
# window, which is well past TREND_SENSITIVITY (10%) and shows as a rising
# trend; 15 distinct selling days is well past SPARSE_DAY_THRESHOLD (3), so
# the rate reads as evidence rather than arithmetic. SKU-1005 sells steadily
# while sitting below its reorder point, which is what makes a reorder
# recommendation for it worth asking for.
#
# Warehouse WH-001 throughout: a forecast is per (product, warehouse), so
# concentrating history gives one pairing enough depth to be meaningful
# rather than spreading it too thin across two.
SALES_WAREHOUSE = "WH-001"

#: (days ago, ((sku, quantity), ...)) -- one seeded order per entry.
SALES_HISTORY: tuple[tuple[int, tuple[tuple[str, int], ...]], ...] = (
    (28, (("SKU-1001", 3), ("SKU-1002", 1))),
    (26, (("SKU-1001", 2),)),
    (24, (("SKU-1001", 4), ("SKU-1005", 1))),
    (22, (("SKU-1001", 3), ("SKU-1002", 2))),
    (20, (("SKU-1001", 3),)),
    (18, (("SKU-1001", 4), ("SKU-1005", 2))),
    (16, (("SKU-1001", 5), ("SKU-1002", 1))),
    (14, (("SKU-1001", 4),)),
    (12, (("SKU-1001", 5), ("SKU-1005", 2))),
    (10, (("SKU-1001", 6), ("SKU-1002", 2))),
    (8, (("SKU-1001", 5),)),
    (6, (("SKU-1001", 7), ("SKU-1005", 3))),
    (4, (("SKU-1001", 6), ("SKU-1002", 3))),
    (2, (("SKU-1001", 8), ("SKU-1005", 3))),
    (1, (("SKU-1001", 7),)),
)

#: Marks an order as seeded rather than placed by a person during the demo.
SEED_ORDER_PREFIX = "SEED-ORD-"


def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash of ``plaintext``.

    Delegates to the domain helper so the seeded admin is hashed by exactly
    the same code that verifies it at login. A second bcrypt call here could
    drift from that one and silently make the demo account unusable.
    """
    return auth.hash_password(plaintext)


def seed_admin_user(session: Session) -> bool:
    """Create the demo admin if absent. Returns True when a row was added.

    An existing admin is left untouched: re-hashing on every run would
    rewrite the row (bcrypt salts differ each time) and could clobber a
    password the developer changed by hand.
    """
    users = UserRepository(session)
    if users.get_by_username(DEV_ADMIN_USERNAME) is not None:
        return False
    password = os.environ.get("SEED_ADMIN_PASSWORD", DEV_ADMIN_PASSWORD)
    users.create(
        username=DEV_ADMIN_USERNAME,
        email=DEV_ADMIN_EMAIL,
        password_hash=hash_password(password),
        role=ADMIN_ROLE,
    )
    return True


def seed_demo_user(session: Session) -> bool:
    """Create the ordinary demo user if absent. True when a row was added.

    Deliberately a copy of :func:`seed_admin_user`'s shape rather than a
    shared helper the two call: the admin seed is working code that other
    tooling (``scripts/grpc_smoke_test.py``) imports constants from, and
    refactoring it to add a second account would put that at risk for no
    gain. Two short functions that each do one thing are cheaper here than
    one parameterised function and a changed call site.

    An existing ``user1`` is left untouched, for the same reason the admin
    is: bcrypt salts differ per call, so re-hashing every run would rewrite
    the row and clobber a password changed by hand.
    """
    users = UserRepository(session)
    if users.get_by_username(DEV_USER_USERNAME) is not None:
        return False
    password = os.environ.get("SEED_USER_PASSWORD", DEV_USER_PASSWORD)
    users.create(
        username=DEV_USER_USERNAME,
        email=DEV_USER_EMAIL,
        password_hash=hash_password(password),
        # The role the schema already defines. No new role is invented, and
        # nothing here enforces it -- see the module docstring.
        role=STAFF_ROLE,
    )
    return True


def seed_products(session: Session) -> int:
    products = ProductRepository(session)
    added = 0
    for spec in PRODUCTS:
        if products.get_by_sku(str(spec["sku"])) is None:
            products.create(**spec)  # type: ignore[arg-type]
            added += 1
    return added


def seed_warehouses(session: Session) -> int:
    warehouses = WarehouseRepository(session)
    added = 0
    for spec in WAREHOUSES:
        if warehouses.get_by_code(spec["code"]) is None:
            warehouses.create(**spec)
            added += 1
    return added


def seed_inventory(session: Session) -> int:
    products = ProductRepository(session)
    warehouses = WarehouseRepository(session)
    inventory = InventoryRepository(session)
    added = 0
    for code, sku, quantity, reorder_point, safety_stock in INVENTORY:
        warehouse = warehouses.get_by_code(code)
        product = products.get_by_sku(sku)
        if warehouse is None or product is None:
            raise RuntimeError(f"seed inventory refers to unknown {code}/{sku}")
        if inventory.get_by_warehouse_and_product(warehouse.id, product.id) is not None:
            continue
        inventory.create(
            warehouse_id=warehouse.id,
            product_id=product.id,
            quantity=quantity,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
        )
        added += 1
    return added


def seed_suppliers(session: Session) -> int:
    suppliers = SupplierRepository(session)
    added = 0
    for spec in SUPPLIERS:
        if suppliers.get_by_name(spec["name"]) is None:
            suppliers.create(**spec)
            added += 1
    return added


def seed_supplier_products(session: Session) -> int:
    products = ProductRepository(session)
    suppliers = SupplierRepository(session)
    links = SupplierProductRepository(session)
    added = 0
    for name, sku, supplier_sku, cost, lead_time, moq, preferred in SUPPLIER_PRODUCTS:
        supplier = suppliers.get_by_name(name)
        product = products.get_by_sku(sku)
        if supplier is None or product is None:
            raise RuntimeError(f"seed supplier_products refers to unknown {name}/{sku}")
        if links.get_by_supplier_and_product(supplier.id, product.id) is not None:
            continue
        links.create(
            supplier_id=supplier.id,
            product_id=product.id,
            supplier_sku=supplier_sku,
            unit_cost=cost,
            lead_time_days=lead_time,
            minimum_order_qty=moq,
            is_preferred=preferred,
        )
        added += 1
    return added



def seed_sales_history(session: Session) -> int:
    """Create the historical orders and their sold lines. Returns rows added.

    Mirrors what :func:`app.domain.orders.place_order` writes -- an order
    header, one ``order_items`` row per line and one ``sales_history`` row
    per line -- so the seeded past is the same shape as a real one. It does
    **not** decrement inventory: the seeded quantities describe stock on hand
    today, and a warehouse that had since been replenished would look exactly
    like this. Subtracting them would only invent a second, contradictory
    history.

    Idempotent on the order number, which is derived from the entry's
    position, so a rerun adds nothing.
    """
    users = UserRepository(session)
    products = ProductRepository(session)
    warehouses = WarehouseRepository(session)
    orders = OrderRepository(session)
    items = OrderItemRepository(session)
    sales = SalesHistoryRepository(session)

    actor = users.get_by_username(DEV_ADMIN_USERNAME)
    warehouse = warehouses.get_by_code(SALES_WAREHOUSE)
    if actor is None or warehouse is None:
        raise RuntimeError("seed sales history needs the admin user and WH-001")

    added = 0
    now = utc_now()
    for index, (days_ago, lines) in enumerate(SALES_HISTORY, start=1):
        number = f"{SEED_ORDER_PREFIX}{index:02d}"
        if orders.get_by_order_number(number) is not None:
            continue

        sold_at = now - timedelta(days=days_ago)
        order = orders.create(
            order_number=number,
            user_id=actor.id,
            warehouse_id=warehouse.id,
            status=ORDER_COMPLETED,
            total_amount=Decimal("0.00"),
        )

        total = Decimal("0.00")
        for sku, quantity in lines:
            product = products.get_by_sku(sku)
            if product is None:
                raise RuntimeError(f"seed sales history refers to unknown {sku}")
            # The price the sale happened at, taken from the catalogue, so
            # the arithmetic below matches what place_order would have done.
            subtotal = (product.unit_price * quantity).quantize(CURRENCY_EXPONENT)
            total += subtotal
            items.create(
                order_id=order.id,
                product_id=product.id,
                quantity=quantity,
                unit_price=product.unit_price,
                subtotal=subtotal,
            )
            sales.create(
                order_id=order.id,
                warehouse_id=warehouse.id,
                product_id=product.id,
                quantity=quantity,
                unit_price=product.unit_price,
                sold_at=sold_at,
            )
            added += 1

        order.total_amount = total
        order.completed_at = sold_at
        orders.save(order)

    return added


def main() -> int:
    # Seeding an uninitialised checkout should just work.
    init_db()

    with session_scope() as session:
        counts = {
            "admin user": int(seed_admin_user(session)),
            "demo user": int(seed_demo_user(session)),
            "products": seed_products(session),
            "warehouses": seed_warehouses(session),
            "inventory rows": seed_inventory(session),
            "suppliers": seed_suppliers(session),
            "supplier products": seed_supplier_products(session),
            "sales history": seed_sales_history(session),
        }

    summary = ", ".join(f"{name}: {count}" for name, count in counts.items())
    print(f"Seeded {get_database_path()}")
    print(f"Rows inserted this run -> {summary}")
    if not any(counts.values()):
        print("Nothing to do; demo data was already present (script is idempotent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
