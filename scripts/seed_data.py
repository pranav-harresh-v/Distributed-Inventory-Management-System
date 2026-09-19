"""Insert deterministic demo data into this node's local database.

Usage (from anywhere)::

    python scripts/seed_data.py

Idempotent: every row is looked up by its natural key (username, SKU,
warehouse code, supplier name, or the composite keys of inventory and
supplier_products) and only inserted when absent. Running the script twice
produces exactly the same rows as running it once.

Development credentials
-----------------------
The demo admin password is ``DEV_ADMIN_PASSWORD`` below -- a throwaway
value for local development only. It is bcrypt-hashed before it reaches
the database; no plaintext password is ever stored. Override it with the
``SEED_ADMIN_PASSWORD`` environment variable. Never reuse it anywhere real.
"""

from __future__ import annotations

import os
import sys
from decimal import Decimal
from pathlib import Path

# Make the project importable when this file is run as a script rather than
# as a module (``python scripts/seed_data.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bcrypt
from sqlalchemy.orm import Session

from app.storage.database import get_database_path, init_db, session_scope
from app.storage.models import ADMIN_ROLE
from app.storage.repositories import (
    InventoryRepository,
    ProductRepository,
    SupplierProductRepository,
    SupplierRepository,
    UserRepository,
    WarehouseRepository,
)

# --- Development-only demo credentials -------------------------------------
DEV_ADMIN_USERNAME = "admin"
DEV_ADMIN_EMAIL = "admin@example.com"
DEV_ADMIN_PASSWORD = "ChangeMe-Dev-Only-123"

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
)


def hash_password(plaintext: str) -> str:
    """Return a bcrypt hash. Plaintext never leaves this function."""
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


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


def main() -> int:
    # Seeding an uninitialised checkout should just work.
    init_db()

    with session_scope() as session:
        counts = {
            "admin user": int(seed_admin_user(session)),
            "products": seed_products(session),
            "warehouses": seed_warehouses(session),
            "inventory rows": seed_inventory(session),
            "suppliers": seed_suppliers(session),
            "supplier products": seed_supplier_products(session),
        }

    summary = ", ".join(f"{name}: {count}" for name, count in counts.items())
    print(f"Seeded {get_database_path()}")
    print(f"Rows inserted this run -> {summary}")
    if not any(counts.values()):
        print("Nothing to do; demo data was already present (script is idempotent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
