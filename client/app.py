"""Streamlit client for the Distributed Inventory Management System.

Run with::

    streamlit run client/app.py

This is a presentation layer and nothing else. Every figure on every page was
computed by the application server; this file formats what came back and
sends what the operator typed. In particular it never decides whether stock
is sufficient, never subtracts a quantity, and never totals an order -- those
are the server's invariants, and duplicating them here would create a second
answer that could disagree with the database.

What this process can reach:

    browser --HTTP :8501--> Streamlit --gRPC :50051--> application server
                                                            |        |
                                                         SQLite   gRPC :50052

There is no SQLite import, no repository import and no address for the LLM
node anywhere in the client package. AI answers are requested from the
application server, which authenticates the caller and reads its own database
before asking the model anything.

All transport lives in :mod:`client.grpc_client`; this module holds the
widgets.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import streamlit as st

# Streamlit executes this file as a script rather than importing it as part
# of the package, so the repository root is not on sys.path by default.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from client.grpc_client import (
    AI_OPERATIONS,
    AuthenticationError,
    ClientError,
    InventoryClient,
    Session,
    get_server_target,
)

PAGES: tuple[str, ...] = (
    "Dashboard",
    "Products",
    "Warehouses",
    "Inventory",
    "Orders",
    "Suppliers",
    "AI Insights",
)

#: Human labels for the three server-side analyses, so the sidebar can read
#: in prose while the wire still carries the exact operation string.
AI_LABELS: dict[str, str] = {
    "DEMAND_FORECAST": "Demand forecast",
    "REORDER_RECOMMENDATION": "Reorder recommendation",
    "INVENTORY_SUMMARY": "Inventory summary",
}


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client() -> InventoryClient:
    """One gRPC channel per Streamlit server, reused across reruns.

    Streamlit re-executes this script top to bottom on every interaction, so
    building a channel at module scope would open a new one per click.
    """
    return InventoryClient()


def session() -> Session | None:
    return st.session_state.get("session")


def require_session() -> Session:
    active = session()
    if active is None:  # pragma: no cover - guarded by the caller
        st.stop()
    return active


def call(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one RPC, rendering any failure as a message instead of a crash.

    Returns ``None`` when the call failed, so a caller can tell "no result"
    from a genuine empty list. An expired or rejected session ends the
    session here rather than leaving a dead token in state for every
    subsequent page to trip over.
    """
    try:
        return action(*args, **kwargs)
    except AuthenticationError as exc:
        st.session_state.pop("session", None)
        st.error(f"{exc} Please sign in again.")
        st.rerun()
    except ClientError as exc:
        st.error(str(exc))
        return None


#: Columns worth seeing first, in the order an operator reads them. A Struct
#: is a map, so `MessageToDict` hands back keys in arbitrary order and a raw
#: render buries `quantity` behind `id` and `updated_at`.
COLUMN_ORDER: tuple[str, ...] = (
    "id",
    "sku",
    "code",
    "name",
    "product_name",
    "warehouse_code",
    "quantity",
    "reorder_point",
    "safety_stock",
    "is_low_stock",
    "unit_price",
    "email",
    "phone",
    "location",
    "description",
    "address",
    "is_active",
    "created_at",
    "updated_at",
)


def order_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put the useful columns first, keeping any the list does not name.

    Unknown keys are appended rather than dropped, so a column added on the
    server shows up without a change here.
    """
    known = [column for column in COLUMN_ORDER if any(column in row for row in rows)]
    extra = sorted({key for row in rows for key in row} - set(known))
    ordered = known + extra
    return [{key: row[key] for key in ordered if key in row} for row in rows]


def show_table(rows: list[dict[str, Any]] | None, empty: str) -> None:
    """Render rows, or say plainly that there are none.

    An empty result and a failed request look identical in a blank table, so
    they are worded differently: `call` has already reported a failure by the
    time this sees ``None``.
    """
    if rows is None:
        return
    if not rows:
        st.info(empty)
        return
    st.dataframe(order_columns(rows), width="stretch", hide_index=True)


def warehouse_codes(client: InventoryClient, token: str) -> list[str]:
    rows = call(client.list_warehouses, token, limit=500) or []
    return [row["code"] for row in rows]


def product_skus(client: InventoryClient, token: str) -> list[str]:
    rows = call(client.list_products, token, limit=500) or []
    return [row["sku"] for row in rows]


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def login_page(client: InventoryClient) -> None:
    st.title("Distributed Inventory Management")
    st.caption(f"Application server: {get_server_target()}")

    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if not submitted:
        return
    if not username or not password:
        st.warning("Enter both a username and a password.")
        return

    result = call(client.login, username, password)
    if result is None:
        return

    # Only the token is kept. The password went into the RPC above and is
    # not written to session state, not logged and not held on `result`; it
    # leaves scope when this function returns.
    st.session_state["session"] = result
    st.rerun()


def sidebar(client: InventoryClient, active: Session) -> str:
    with st.sidebar:
        st.subheader(active.username)
        st.caption(f"Role: {active.role}")
        if active.expires_at:
            st.caption(f"Session expires: {active.expires_at}")

        page = st.radio("Page", PAGES, label_visibility="collapsed")

        st.divider()
        if st.button("Sign out", width="stretch"):
            # Revoked on the server first. Clearing only the local state
            # would leave the session row valid on the node until it
            # expired, so a copied token would keep working.
            call(client.logout, active.token)
            st.session_state.pop("session", None)
            st.rerun()

    return str(page)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def dashboard_page(client: InventoryClient, active: Session) -> None:
    st.header("Dashboard")

    products = call(client.list_products, active.token, limit=500)
    warehouses = call(client.list_warehouses, active.token, limit=500)
    low_stock = call(client.list_low_stock, active.token)

    left, middle, right = st.columns(3)
    left.metric("Products", len(products) if products is not None else "-")
    middle.metric("Warehouses", len(warehouses) if warehouses is not None else "-")
    right.metric("Low stock lines", len(low_stock) if low_stock is not None else "-")

    st.subheader("Low stock across every warehouse")
    st.caption(
        "A line is low when its quantity is at or below its reorder point. "
        "The server decides that; this page only displays the flag."
    )
    show_table(low_stock, "No product is below its reorder point.")


def products_page(client: InventoryClient, active: Session) -> None:
    st.header("Products")

    active_only = st.checkbox("Active products only", value=False)
    show_table(
        call(client.list_products, active.token, active_only=active_only, limit=500),
        "No products yet.",
    )

    create, update = st.tabs(["Create", "Update"])

    with create, st.form("create_product"):
        sku = st.text_input("SKU")
        name = st.text_input("Name")
        description = st.text_area("Description")
        # A text input, not a number input: prices cross the wire as exact
        # decimal strings because a protobuf Struct stores numbers as
        # doubles, and a double cannot hold 1234567.89 exactly.
        unit_price = st.text_input("Unit price", value="0.00")
        if st.form_submit_button("Create product"):
            message = call(
                client.create_product,
                active.token,
                sku=sku,
                name=name,
                description=description,
                unit_price=unit_price,
            )
            if message:
                st.success(message)
                st.rerun()

    with update, st.form("update_product"):
        st.caption("Identify the product by SKU. Leave a field blank to keep it.")
        target = st.text_input("SKU of the product to update", key="update_sku")
        new_name = st.text_input("New name")
        new_description = st.text_area("New description")
        new_price = st.text_input("New unit price")
        is_active = st.selectbox("Active", ["unchanged", "yes", "no"])
        if st.form_submit_button("Update product"):
            fields: dict[str, Any] = {
                "sku_key": target,
                "name": new_name,
                "description": new_description,
                "unit_price": new_price,
            }
            if is_active != "unchanged":
                fields["is_active"] = is_active == "yes"
            message = call(client.update_product, active.token, **fields)
            if message:
                st.success(message)
                st.rerun()


def warehouses_page(client: InventoryClient, active: Session) -> None:
    st.header("Warehouses")

    show_table(
        call(client.list_warehouses, active.token, limit=500), "No warehouses yet."
    )

    with st.form("create_warehouse"):
        code = st.text_input("Code")
        name = st.text_input("Name")
        location = st.text_input("Location")
        if st.form_submit_button("Create warehouse"):
            message = call(
                client.create_warehouse,
                active.token,
                code=code,
                name=name,
                location=location,
            )
            if message:
                st.success(message)
                st.rerun()


def inventory_page(client: InventoryClient, active: Session) -> None:
    st.header("Inventory")

    codes = warehouse_codes(client, active.token)
    if not codes:
        st.info("Create a warehouse before managing stock.")
        return

    code = st.selectbox("Warehouse", codes)

    rows = call(client.list_warehouse_inventory, active.token, warehouse_code=code)
    if rows:
        low = [row for row in rows if row.get("is_low_stock")]
        if low:
            st.warning(f"{len(low)} of {len(rows)} lines are at or below reorder point.")
            st.subheader("Low stock")
            show_table(low, "")
        else:
            st.success("Every line in this warehouse is above its reorder point.")
        st.subheader("All stock")
    show_table(rows, "This warehouse holds no stock records.")

    initial, add = st.tabs(["Set initial stock", "Add stock"])

    with initial, st.form("set_initial_stock"):
        st.caption("Creates the stock row. Rejected if the product already has one.")
        sku = st.text_input("SKU", key="initial_sku")
        quantity = st.number_input("Quantity", min_value=0, step=1, value=0)
        reorder_point = st.number_input("Reorder point", min_value=0, step=1, value=0)
        safety_stock = st.number_input("Safety stock", min_value=0, step=1, value=0)
        if st.form_submit_button("Set initial stock"):
            message = call(
                client.set_initial_stock,
                active.token,
                warehouse_code=code,
                sku=sku,
                quantity=int(quantity),
                reorder_point=int(reorder_point),
                safety_stock=int(safety_stock),
            )
            if message:
                st.success(message)
                st.rerun()

    with add, st.form("add_stock"):
        st.caption(
            "Increment only. The server adds inside the database "
            "(SET quantity = quantity + n), so two concurrent additions "
            "cannot lose one another."
        )
        sku = st.text_input("SKU", key="add_sku")
        amount = st.number_input("Units to add", min_value=1, step=1, value=1)
        if st.form_submit_button("Add stock"):
            message = call(
                client.add_stock,
                active.token,
                warehouse_code=code,
                sku=sku,
                amount=int(amount),
            )
            if message:
                st.success(message)
                st.rerun()


def orders_page(client: InventoryClient, active: Session) -> None:
    st.header("Place an order")
    st.caption(
        "The server places the order atomically: every line succeeds or none "
        "does. Stock is never calculated or decremented in this client."
    )

    codes = warehouse_codes(client, active.token)
    skus = product_skus(client, active.token)
    if not codes or not skus:
        st.info("At least one warehouse and one product are needed to order.")
        return

    code = st.selectbox("Warehouse", codes, key="order_warehouse")

    line_count = st.number_input(
        "Number of lines", min_value=1, max_value=20, step=1, value=1
    )

    with st.form("place_order"):
        lines: list[dict[str, Any]] = []
        for index in range(int(line_count)):
            left, right = st.columns([3, 1])
            sku = left.selectbox(f"Product {index + 1}", skus, key=f"order_sku_{index}")
            quantity = right.number_input(
                "Quantity", min_value=1, step=1, value=1, key=f"order_qty_{index}"
            )
            lines.append({"sku": sku, "quantity": int(quantity)})

        if st.form_submit_button("Place order"):
            # Duplicate SKUs are sent as-is: the server consolidates them
            # into one line, which is the behaviour the domain layer defines
            # and not something to second-guess here.
            message = call(client.place_order, active.token, code, lines)
            if message:
                st.success(message)


def suppliers_page(client: InventoryClient, active: Session) -> None:
    st.header("Suppliers")

    show_table(call(client.list_suppliers, active.token, limit=500), "No suppliers yet.")

    create, associate = st.tabs(["Create supplier", "Link a product"])

    with create, st.form("create_supplier"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        address = st.text_area("Address")
        if st.form_submit_button("Create supplier"):
            message = call(
                client.create_supplier,
                active.token,
                name=name,
                email=email,
                phone=phone,
                address=address,
            )
            if message:
                st.success(message)
                st.rerun()

    with associate:
        st.caption(
            "Records what a supplier charges for a product and how long they "
            "take. Reorder recommendations read these terms."
        )
        with st.form("associate_product"):
            supplier_name = st.text_input("Supplier name")
            sku = st.text_input("Product SKU", key="link_sku")
            unit_cost = st.text_input("Unit cost", value="0.00")
            lead_time_days = st.number_input(
                "Lead time (days)", min_value=0, step=1, value=7
            )
            minimum_order_qty = st.number_input(
                "Minimum order quantity", min_value=1, step=1, value=1
            )
            is_preferred = st.checkbox("Preferred supplier for this product")
            if st.form_submit_button("Link product"):
                message = call(
                    client.associate_supplier_product,
                    active.token,
                    supplier_name=supplier_name,
                    sku=sku,
                    unit_cost=unit_cost,
                    lead_time_days=int(lead_time_days),
                    minimum_order_qty=int(minimum_order_qty),
                    is_preferred=is_preferred,
                )
                if message:
                    st.success(message)

    # Stated rather than worked around. The backend registers no `get`
    # action for supplier/product links, and inventing one here would mean
    # either a second data path or a backend change this step does not make.
    st.divider()
    st.caption(
        "Existing supplier/product links are visible through a reorder "
        "recommendation on AI Insights, which returns the supplier terms it "
        "used. The server exposes no read action for the links themselves."
    )


def ai_page(client: InventoryClient, active: Session) -> None:
    st.header("AI Insights")
    # Deliberately names the services rather than their ports: this process
    # knows one address, the application server's, and a test asserts that no
    # other port number appears anywhere in the client package.
    st.caption(
        "Requests go to the application server, which reads its own database, "
        "computes every number, and then asks the LLM node to explain them. "
        "This client never contacts the LLM node or Ollama, and no answer can "
        "change inventory."
    )

    operation = st.selectbox(
        "Analysis", AI_OPERATIONS, format_func=lambda value: AI_LABELS[value]
    )

    payload: dict[str, Any] = {}
    codes = warehouse_codes(client, active.token)

    if operation == "INVENTORY_SUMMARY":
        scope = st.selectbox("Scope", ["Every warehouse", *codes])
        if scope != "Every warehouse":
            payload["warehouse_code"] = scope
    else:
        skus = product_skus(client, active.token)
        if not codes or not skus:
            st.info("A product and a warehouse are needed for this analysis.")
            return
        payload["sku"] = st.selectbox("Product", skus)
        payload["warehouse_code"] = st.selectbox("Warehouse", codes, key="ai_warehouse")
        payload["history_days"] = int(
            st.number_input("Days of history", min_value=1, max_value=365, value=30)
        )
        if operation == "DEMAND_FORECAST":
            payload["horizon_days"] = int(
                st.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=14)
            )

    question = st.text_input("Optional question for the model")
    if question:
        payload["question"] = question

    if not st.button("Run analysis"):
        return

    with st.spinner("The model is generating an answer. This can take a minute."):
        result = call(client.business_request, active.token, operation, payload)
    if result is None:
        return

    st.subheader(AI_LABELS.get(operation, operation))
    if result.get("advisory"):
        st.warning(
            "Advisory only. Every figure below was computed by the "
            "application server; the prose is an explanation of those "
            "figures and has not changed any stock."
        )
    st.write(result.get("answer", ""))
    st.caption(f"Model: {result.get('model', 'unknown')} | Request: {result.get('request_id', '')}")

    with st.expander("The data the answer was based on"):
        st.json(result.get("context", {}))


PAGE_HANDLERS: dict[str, Callable[[InventoryClient, Session], None]] = {
    "Dashboard": dashboard_page,
    "Products": products_page,
    "Warehouses": warehouses_page,
    "Inventory": inventory_page,
    "Orders": orders_page,
    "Suppliers": suppliers_page,
    "AI Insights": ai_page,
}


def main() -> None:
    st.set_page_config(page_title="Inventory Management", layout="wide")
    client = get_client()

    active = session()
    if active is None:
        login_page(client)
        return

    page = sidebar(client, active)
    PAGE_HANDLERS[page](client, require_session())


# `streamlit run client/app.py` executes this file as __main__, so this is
# the real entry point. Guarded rather than called at module scope so the
# module can be imported -- by a test, or by `python -c` -- without building
# a UI or opening a channel.
if __name__ == "__main__":
    main()
