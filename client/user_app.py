"""Streamlit client for ordinary users.

Run with::

    streamlit run client/user_app.py

A second, separate entrypoint alongside the admin client in
``client/app.py``. The two are independent Streamlit applications that share
one thing deliberately -- :mod:`client.grpc_client` -- and nothing else. No
admin page, helper or navigation entry is imported, reused or altered here,
so this file cannot change how the admin application behaves.

Where the admin client manages the catalogue, this one only *consumes* it:
browse products, check availability, place an order. There is no create,
update or stock adjustment anywhere in this file.

    user browser --HTTP :8502--> Streamlit --gRPC :50051--> application server
                                                                   |
                                                                SQLite

The same architectural rules as the admin client apply, for the same
reasons: no SQLite import, no repository import, no address for the LLM
node. Ordering in particular is entirely the server's business -- this file
never checks whether stock is sufficient, never subtracts a quantity and
never totals a basket. It sends the request and reports what came back. A
second opinion computed here could only ever disagree with the database,
and the database is the one that decides.

Scope note: order history is not offered. The backend exposes no
actor-aware read, so a "my orders" view could not be built without either a
backend change or letting one user read another's orders. Rather than fake
it, the feature is absent.
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
    AuthenticationError,
    ClientError,
    InventoryClient,
    PreconditionError,
    Session,
    get_server_target,
)

#: The four things a user can do. Deliberately short: this is not the admin
#: console and does not mirror its navigation.
PAGES: tuple[str, ...] = ("Browse Products", "Availability", "Place Order")

#: Session-state keys, namespaced so this app's state can never be confused
#: with the admin app's when reading either file.
SESSION_KEY = "user_session"
CART_KEY = "cart"

#: What `post order.place` gives back. `StatusResponse` carries only
#: `success` and `message`, so there is no structured order id and no
#: structured line list to read: the server builds one sentence naming the
#: order number, the line count, the total and the status. That sentence is
#: shown verbatim rather than picked apart -- parsing prose into fields would
#: invent a contract the proto does not offer and would break the day the
#: wording changed.

#: Carries the outcome of a submitted order across the rerun that follows it.
#: Streamlit renders the sidebar before the page body, so a basket mutated
#: during the body would leave the sidebar's count one interaction stale --
#: showing "1 line(s)" over an emptied basket. Mutating, rerunning and
#: rendering the outcome afterwards keeps the two honest.
ORDER_RESULT_KEY = "last_order_result"

#: Columns a shopper needs, in reading order. Every one of these is a field
#: the server actually returns -- `serialize_product` and
#: `serialize_inventory` define them; none is invented here.
PRODUCT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("sku", "SKU"),
    ("name", "Product"),
    ("description", "Description"),
    ("unit_price", "Unit price"),
)

AVAILABILITY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("warehouse_code", "Warehouse"),
    ("sku", "SKU"),
    ("product_name", "Product"),
    ("quantity", "In stock"),
    ("reorder_point", "Reorder point"),
    ("is_low_stock", "Low stock"),
)

#: The order contract caps a request at 100 lines; the basket is held well
#: below that so the limit is never the thing a user discovers.
MAX_CART_LINES = 20


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
@st.cache_resource
def get_client() -> InventoryClient:
    """One gRPC channel per Streamlit server, reused across reruns."""
    return InventoryClient()


def session() -> Session | None:
    return st.session_state.get(SESSION_KEY)


def cart() -> list[dict[str, Any]]:
    return st.session_state.setdefault(CART_KEY, [])


def call(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one RPC, rendering any failure as a message rather than a crash.

    Returns ``None`` when the call failed, so a caller can tell "no result"
    from a genuinely empty list. Every gRPC status the server can return is
    already translated into a typed exception by
    :mod:`client.grpc_client`; this only decides how each is shown.

    An expired or rejected session ends the session here, so a dead token
    cannot linger in state for every later page to trip over.
    """
    try:
        return action(*args, **kwargs)
    except AuthenticationError as exc:
        _end_session()
        st.error(f"{exc} Please sign in again.")
        st.rerun()
    except ClientError as exc:
        st.error(str(exc))
        return None


def _end_session() -> None:
    """Drop every trace of the signed-in user from local state."""
    for key in (SESSION_KEY, CART_KEY, ORDER_RESULT_KEY):
        st.session_state.pop(key, None)


def show_rows(
    rows: list[dict[str, Any]] | None,
    columns: tuple[tuple[str, str], ...],
    empty: str,
) -> None:
    """Render selected columns under friendly headings.

    Only the named fields are shown. Ids and timestamps exist on the wire
    but mean nothing to a shopper, and a Struct hands back its keys in
    arbitrary order, so the selection is explicit rather than incidental.
    """
    if rows is None:
        return
    if not rows:
        st.info(empty)
        return
    st.dataframe(
        [{label: row.get(key) for key, label in columns} for row in rows],
        width="stretch",
        hide_index=True,
    )


def warehouse_codes(client: InventoryClient, token: str) -> list[str]:
    rows = call(client.list_warehouses, token, limit=500) or []
    return [row["code"] for row in rows]


def product_rows(client: InventoryClient, token: str) -> list[dict[str, Any]]:
    return call(client.list_products, token, active_only=True, limit=500) or []


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def login_page(client: InventoryClient) -> None:
    st.title("Inventory Store")
    st.caption(f"Application server: {get_server_target()}")

    with st.form("user_login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if not submitted:
        return
    if not username or not password:
        st.warning("Enter both a username and a password.")
        return

    result = call(client.login, username, password)
    if result is None:
        return

    # Only the token is kept. The password was an argument to the RPC above
    # and is not written to session state, not logged and not held on
    # `result` -- it leaves scope when this function returns.
    st.session_state[SESSION_KEY] = result
    st.session_state[CART_KEY] = []
    st.rerun()


def sidebar(client: InventoryClient, active: Session) -> str:
    with st.sidebar:
        st.subheader(f"Hello, {active.username}")
        if active.expires_at:
            st.caption(f"Signed in until {active.expires_at}")

        page = st.radio("Menu", PAGES, label_visibility="collapsed")

        basket = cart()
        if basket:
            st.caption(f"Basket: {len(basket)} line(s)")

        st.divider()
        if st.button("Log out", width="stretch"):
            # Revoked on the server first. Clearing only local state would
            # leave the session row valid on the node until it expired, so
            # a copied token would keep working.
            call(client.logout, active.token)
            _end_session()
            st.rerun()

    return str(page)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def browse_page(client: InventoryClient, active: Session) -> None:
    st.header("Browse products")
    st.caption("Everything currently offered. Prices are per unit.")

    rows = product_rows(client, active.token)
    search = st.text_input("Search by name or SKU").strip().lower()
    if search:
        rows = [
            row
            for row in rows
            if search in str(row.get("name", "")).lower()
            or search in str(row.get("sku", "")).lower()
        ]

    show_rows(rows, PRODUCT_COLUMNS, "No products are available right now.")


def availability_page(client: InventoryClient, active: Session) -> None:
    st.header("Availability")
    st.caption(
        "Stock levels as the server reports them. Nothing here is calculated "
        "in the browser."
    )

    codes = warehouse_codes(client, active.token)
    if not codes:
        st.info("No warehouses are available.")
        return

    code = st.selectbox("Warehouse", codes)
    rows = call(client.list_warehouse_inventory, active.token, warehouse_code=code)

    if rows:
        # `is_low_stock` is decided by the server; this only counts the flags.
        low = [row for row in rows if row.get("is_low_stock")]
        if low:
            st.warning(f"{len(low)} of {len(rows)} lines are running low.")

    show_rows(rows, AVAILABILITY_COLUMNS, "This warehouse holds no stock.")


def validate_line(sku: Any, quantity: Any) -> str | None:
    """Check one basket line. Returns an error message, or None if it is fine.

    A courtesy, not a control. The widgets already constrain most of this --
    a selectbox always has a selection, a number_input with ``min_value=1``
    will not hand back zero -- but "already constrained" is not "checked",
    and a cleared numeric field can come back as None in some Streamlit
    versions. Catching it here turns a confusing INVALID_ARGUMENT into a
    sentence next to the field that caused it.

    The server validates every one of these again and is the only thing that
    decides whether an order is acceptable. Nothing here may be relied on:
    it exists to save a round trip, not to stand in for one.
    """
    if not sku:
        return "Choose a product before adding it to the basket."
    if quantity is None or quantity == "":
        return "Enter a quantity."
    try:
        amount = int(quantity)
    except (TypeError, ValueError):
        return "Quantity must be a whole number."
    if amount != float(quantity):
        return "Quantity must be a whole number."
    if amount < 1:
        return "Quantity must be at least 1."
    return None


def validate_basket(basket: list[dict[str, Any]]) -> str | None:
    """Check the basket as a whole, just before it is sent."""
    if not basket:
        return "Your basket is empty. Add an item before placing an order."
    for line in basket:
        problem = validate_line(line.get("sku"), line.get("quantity"))
        if problem is not None:
            return f"{line.get('sku', 'A line')}: {problem}"
    return None


def _add_to_cart(sku: str, quantity: int) -> None:
    """Add a line, merging a repeat of the same product.

    The server consolidates duplicate lines anyway, so merging here only
    makes the basket read the way a user expects; it changes nothing about
    what the order means.
    """
    basket = cart()
    for line in basket:
        if line["sku"] == sku:
            line["quantity"] += quantity
            return
    basket.append({"sku": sku, "quantity": quantity})


def order_page(client: InventoryClient, active: Session) -> None:
    st.header("Place an order")
    st.caption(
        "Stock is checked, reserved and decremented by the server, in one "
        "transaction. Either every line succeeds or none does."
    )

    # The outcome of the submit that caused this rerun, if there was one.
    _render_order_result(client, active)

    codes = warehouse_codes(client, active.token)
    products = product_rows(client, active.token)
    if not codes or not products:
        st.info("Ordering needs at least one warehouse and one product.")
        return

    labels = {f"{row['sku']} - {row['name']}": row["sku"] for row in products}

    code = st.selectbox("Deliver from warehouse", codes, key="order_warehouse")

    # --- add a line -------------------------------------------------------
    st.subheader("1. Add items")
    with st.form("add_line", clear_on_submit=True):
        left, right = st.columns([3, 1])
        choice = left.selectbox("Product", sorted(labels))
        quantity = right.number_input("Quantity", min_value=1, step=1, value=1)
        if st.form_submit_button("Add to basket"):
            problem = validate_line(labels.get(choice), quantity)
            if problem is not None:
                st.warning(problem)
            elif len(cart()) >= MAX_CART_LINES:
                st.warning(f"A basket holds at most {MAX_CART_LINES} lines.")
            else:
                _add_to_cart(labels[choice], int(quantity))
                st.rerun()

    # --- review -----------------------------------------------------------
    st.subheader("2. Review")
    basket = cart()
    if not basket:
        st.info("Your basket is empty. Add an item above.")
        return

    prices = {row["sku"]: row.get("unit_price") for row in products}
    names = {row["sku"]: row.get("name") for row in products}
    st.dataframe(
        [
            {
                "SKU": line["sku"],
                "Product": names.get(line["sku"], ""),
                "Quantity": line["quantity"],
                # Shown for orientation only. The server computes the total
                # it charges, and that figure is the one that counts.
                "Unit price": prices.get(line["sku"], ""),
            }
            for line in basket
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The order total is calculated by the server when the order is "
        "placed, using the prices held in its own database."
    )

    remove, clear = st.columns(2)
    removable = [line["sku"] for line in basket]
    with remove:
        to_remove = st.selectbox("Remove a line", removable, key="remove_line")
        if st.button("Remove", width="stretch"):
            st.session_state[CART_KEY] = [
                line for line in basket if line["sku"] != to_remove
            ]
            st.rerun()
    with clear:
        st.write("")
        st.write("")
        if st.button("Empty basket", width="stretch"):
            st.session_state[CART_KEY] = []
            st.rerun()

    # --- submit -----------------------------------------------------------
    st.subheader("3. Submit")
    if not st.button("Place order", type="primary", width="stretch"):
        return

    # Checked here so an obviously bad basket is answered immediately. The
    # server checks all of it again, and its answer is the one that counts.
    problem = validate_basket(basket)
    if problem is not None:
        st.warning(problem)
        return

    # Kept so the confirmation can list what was ordered: the reply carries
    # no line data, so this is the request echoed back, not server state.
    submitted = [dict(line) for line in basket]

    # The basket is sent exactly as the existing contract expects: a
    # warehouse and a list of {sku, quantity}. No new RPC, no new payload
    # shape, and nothing about stock decided here.
    try:
        message = client.place_order(active.token, code, list(basket))
    except PreconditionError as exc:
        # The request was well formed; the warehouse simply could not meet
        # it. The basket is kept so the user can adjust a quantity rather
        # than rebuild the whole order.
        st.session_state[ORDER_RESULT_KEY] = (
            "short",
            f"Order not placed: {exc}",
            code,
            [],
        )
        st.rerun()
    except AuthenticationError as exc:
        _end_session()
        st.error(f"{exc} Please sign in again.")
        st.rerun()
    except ClientError as exc:
        st.session_state[ORDER_RESULT_KEY] = (
            "error",
            f"Order not placed: {exc}",
            code,
            [],
        )
        st.rerun()

    # Only reached when the server confirmed the order.
    st.session_state[CART_KEY] = []
    st.session_state[ORDER_RESULT_KEY] = ("placed", message, code, submitted)
    st.rerun()


def _render_order_result(client: InventoryClient, active: Session) -> None:
    """Show the outcome of the order that triggered this rerun, once.

    Popped rather than read, so a result is shown for the submission that
    produced it and does not linger over the next one.
    """
    result = st.session_state.pop(ORDER_RESULT_KEY, None)
    if result is None:
        return

    outcome, message, code, submitted = result
    if outcome == "placed":
        # The server's own sentence, verbatim. It names the order number,
        # the line count, the total it charged and the final status -- all
        # of them the server's figures, none recomputed here.
        st.success(message)
        if submitted:
            st.caption("Items ordered")
            st.dataframe(
                [
                    {"SKU": line["sku"], "Quantity": line["quantity"]}
                    for line in submitted
                ],
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "The lines above are the ones submitted. The confirmation "
                "message above them is the server's record of what it wrote."
            )
    else:
        st.error(message)
        if outcome == "short":
            st.info("Your basket has been kept. Adjust a quantity and try again.")

    _show_refreshed_stock(client, active, code)
    st.divider()


def _show_refreshed_stock(
    client: InventoryClient, active: Session, code: str
) -> None:
    """Re-read stock from the server after an order attempt.

    Re-read rather than adjusted locally: whether the order succeeded, failed
    or partly raced another order, the server's figures are the only
    trustworthy ones.
    """
    st.subheader("Stock now, as the server reports it")
    rows = call(client.list_warehouse_inventory, active.token, warehouse_code=code)
    show_rows(rows, AVAILABILITY_COLUMNS, "This warehouse holds no stock.")


PAGE_HANDLERS: dict[str, Callable[[InventoryClient, Session], None]] = {
    "Browse Products": browse_page,
    "Availability": availability_page,
    "Place Order": order_page,
}


def main() -> None:
    st.set_page_config(page_title="Inventory Store", layout="wide")
    client = get_client()

    active = session()
    if active is None:
        login_page(client)
        return

    page = sidebar(client, active)
    PAGE_HANDLERS[page](client, active)


# `streamlit run client/user_app.py` executes this file as __main__, so this
# is the real entry point. Guarded rather than called at module scope so the
# module can be imported by a test without building a UI or opening a
# channel.
if __name__ == "__main__":
    main()
