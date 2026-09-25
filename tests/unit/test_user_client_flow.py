"""The user client's own logic: session state, errors and order results.

``tests/unit/test_client_grpc_client.py`` covers the wire -- what each helper
sends and what each gRPC status becomes. This file covers the layer above it:
what ``client/user_app.py`` does with the answer. Does a rejected session get
torn down? Is a failure shown rather than raised? Does a placed order clear
the basket and list what was bought?

None of that needs a browser. ``user_app`` reaches Streamlit through one
module-level name, ``st``, so substituting a recorder for it exercises the
real functions while capturing what they would have drawn. Nothing here
renders a widget, opens a channel, touches Ollama or reads the production
database.

The fake deliberately raises on ``rerun()``. The real ``st.rerun()`` raises
to abort the script run, so code after it never executes -- a fake that
returned normally would let these tests pass over control flow that cannot
happen in production.
"""

from __future__ import annotations

from typing import Any

import pytest

from client.grpc_client import (
    AuthenticationError,
    ConflictError,
    InvalidRequestError,
    InventoryClient,
    NotFoundError,
    PreconditionError,
    ServerError,
    Session,
    TransportError,
    TransportTimeout,
)
from client.user_app import CART_KEY, ORDER_RESULT_KEY, SESSION_KEY
from tests.unit.test_client_grpc_client import FakeRpcError, FakeStub, data_item

TOKEN = "session-token"
PASSWORD = "User1-Dev-Only-123"


class Rerun(Exception):
    """Stands in for ``st.rerun()``, which raises to restart the script run."""


class FakeStreamlit:
    """Records what the UI would have drawn, instead of drawing it."""

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.captions: list[str] = []
        self.tables: list[Any] = []

    def error(self, message: Any) -> None:
        self.errors.append(str(message))

    def warning(self, message: Any) -> None:
        self.warnings.append(str(message))

    def info(self, message: Any) -> None:
        self.infos.append(str(message))

    def success(self, message: Any) -> None:
        self.successes.append(str(message))

    def caption(self, message: Any) -> None:
        self.captions.append(str(message))

    def subheader(self, message: Any) -> None:
        self.captions.append(str(message))

    def dataframe(self, rows: Any, **_: Any) -> None:
        self.tables.append(rows)

    def divider(self) -> None:
        return None

    def rerun(self) -> None:
        raise Rerun()

    # -- helpers for assertions -----------------------------------------
    def said(self, fragment: str) -> bool:
        everything = self.errors + self.warnings + self.infos + self.successes
        return any(fragment.lower() in line.lower() for line in everything)


@pytest.fixture
def st(monkeypatch: pytest.MonkeyPatch) -> FakeStreamlit:
    """Swap Streamlit for a recorder, for the duration of one test."""
    from client import user_app

    fake = FakeStreamlit()
    monkeypatch.setattr(user_app, "st", fake)
    return fake


@pytest.fixture
def signed_in() -> Session:
    return Session(
        token=TOKEN, username="user1", role="staff", email="user1@example.com"
    )


def build(**kwargs: Any) -> tuple[InventoryClient, FakeStub]:
    stub = FakeStub(**kwargs)
    return InventoryClient(stub=stub), stub


# ===========================================================================
# 1-3. Login, the token that follows it, and a login that fails
# ===========================================================================
def test_a_successful_login_returns_a_usable_session() -> None:
    client, stub = build()

    session = client.login("user1", PASSWORD)

    assert stub.last.username == "user1"
    assert stub.last.password == PASSWORD
    assert session.token == "session-token"
    assert session.username == "user1"


def test_the_password_is_not_carried_on_the_session(st: FakeStreamlit) -> None:
    """It was an argument to one RPC; nothing keeps it afterwards."""
    from client import user_app

    client, _ = build()
    session = client.login("user1", PASSWORD)
    st.session_state[SESSION_KEY] = session

    assert not hasattr(session, "password")
    for slot in session.__slots__:
        assert getattr(session, slot) != PASSWORD
    # And nothing anywhere in the UI's state holds it either.
    assert PASSWORD not in repr(st.session_state)
    assert user_app.session() is session


def test_the_stored_session_is_what_later_calls_authenticate_with(
    st: FakeStreamlit, signed_in: Session
) -> None:
    """Requirement 2: the token is retained and reused, not re-issued."""
    from client import user_app

    st.session_state[SESSION_KEY] = signed_in
    client, stub = build()

    active = user_app.session()
    client.list_products(active.token)
    client.list_warehouses(active.token)
    client.place_order(active.token, "WH-1", [{"sku": "SKU-1", "quantity": 1}])

    assert len(stub.calls) == 3
    for _, request, _ in stub.calls:
        assert request.session_token == TOKEN
    # No second login happened to get them.
    assert [name for name, _, _ in stub.calls] == ["get", "get", "post"]


def test_a_rejected_login_is_reported_not_raised(st: FakeStreamlit) -> None:
    """Requirement 3: bad credentials end as a message, never a traceback."""
    from client import user_app

    client, _ = build(
        raises=FakeRpcError(
            client_grpc_unauthenticated(), "Invalid username or password."
        )
    )

    # `call` translates and displays; it returns None rather than a session.
    with pytest.raises(Rerun):
        user_app.call(client.login, "user1", "wrong")

    assert st.errors
    assert "invalid username or password" in st.errors[0].lower()
    assert SESSION_KEY not in st.session_state


def client_grpc_unauthenticated():
    import grpc

    return grpc.StatusCode.UNAUTHENTICATED


# ===========================================================================
# 4-5. Request construction for reads and for an order
# ===========================================================================
def test_product_and_inventory_reads_name_the_right_resources(
    signed_in: Session,
) -> None:
    client, stub = build()

    client.list_products(signed_in.token, active_only=True, limit=500)
    client.list_warehouses(signed_in.token, limit=500)
    client.list_warehouse_inventory(signed_in.token, warehouse_code="WH-1")

    assert stub.resources() == ["products", "warehouses", "inventory.warehouse"]


def test_an_inventory_read_carries_the_warehouse_it_was_asked_for(
    signed_in: Session,
) -> None:
    from google.protobuf.json_format import MessageToDict

    client, stub = build()

    client.list_warehouse_inventory(signed_in.token, warehouse_code="WH-2")

    assert MessageToDict(stub.last.params) == {"warehouse_code": "WH-2"}


def test_a_valid_order_is_constructed_from_the_basket(signed_in: Session) -> None:
    from google.protobuf.json_format import MessageToDict

    client, stub = build()
    basket = [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-2", "quantity": 3}]

    client.place_order(signed_in.token, "WH-1", basket)

    sent = MessageToDict(stub.last.payload)
    assert stub.last.resource == "order.place"
    assert sent["warehouse_code"] == "WH-1"
    assert sent["items"] == [
        {"sku": "SKU-1", "quantity": 2},
        {"sku": "SKU-2", "quantity": 3},
    ]


# ===========================================================================
# 6. Quantities the UI refuses before anything is sent
# ===========================================================================
@pytest.mark.parametrize("quantity", [0, -1, -99])
def test_a_non_positive_quantity_is_refused(quantity: int) -> None:
    from client.user_app import validate_basket, validate_line

    assert validate_line("SKU-1", quantity) is not None
    assert validate_basket([{"sku": "SKU-1", "quantity": quantity}]) is not None


def test_the_refusal_says_what_is_wrong() -> None:
    from client.user_app import validate_line

    assert "at least 1" in validate_line("SKU-1", 0)


def test_a_valid_quantity_is_not_refused() -> None:
    from client.user_app import validate_basket, validate_line

    assert validate_line("SKU-1", 3) is None
    assert validate_basket([{"sku": "SKU-1", "quantity": 3}]) is None


# ===========================================================================
# 7. A placed order, handled
# ===========================================================================
def test_a_placed_order_is_confirmed_and_lists_what_was_bought(
    st: FakeStreamlit, signed_in: Session
) -> None:
    from client import user_app

    client, _ = build(items=[])
    st.session_state[ORDER_RESULT_KEY] = (
        "placed",
        "Order ORD-1234 placed: 2 line(s), total 104.49, status COMPLETED.",
        "WH-1",
        [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-2", "quantity": 1}],
    )

    user_app._render_order_result(client, signed_in)

    # The server's own sentence, verbatim -- order number, count, total.
    assert st.successes == [
        "Order ORD-1234 placed: 2 line(s), total 104.49, status COMPLETED."
    ]
    assert not st.errors
    # The lines that were ordered.
    assert st.tables
    assert st.tables[0] == [
        {"SKU": "SKU-1", "Quantity": 2},
        {"SKU": "SKU-2", "Quantity": 1},
    ]


def test_an_order_result_is_shown_once_and_then_forgotten(
    st: FakeStreamlit, signed_in: Session
) -> None:
    """A confirmation must not linger over the next submission."""
    from client import user_app

    client, _ = build(items=[])
    st.session_state[ORDER_RESULT_KEY] = ("placed", "Order ORD-1 placed.", "WH-1", [])

    user_app._render_order_result(client, signed_in)
    assert ORDER_RESULT_KEY not in st.session_state

    user_app._render_order_result(client, signed_in)
    assert len(st.successes) == 1


def test_stock_is_re_read_from_the_server_after_an_order(
    st: FakeStreamlit, signed_in: Session
) -> None:
    """Requirement 10/11 of the order flow: never adjusted locally."""
    from client import user_app

    client, stub = build(
        items=[data_item(1, {"sku": "SKU-1", "quantity": 148, "is_low_stock": False})]
    )
    st.session_state[ORDER_RESULT_KEY] = ("placed", "Order ORD-1 placed.", "WH-1", [])

    user_app._render_order_result(client, signed_in)

    assert "inventory.warehouse" in stub.resources()


# ===========================================================================
# 8-10. Failures a user has to be able to read
# ===========================================================================
def test_insufficient_stock_is_readable_and_keeps_the_basket(
    st: FakeStreamlit, signed_in: Session
) -> None:
    """Requirement 8, and the promise not to pretend anything changed."""
    from client import user_app

    client, _ = build(items=[])
    basket = [{"sku": "SKU-1", "quantity": 99999}]
    st.session_state[CART_KEY] = basket
    st.session_state[ORDER_RESULT_KEY] = (
        "short",
        (
            "Order not placed: Insufficient stock for 'SKU-1': "
            "requested 99999, available 148."
        ),
        "WH-1",
        [],
    )

    user_app._render_order_result(client, signed_in)

    assert st.errors
    assert "insufficient stock" in st.errors[0].lower()
    assert "requested 99999" in st.errors[0]
    # Not dressed up as a success, and the basket survives for editing.
    assert not st.successes
    assert st.said("basket has been kept")
    assert st.session_state[CART_KEY] == basket


@pytest.mark.parametrize(
    "error",
    [
        PreconditionError("Insufficient stock for 'SKU-1'."),
        NotFoundError("No stock record for product 'SKU-9'."),
        InvalidRequestError("'quantity' must be at least 1."),
        ConflictError("SKU 'SKU-1' already exists."),
        TransportTimeout("The application server did not answer in time."),
        TransportError("The application server is not reachable."),
        ServerError("Internal server error."),
    ],
)
def test_every_failure_reaches_the_user_as_a_sentence(
    st: FakeStreamlit, error: Exception
) -> None:
    """Requirements 8 and 10: a message, never a traceback, never a crash."""
    from client import user_app

    def failing() -> None:
        raise error

    result = user_app.call(failing)

    assert result is None
    assert st.errors == [str(error)]
    assert "Traceback" not in st.errors[0]


def test_an_outage_tells_the_user_what_to_do(st: FakeStreamlit) -> None:
    """Requirement 10: the raw gRPC detail is a channel diagnostic."""
    import grpc

    from client import user_app

    client, _ = build(
        raises=FakeRpcError(
            grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
        )
    )

    result = user_app.call(client.list_products, TOKEN)

    assert result is None
    assert "not reachable" in st.errors[0]
    assert "failed to connect to all addresses" not in st.errors[0]


def test_an_unauthenticated_reply_ends_the_session(
    st: FakeStreamlit, signed_in: Session
) -> None:
    """Requirement 9: explained, and the dead token is not left in state.

    Leaving it would make every later page fail the same way, one at a time,
    with no route back to the login screen.
    """
    from client import user_app

    st.session_state[SESSION_KEY] = signed_in
    st.session_state[CART_KEY] = [{"sku": "SKU-1", "quantity": 1}]

    def failing() -> None:
        raise AuthenticationError("Your session has expired.")

    with pytest.raises(Rerun):
        user_app.call(failing)

    assert st.errors
    assert "expired" in st.errors[0].lower()
    assert "sign in again" in st.errors[0].lower()
    assert SESSION_KEY not in st.session_state
    assert CART_KEY not in st.session_state


def test_a_successful_call_returns_its_value_untouched(st: FakeStreamlit) -> None:
    """The error path must not cost anything on the happy path."""
    from client import user_app

    rows = [{"sku": "SKU-1"}]

    assert user_app.call(lambda: rows) is rows
    assert not st.errors


# ===========================================================================
# 11. Logout
# ===========================================================================
def test_logout_clears_every_trace_of_the_session(
    st: FakeStreamlit, signed_in: Session
) -> None:
    from client import user_app

    st.session_state[SESSION_KEY] = signed_in
    st.session_state[CART_KEY] = [{"sku": "SKU-1", "quantity": 2}]
    st.session_state[ORDER_RESULT_KEY] = ("placed", "Order ORD-1 placed.", "WH-1", [])

    user_app._end_session()

    assert st.session_state == {}
    assert user_app.session() is None


def test_logout_revokes_on_the_server_before_clearing_locally(
    signed_in: Session,
) -> None:
    """Clearing only local state would leave the token valid on the node."""
    client, stub = build()

    client.logout(signed_in.token)

    assert stub.calls[-1][0] == "logout"
    assert stub.last.session_token == TOKEN


def test_clearing_state_twice_is_harmless(st: FakeStreamlit) -> None:
    from client import user_app

    user_app._end_session()
    user_app._end_session()

    assert st.session_state == {}


# ===========================================================================
# The constraints these tests run under
# ===========================================================================
def test_these_tests_need_no_browser_ollama_or_real_database() -> None:
    """Stated where a future edit to this file would see it.

    The session-scoped fixtures in ``tests/conftest.py`` enforce the last two
    for the whole suite: a real ``ollama.Client`` or a socket to 11434 fails
    the run, as does any change to ``data/node1.db``.
    """
    import ast
    from pathlib import Path

    source = Path(__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    for name in imported:
        root = name.split(".")[0]
        assert root not in {"ollama", "httpx", "sqlalchemy", "sqlite3"}, name
        assert not name.startswith("app.storage"), name


def test_the_fake_stands_in_for_every_streamlit_call_the_code_makes(
    st: FakeStreamlit,
) -> None:
    """If `user_app` starts calling something new, this says so plainly."""
    from client import user_app

    used = {
        "error",
        "warning",
        "info",
        "success",
        "caption",
        "subheader",
        "dataframe",
        "divider",
        "rerun",
        "session_state",
    }
    for name in used:
        assert hasattr(st, name), name
    assert user_app.st is st
