"""The Streamlit client's gRPC transport.

No browser and no server: every test drives a fake stub, which is possible
precisely because ``client/grpc_client.py`` carries no Streamlit import. What
is pinned down is the boundary -- what goes onto the wire for each page, and
what each gRPC status becomes by the time the UI has to render it.

The architectural assertions at the bottom are the ones that matter for the
project's constraints. The client must reach the database only through the
application server and must never hold an address for the LLM node, so this
package may not import SQLAlchemy, the storage layer or the LLM client, and
may not know that port 50052 exists. All of it is checked against the source
rather than taken on trust.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct

from client import grpc_client as transport
from client.grpc_client import (
    AI_OPERATIONS,
    AuthenticationError,
    ClientError,
    ConflictError,
    InvalidRequestError,
    InventoryClient,
    NotFoundError,
    PreconditionError,
    ServerError,
    TransportError,
    TransportTimeout,
    get_server_target,
    restore_whole_numbers,
    to_struct,
)
from proto import client_pb2

TOKEN = "session-token"


class FakeRpcError(grpc.RpcError):
    """A gRPC failure carrying the status a real one would."""

    def __init__(self, code: grpc.StatusCode, details: str = "boom") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


def data_item(record_id: int, payload: dict[str, Any]) -> client_pb2.DataItem:
    item = client_pb2.DataItem(id=str(record_id))
    item.data.update(payload)
    return item


class FakeStub:
    """Records every request and answers or fails on cue."""

    def __init__(
        self,
        *,
        raises: BaseException | None = None,
        items: list[client_pb2.DataItem] | None = None,
        message: str = "Done.",
        data: dict[str, Any] | None = None,
    ) -> None:
        self.raises = raises
        self.items = items or []
        self.message = message
        self.data = data or {}
        self.calls: list[tuple[str, Any, float | None]] = []

    def _record(self, name: str, request: Any, timeout: float | None) -> None:
        self.calls.append((name, request, timeout))
        if self.raises is not None:
            raise self.raises

    def login(self, request: Any, timeout: float | None = None) -> Any:
        self._record("login", request, timeout)
        response = client_pb2.LoginResponse(
            success=True,
            session_token=TOKEN,
            message="Login successful.",
            user=client_pb2.UserInfo(
                id=1,
                username=request.username,
                email="ops@example.com",
                role="manager",
                is_active=True,
            ),
        )
        return response

    def logout(self, request: Any, timeout: float | None = None) -> Any:
        self._record("logout", request, timeout)
        return client_pb2.StatusResponse(success=True, message="Logged out.")

    def post(self, request: Any, timeout: float | None = None) -> Any:
        self._record("post", request, timeout)
        return client_pb2.StatusResponse(success=True, message=self.message)

    def get(self, request: Any, timeout: float | None = None) -> Any:
        self._record("get", request, timeout)
        return client_pb2.GetResponse(success=True, message="OK", items=self.items)

    def processBusinessRequest(self, request: Any, timeout: float | None = None) -> Any:
        self._record("business", request, timeout)
        response = client_pb2.BusinessResponse(
            success=True, message="Advisory answer.", request_id="REQ-1"
        )
        response.data.update(self.data)
        return response

    @property
    def last(self) -> Any:
        return self.calls[-1][1]

    @property
    def last_timeout(self) -> float | None:
        return self.calls[-1][2]

    def resources(self) -> list[str]:
        return [request.resource for _, request, _ in self.calls if hasattr(request, "resource")]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("INVENTORY_GRPC_HOST", "INVENTORY_GRPC_PORT"):
        monkeypatch.delenv(name, raising=False)


def build(**kwargs: Any) -> tuple[InventoryClient, FakeStub]:
    stub = FakeStub(**kwargs)
    return InventoryClient(stub=stub), stub


def payload_of(request: Any) -> dict[str, Any]:
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(request.payload)


def params_of(request: Any) -> dict[str, Any]:
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(request.params)


# --- configuration ---------------------------------------------------------
def test_the_default_target_is_the_application_server_on_50051() -> None:
    assert get_server_target() == "127.0.0.1:50051"


def test_the_target_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVENTORY_GRPC_HOST", "10.0.0.4")
    monkeypatch.setenv("INVENTORY_GRPC_PORT", "6000")

    assert get_server_target() == "10.0.0.4:6000"


def test_an_unusable_port_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVENTORY_GRPC_PORT", "not-a-port")

    with pytest.raises(ValueError, match="INVENTORY_GRPC_PORT"):
        get_server_target()


def test_constructing_a_client_does_not_require_the_server() -> None:
    """Channels connect lazily, so the login page can render during an outage."""
    client = InventoryClient("127.0.0.1:1")
    try:
        assert client.target == "127.0.0.1:1"
    finally:
        client.close()


# --- struct conversion -----------------------------------------------------
def test_blank_fields_are_dropped_rather_than_sent_as_empty_strings() -> None:
    """An untouched Streamlit text input returns "", which is not a value."""
    struct = to_struct({"sku": "SKU-1", "description": "", "name": None})

    assert set(struct.keys()) == {"sku"}


@pytest.mark.parametrize("value", [False, 0])
def test_falsey_values_that_are_real_values_survive(value: Any) -> None:
    struct = to_struct({"is_active": value})

    assert "is_active" in struct


def test_money_crosses_as_an_exact_string() -> None:
    """A Struct stores numbers as doubles, so prices must not become one."""
    struct = to_struct({"unit_price": "1234567.89"})

    assert struct["unit_price"] == "1234567.89"
    assert isinstance(struct["unit_price"], str)


def test_an_empty_payload_is_allowed() -> None:
    assert to_struct(None) == Struct()


# --- session ---------------------------------------------------------------
def test_login_sends_the_credentials_and_returns_a_session() -> None:
    client, stub = build()

    session = client.login("ops", "secret")

    assert stub.last.username == "ops"
    assert stub.last.password == "secret"
    assert session.token == TOKEN
    assert session.username == "ops"
    assert session.role == "manager"


def test_the_password_is_not_kept_on_the_session() -> None:
    """The credential is used once; only the token is carried afterwards."""
    client, _ = build()

    session = client.login("ops", "hunter2")

    assert "hunter2" not in repr(vars(session) if hasattr(session, "__dict__") else {})
    for slot in session.__slots__:
        assert getattr(session, slot) != "hunter2"
    assert not hasattr(session, "password")


def test_logout_revokes_the_session_on_the_server() -> None:
    client, stub = build()

    client.logout(TOKEN)

    assert stub.calls[-1][0] == "logout"
    assert stub.last.session_token == TOKEN


def test_bad_credentials_surface_as_an_authentication_error() -> None:
    client, _ = build(raises=FakeRpcError(grpc.StatusCode.UNAUTHENTICATED, "denied"))

    with pytest.raises(AuthenticationError):
        client.login("ops", "wrong")


# --- every page sends its token and its resource ---------------------------
def test_every_call_carries_the_session_token() -> None:
    client, stub = build()

    client.list_products(TOKEN)
    client.list_warehouses(TOKEN)
    client.list_suppliers(TOKEN)
    client.create_product(TOKEN, sku="SKU-1")

    for _, request, _ in stub.calls:
        assert request.session_token == TOKEN


@pytest.mark.parametrize(
    ("action", "kwargs", "resource"),
    [
        ("list_products", {}, "products"),
        ("create_product", {"sku": "SKU-1"}, "product.create"),
        ("update_product", {"sku_key": "SKU-1"}, "product.update"),
        ("list_warehouses", {}, "warehouses"),
        ("create_warehouse", {"code": "WH-1"}, "warehouse.create"),
        ("list_warehouse_inventory", {"warehouse_code": "WH-1"}, "inventory.warehouse"),
        ("list_low_stock", {}, "inventory.low_stock"),
        ("set_initial_stock", {"sku": "SKU-1"}, "inventory.set_initial_stock"),
        ("add_stock", {"sku": "SKU-1"}, "inventory.add_stock"),
        ("list_suppliers", {}, "suppliers"),
        ("create_supplier", {"name": "Acme"}, "supplier.create"),
        ("associate_supplier_product", {"sku": "SKU-1"}, "supplier.associate_product"),
    ],
)
def test_each_helper_names_the_resource_the_server_registered(
    action: str, kwargs: dict[str, Any], resource: str
) -> None:
    """These strings are the wire contract; a typo is an INVALID_ARGUMENT."""
    client, stub = build()

    getattr(client, action)(TOKEN, **kwargs)

    assert stub.resources() == [resource]


def test_a_get_returns_plain_rows_with_no_protobuf_left() -> None:
    client, _ = build(
        items=[
            data_item(1, {"sku": "SKU-1", "name": "Widget", "unit_price": "24.99"}),
            data_item(2, {"sku": "SKU-2", "name": "Gadget", "unit_price": "5.00"}),
        ]
    )

    rows = client.list_products(TOKEN)

    assert rows == [
        {"sku": "SKU-1", "name": "Widget", "unit_price": "24.99"},
        {"sku": "SKU-2", "name": "Gadget", "unit_price": "5.00"},
    ]


def test_an_empty_result_is_an_empty_list_not_an_error() -> None:
    client, _ = build(items=[])

    assert client.list_products(TOKEN) == []


def test_a_post_returns_the_servers_confirmation() -> None:
    client, _ = build(message="Product 'SKU-1' created.")

    assert client.create_product(TOKEN, sku="SKU-1") == "Product 'SKU-1' created."


def test_the_low_stock_flag_is_the_servers_to_decide() -> None:
    """The client reads `is_low_stock`; it never compares the two numbers."""
    client, _ = build(
        items=[data_item(1, {"sku": "SKU-1", "quantity": 4, "reorder_point": 25, "is_low_stock": True})]
    )

    rows = client.list_low_stock(TOKEN)

    assert rows[0]["is_low_stock"] is True


# --- orders ----------------------------------------------------------------
def test_an_order_sends_the_warehouse_and_every_line() -> None:
    client, stub = build()

    client.place_order(
        TOKEN, "WH-1", [{"sku": "SKU-1", "quantity": 2}, {"sku": "SKU-2", "quantity": 5}]
    )

    sent = payload_of(stub.last)
    assert stub.last.resource == "order.place"
    assert sent["warehouse_code"] == "WH-1"
    assert [line["sku"] for line in sent["items"]] == ["SKU-1", "SKU-2"]
    assert [line["quantity"] for line in sent["items"]] == [2, 5]


def test_insufficient_stock_is_its_own_error_not_bad_input() -> None:
    """FAILED_PRECONDITION: the request was correct, the warehouse was short.

    Kept distinct from InvalidRequestError so the UI can say "not enough
    stock" rather than "bad input" -- the same order may well succeed once
    the warehouse is restocked.
    """
    client, _ = build(
        raises=FakeRpcError(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Insufficient stock for 'SKU-1': requested 10, available 3.",
        )
    )

    with pytest.raises(PreconditionError, match="Insufficient stock"):
        client.place_order(TOKEN, "WH-1", [{"sku": "SKU-1", "quantity": 10}])


def test_the_client_performs_no_stock_arithmetic() -> None:
    """Placing an order sends the request and nothing else."""
    client, stub = build(message="Order ORD-1 placed.")

    result = client.place_order(TOKEN, "WH-1", [{"sku": "SKU-1", "quantity": 2}])

    assert result == "Order ORD-1 placed."
    assert len(stub.calls) == 1


# --- AI insights -----------------------------------------------------------
def test_the_three_operations_match_the_servers_vocabulary() -> None:
    from app.api.business_service import BUSINESS_ACTIONS

    assert set(AI_OPERATIONS) == set(BUSINESS_ACTIONS)


@pytest.mark.parametrize("operation", AI_OPERATIONS)
def test_an_ai_request_goes_to_the_application_server(operation: str) -> None:
    """Never to :50052 -- the app server owns both the data and the LLM hop."""
    client, stub = build(data={"answer": "Looks fine.", "model": "phi3:mini"})

    client.business_request(TOKEN, operation, {"sku": "SKU-1"})

    assert stub.calls[-1][0] == "business"
    assert stub.last.operation == operation
    assert stub.last.session_token == TOKEN


def test_the_ai_payload_carries_the_operators_selections() -> None:
    client, stub = build(data={"answer": "..."})

    client.business_request(
        TOKEN,
        "DEMAND_FORECAST",
        {"sku": "SKU-1", "warehouse_code": "WH-1", "horizon_days": 14},
    )

    assert payload_of(stub.last) == {
        "sku": "SKU-1",
        "warehouse_code": "WH-1",
        "horizon_days": 14,
    }


def test_the_answer_and_its_evidence_both_come_back() -> None:
    client, _ = build(
        data={
            "answer": "Order 40 units.",
            "model": "phi3:mini",
            "advisory": True,
            "query_type": "REORDER_RECOMMENDATION",
            "context": {"baseline_reorder_quantity": 40},
        }
    )

    result = client.business_request(TOKEN, "REORDER_RECOMMENDATION")

    assert result["answer"] == "Order 40 units."
    assert result["advisory"] is True
    assert result["context"]["baseline_reorder_quantity"] == 40


def test_a_generated_request_id_is_preserved() -> None:
    client, _ = build(data={"answer": "..."})

    assert client.business_request(TOKEN, "INVENTORY_SUMMARY")["request_id"] == "REQ-1"


def test_an_ai_request_gets_a_longer_deadline_than_ordinary_crud() -> None:
    """Generation can take a minute; a 15s CRUD deadline would abort it."""
    client, stub = build(data={"answer": "..."})

    client.list_products(TOKEN)
    crud_timeout = stub.last_timeout
    client.business_request(TOKEN, "INVENTORY_SUMMARY")

    assert stub.last_timeout > crud_timeout
    assert stub.last_timeout == transport.BUSINESS_TIMEOUT_SECONDS


# --- error mapping ---------------------------------------------------------
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (grpc.StatusCode.UNAUTHENTICATED, AuthenticationError),
        (grpc.StatusCode.INVALID_ARGUMENT, InvalidRequestError),
        (grpc.StatusCode.NOT_FOUND, NotFoundError),
        (grpc.StatusCode.ALREADY_EXISTS, ConflictError),
        (grpc.StatusCode.FAILED_PRECONDITION, PreconditionError),
        (grpc.StatusCode.DEADLINE_EXCEEDED, TransportTimeout),
        (grpc.StatusCode.UNAVAILABLE, TransportError),
        (grpc.StatusCode.INTERNAL, ServerError),
    ],
)
def test_each_status_becomes_its_own_error(
    code: grpc.StatusCode, expected: type[ClientError]
) -> None:
    client, _ = build(raises=FakeRpcError(code))

    with pytest.raises(expected) as caught:
        client.list_products(TOKEN)

    assert caught.value.code == code


def test_an_unmapped_status_is_still_catchable() -> None:
    client, _ = build(raises=FakeRpcError(grpc.StatusCode.RESOURCE_EXHAUSTED))

    with pytest.raises(ServerError):
        client.list_products(TOKEN)


def test_every_error_shares_one_base() -> None:
    for error in (
        AuthenticationError,
        InvalidRequestError,
        NotFoundError,
        ConflictError,
        PreconditionError,
        TransportTimeout,
        TransportError,
        ServerError,
    ):
        assert issubclass(error, ClientError)


def test_an_outage_is_reported_in_words_an_operator_can_act_on() -> None:
    """The raw detail is a channel diagnostic; the fallback says what to do."""
    client, _ = build(
        raises=FakeRpcError(
            grpc.StatusCode.UNAVAILABLE, "failed to connect to all addresses"
        )
    )

    with pytest.raises(TransportError, match="not reachable"):
        client.list_products(TOKEN)


def test_no_grpc_exception_reaches_the_caller() -> None:
    client, _ = build(raises=FakeRpcError(grpc.StatusCode.NOT_FOUND))

    try:
        client.list_products(TOKEN)
    except ClientError:
        pass
    except grpc.RpcError:  # pragma: no cover - the failure this guards against
        pytest.fail("a raw grpc.RpcError escaped the transport layer")


# --- the architectural boundary --------------------------------------------
def client_sources() -> list[Path]:
    return sorted(Path(transport.__file__).resolve().parent.rglob("*.py"))


def imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_client_never_touches_a_database() -> None:
    """It has no path to SQLite, by construction rather than by convention."""
    forbidden = {"sqlalchemy", "sqlite3"}
    for path in client_sources():
        for module in imported_modules(path.read_text(encoding="utf-8")):
            assert module.split(".")[0] not in forbidden, f"{path.name}: {module}"


def test_the_client_never_imports_the_application_storage_layer() -> None:
    for path in client_sources():
        for module in imported_modules(path.read_text(encoding="utf-8")):
            assert not module.startswith("app.storage"), f"{path.name}: {module}"
            assert not module.startswith("app.domain"), f"{path.name}: {module}"


def test_the_client_never_reaches_the_llm_node_or_ollama() -> None:
    """AI answers arrive via the application server, which owns that hop."""
    forbidden = {"ollama", "httpx", "requests"}
    for path in client_sources():
        for module in imported_modules(path.read_text(encoding="utf-8")):
            root = module.split(".")[0]
            assert root not in forbidden, f"{path.name}: {module}"
            assert not module.startswith("llm"), f"{path.name}: {module}"
            assert not module.startswith("app.api.llm_client"), f"{path.name}: {module}"


def code_constants(source: str) -> list[Any]:
    """Literals in real code, with docstrings excluded.

    These modules *discuss* the LLM port in prose precisely because they must
    never dial it, so a raw text search would flag the documentation that
    promises the opposite. Comments never enter the syntax tree; docstrings
    do, and are skipped explicitly.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and id(node) not in docstrings
    ]


@pytest.mark.parametrize("port", [50052, 11434])
def test_the_client_never_names_the_llm_or_ollama_port(port: int) -> None:
    """It talks to 50051 and knows no other address."""
    for path in client_sources():
        for constant in code_constants(path.read_text(encoding="utf-8")):
            if isinstance(constant, str):
                assert str(port) not in constant, f"{path.name}: {constant!r}"
            elif isinstance(constant, int) and not isinstance(constant, bool):
                assert constant != port, path.name


def test_the_transport_layer_needs_no_browser() -> None:
    """Streamlit lives in client/app.py alone, which is what makes this file possible."""
    source = Path(transport.__file__).read_text(encoding="utf-8")

    assert "streamlit" not in imported_modules(source)


# --- the UI's wiring, checked without rendering a widget -------------------
#
# `client/app.py` is importable because it calls main() only under
# __main__, so its tables can be asserted directly. Nothing below builds a
# widget, opens a channel or needs a browser.
def test_every_page_has_a_handler() -> None:
    from client import app

    assert set(app.PAGE_HANDLERS) == set(app.PAGES)


def test_every_ai_operation_has_a_label() -> None:
    from client import app

    assert set(app.AI_LABELS) == set(app.AI_OPERATIONS)


def test_the_ui_reaches_the_server_only_through_the_transport_layer() -> None:
    """No page builds its own channel or stub."""
    from client import app

    modules = imported_modules(Path(app.__file__).read_text(encoding="utf-8"))

    assert "client.grpc_client" in modules
    assert "grpc" not in modules
    assert not any(module.startswith("proto") for module in modules)


def test_streamlit_is_confined_to_the_ui_module() -> None:
    from client import app

    ui = Path(app.__file__).resolve()
    for path in client_sources():
        if path == ui:
            continue
        assert "streamlit" not in imported_modules(path.read_text(encoding="utf-8")), path.name


# --- undoing the Struct round trip ----------------------------------------
#
# A Struct has one numeric type, so every integer the server sent arrives as
# a double. Left alone, a stock table reads "150.0 units" and a reorder point
# reads "40.0" for columns the schema stores as INTEGER.
def test_whole_numbers_survive_the_struct_round_trip() -> None:
    client, _ = build(
        items=[data_item(1, {"sku": "SKU-1", "quantity": 150, "reorder_point": 40})]
    )

    row = client.list_products(TOKEN)[0]

    assert row["quantity"] == 150
    assert isinstance(row["quantity"], int)
    assert isinstance(row["reorder_point"], int)


def test_a_genuine_fraction_is_left_alone() -> None:
    assert restore_whole_numbers(2.5) == 2.5


def test_a_flag_never_becomes_a_number() -> None:
    """bool subclasses int, so True.is_integer() would turn a flag into 1."""
    restored = restore_whole_numbers({"is_low_stock": True, "is_active": False})

    assert restored["is_low_stock"] is True
    assert restored["is_active"] is False


def test_money_is_untouched_because_it_never_arrives_as_a_number() -> None:
    client, _ = build(items=[data_item(1, {"unit_price": "24.99"})])

    assert client.list_products(TOKEN)[0]["unit_price"] == "24.99"


def test_nested_context_is_restored_too() -> None:
    """The AI reply carries its evidence in a nested Struct."""
    client, _ = build(
        data={
            "answer": "...",
            "context": {
                "stock": {"quantity": 8, "reorder_point": 25},
                "history": [{"units": 12}, {"units": 30}],
            },
        }
    )

    context = client.business_request(TOKEN, "REORDER_RECOMMENDATION")["context"]

    assert context["stock"] == {"quantity": 8, "reorder_point": 25}
    assert [row["units"] for row in context["history"]] == [12, 30]


def test_useful_columns_come_first_in_a_table() -> None:
    """A Struct is a map, so MessageToDict key order is arbitrary."""
    from client.app import order_columns

    rows = [{"updated_at": "t", "id": 1, "quantity": 149, "sku": "SKU-1"}]

    assert list(order_columns(rows)[0]) == ["id", "sku", "quantity", "updated_at"]


def test_an_unknown_column_is_kept_rather_than_dropped() -> None:
    """A column added on the server must still appear."""
    from client.app import order_columns

    rows = [{"sku": "SKU-1", "brand_new_field": 7}]

    ordered = order_columns(rows)[0]
    assert ordered["brand_new_field"] == 7
    assert list(ordered) == ["sku", "brand_new_field"]


def test_ordering_tolerates_rows_with_different_keys() -> None:
    from client.app import order_columns

    rows = [{"sku": "SKU-1", "quantity": 5}, {"sku": "SKU-2"}]

    assert order_columns(rows) == [{"sku": "SKU-1", "quantity": 5}, {"sku": "SKU-2"}]
