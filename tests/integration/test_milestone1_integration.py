"""Milestone 1 integration suite: the six graded scenarios, end to end.

Every test here drives **real gRPC servers over real channels**. Nothing is
patched inside the application server: it is started with
``app.server.create_server``, bound to an ephemeral port and a temporary
SQLite file, and reached through a generated stub exactly as the Streamlit
client reaches it. The only thing ever faked is the model runtime, and it is
faked at the far edge of the system.

    stub --gRPC--> application server --gRPC--> fake LLM node
                         |
                   temporary SQLite

**What makes this file different from the other integration modules.**
``test_grpc_business_request.py`` runs the *real* LLM servicer with a fake
Ollama behind it, which proves the prompt layer renders what the database
holds. This file replaces the LLM node itself with
:class:`RecordingLLMService`, a deterministic fake that records the
``GetLLMAnswerRequest`` it was handed. That is what lets scenario 5 assert
the ``query_type`` and ``context`` **as they crossed the wire** rather than
inferring them from a prompt built downstream. The two are complementary:
one checks the far side's behaviour, this one checks what the application
server actually sent.

**Determinism.** No test needs Ollama, a model, or the network. The fake
answers instantly and identically every time, so a run is reproducible and
the suite's duration does not depend on a CPU generating tokens. A
session-scoped autouse fixture in ``tests/conftest.py`` enforces this: it
fails the run if anything constructs a real ``ollama.Client`` or opens a
socket to port 11434 or off-loopback. Real Ollama is exercised separately by
``scripts/demo_llm_queries.py``, never by pytest.

**Database isolation.** Every test gets its own SQLite file under pytest's
``tmp_path`` via the ``session_factory`` fixture, and a second session-scoped
autouse fixture compares the size and mtime of ``data/node1.db`` across the
whole run, so a stray import of the module-level engine fails the suite
instead of quietly corrupting development data.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.llm_client import LLMClient
from app.server import create_server as create_app_server
from app.storage.repositories import (
    InventoryRepository,
    OrderItemRepository,
    OrderRepository,
    SalesHistoryRepository,
)
from proto import client_pb2, client_pb2_grpc, llm_pb2, llm_pb2_grpc
from tests.conftest import TEST_PASSWORD

ANSWER = "Stock covers roughly nine days at the current rate."
MODEL = "fake-model:test"


# ---------------------------------------------------------------------------
# A deterministic fake LLM node
# ---------------------------------------------------------------------------
class RecordingLLMService(llm_pb2_grpc.LLMServiceServicer):
    """A real gRPC servicer that answers instantly and remembers the request.

    This stands in for the whole LLM node -- servicer, prompt layer, Ollama
    adapter and model. Two properties matter:

    * **Deterministic.** The same answer every time, with no generation and
      no runtime, so the suite is reproducible and fast.
    * **Recording.** ``requests`` holds the ``GetLLMAnswerRequest`` messages
      as they arrived, which is the only place the ``query_type`` and
      ``context`` the application server *sent* can be observed directly.

    ``abort_with`` makes it fail on cue, which is how scenario 6 takes the
    model away without stopping the node or touching the network.
    """

    def __init__(
        self,
        *,
        answer: str = ANSWER,
        model: str = MODEL,
        abort_with: grpc.StatusCode | None = None,
        detail: str = "the fake LLM node was told to fail",
    ) -> None:
        self.requests: list[llm_pb2.GetLLMAnswerRequest] = []
        self._answer = answer
        self._model = model
        self._abort_with = abort_with
        self._detail = detail

    # Named for the generated base class, which takes the spelling from
    # proto/llm.proto. gRPC dispatches on this exact name.
    def getLLMAnswer(
        self, request: llm_pb2.GetLLMAnswerRequest, context: grpc.ServicerContext
    ) -> llm_pb2.GetLLMAnswerResponse:
        self.requests.append(request)
        if self._abort_with is not None:
            context.abort(self._abort_with, self._detail)
        return llm_pb2.GetLLMAnswerResponse(
            request_id=request.request_id, answer=self._answer, model=self._model
        )

    @property
    def last(self) -> llm_pb2.GetLLMAnswerRequest:
        return self.requests[-1]

    def last_context(self) -> dict[str, Any]:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(self.last.context)


def start_llm_node(service: RecordingLLMService) -> tuple[grpc.Server, int]:
    """Serve the fake on an ephemeral port, so runs never collide."""
    server = grpc.server(_single_thread_pool())
    llm_pb2_grpc.add_LLMServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def _single_thread_pool():
    from concurrent import futures

    return futures.ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def llm_service() -> RecordingLLMService:
    return RecordingLLMService()


@pytest.fixture
def nodes(
    session_factory: sessionmaker[DbSession], llm_service: RecordingLLMService
) -> Iterator[client_pb2_grpc.ClientServiceStub]:
    """Both nodes, on ephemeral ports, over real channels.

    The application server is the production object: no patching, no test
    subclass. Only its LLM target is pointed at the fake, which is the one
    seam the design already provides for exactly this purpose.
    """
    yield from _nodes(session_factory, llm_service)


def _nodes(
    session_factory: sessionmaker[DbSession], llm_service: RecordingLLMService
) -> Iterator[client_pb2_grpc.ClientServiceStub]:
    llm_server, llm_port = start_llm_node(llm_service)
    llm_client = LLMClient(f"127.0.0.1:{llm_port}", timeout=5.0)

    app_server, app_port = create_app_server(
        session_factory, host="127.0.0.1", port=0, max_workers=4, llm_client=llm_client
    )
    app_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{app_port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            yield client_pb2_grpc.ClientServiceStub(channel)
    finally:
        app_server.stop(None).wait()
        llm_client.close()
        llm_server.stop(None).wait()


@pytest.fixture
def estate(session_factory: sessionmaker[DbSession], auth_user: Any) -> dict[str, Any]:
    """A known starting position, committed before any RPC runs.

    Quantities are chosen so one order can succeed and another can be made to
    fail on its second line: SKU-A holds 50, SKU-B holds 6.
    """
    from app.storage.models import Inventory, Product, Warehouse

    with session_factory() as db:
        warehouse = Warehouse(code="WH-1", name="Main", location="Bengaluru")
        first = Product(sku="SKU-A", name="Widget", unit_price=Decimal("24.99"))
        second = Product(sku="SKU-B", name="Gadget", unit_price=Decimal("79.50"))
        db.add_all([warehouse, first, second])
        db.flush()

        db.add_all(
            [
                Inventory(
                    warehouse_id=warehouse.id,
                    product_id=first.id,
                    quantity=50,
                    reorder_point=10,
                ),
                Inventory(
                    warehouse_id=warehouse.id,
                    product_id=second.id,
                    quantity=6,
                    reorder_point=10,
                ),
            ]
        )
        db.commit()
        return {
            "warehouse_id": warehouse.id,
            "product_a": first.id,
            "product_b": second.id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def login(stub: client_pb2_grpc.ClientServiceStub, username: str) -> str:
    response = stub.login(
        client_pb2.LoginRequest(username=username, password=TEST_PASSWORD)
    )
    assert response.success is True
    return response.session_token


def post(
    stub: client_pb2_grpc.ClientServiceStub,
    token: str,
    resource: str,
    payload: dict[str, Any],
) -> client_pb2.StatusResponse:
    request = client_pb2.PostRequest(session_token=token, resource=resource)
    struct = Struct()
    struct.update(payload)
    request.payload.CopyFrom(struct)
    return stub.post(request)


def get(
    stub: client_pb2_grpc.ClientServiceStub,
    token: str,
    resource: str,
    params: dict[str, Any] | None = None,
) -> client_pb2.GetResponse:
    request = client_pb2.GetRequest(session_token=token, resource=resource)
    if params:
        struct = Struct()
        struct.update(params)
        request.params.CopyFrom(struct)
    return stub.get(request)


def ask(
    stub: client_pb2_grpc.ClientServiceStub,
    token: str,
    operation: str,
    payload: dict[str, Any] | None = None,
) -> client_pb2.BusinessResponse:
    request = client_pb2.BusinessRequest(
        session_token=token, operation=operation, request_id="req-1"
    )
    if payload:
        struct = Struct()
        struct.update(payload)
        request.payload.CopyFrom(struct)
    return stub.processBusinessRequest(request)


def stock(
    stub: client_pb2_grpc.ClientServiceStub, token: str, sku: str
) -> int:
    response = get(
        stub, token, "inventory.stock", {"warehouse_code": "WH-1", "sku": sku}
    )
    # A Struct stores numbers as doubles; the quantity is an INTEGER column.
    return int(response.items[0].data["quantity"])


def database_snapshot(session_factory: sessionmaker[DbSession]) -> dict[str, Any]:
    """Everything an order or an analysis could possibly disturb."""
    with session_factory() as db:
        return {
            "orders": len(OrderRepository(db).list()),
            "order_items": len(OrderItemRepository(db).list()),
            "sales": len(SalesHistoryRepository(db).list()),
            "stock": {
                record.product_id: record.quantity
                for record in InventoryRepository(db).list()
            },
        }


# ===========================================================================
# Scenario 1 -- authentication lifecycle
# ===========================================================================
def test_scenario_1_login_use_logout_and_the_token_dies(
    nodes: client_pb2_grpc.ClientServiceStub, estate: dict[str, Any], auth_user: Any
) -> None:
    """The full lifecycle over one real channel, in one test.

    The last step is the one that matters: logout must revoke the session on
    the *server*. If it only cleared client state, the token would keep
    working until it expired, and a copied token would outlive the logout.
    """
    stub = nodes

    # 1. login through the real service
    response = stub.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD)
    )
    assert response.success is True
    token = response.session_token
    assert token
    assert response.user.username == auth_user.username

    # 2. the token authenticates a real request
    listed = get(stub, token, "products")
    assert listed.success is True
    assert len(listed.items) == 2

    # 3. logout
    revoked = stub.logout(client_pb2.LogoutRequest(session_token=token))
    assert revoked.success is True

    # 4. the same token is now refused -- on both verbs
    for attempt in (
        lambda: get(stub, token, "products"),
        lambda: post(stub, token, "product.create", {"sku": "X", "name": "X"}),
    ):
        with pytest.raises(grpc.RpcError) as exc:
            attempt()
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_scenario_1_a_revoked_token_cannot_reach_the_model_either(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    llm_service: RecordingLLMService,
) -> None:
    """Authentication is checked before the operation is resolved.

    The fake records every request it receives, so an empty list proves the
    application server stopped the call rather than merely ignoring the
    answer.
    """
    stub = nodes
    token = login(stub, auth_user.username)
    stub.logout(client_pb2.LogoutRequest(session_token=token))

    with pytest.raises(grpc.RpcError) as exc:
        ask(stub, token, "INVENTORY_SUMMARY")

    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert llm_service.requests == []


# ===========================================================================
# Scenario 2 -- inventory through gRPC
# ===========================================================================
def test_scenario_2_set_known_stock_then_read_it_back(
    nodes: client_pb2_grpc.ClientServiceStub, estate: dict[str, Any], auth_user: Any
) -> None:
    """Create, stock and query a product without touching the database."""
    stub = nodes
    token = login(stub, auth_user.username)

    assert post(stub, token, "product.create", {"sku": "SKU-C", "name": "Cable", "unit_price": "3.25"}).success
    assert post(
        stub,
        token,
        "warehouse.create",
        {"code": "WH-2", "name": "Annexe", "location": "Pune"},
    ).success
    assert post(
        stub,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-2", "sku": "SKU-C", "quantity": 42, "reorder_point": 5},
    ).success

    response = get(
        stub, token, "inventory.stock", {"warehouse_code": "WH-2", "sku": "SKU-C"}
    )

    assert response.success is True
    row = response.items[0].data
    assert int(row["quantity"]) == 42
    assert row["sku"] == "SKU-C"
    assert row["warehouse_code"] == "WH-2"
    assert row["is_low_stock"] is False


def test_scenario_2_adding_stock_moves_the_number_the_server_reports(
    nodes: client_pb2_grpc.ClientServiceStub, estate: dict[str, Any], auth_user: Any
) -> None:
    stub = nodes
    token = login(stub, auth_user.username)
    assert stock(stub, token, "SKU-A") == 50

    post(stub, token, "inventory.add_stock", {"warehouse_code": "WH-1", "sku": "SKU-A", "amount": 25})

    assert stock(stub, token, "SKU-A") == 75


# ===========================================================================
# Scenario 3 -- a successful multi-item order
# ===========================================================================
def test_scenario_3_a_multi_item_order_writes_every_row_and_moves_stock(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    session_factory: sessionmaker[DbSession],
) -> None:
    """One RPC, four consequences, all of which must land together.

    The order header, the lines, the stock decrements and the sales rows are
    written inside a single ``session_scope``, so this asserts all four
    rather than trusting that one implies the others.
    """
    stub = nodes
    token = login(stub, auth_user.username)

    response = post(
        stub,
        token,
        "order.place",
        {
            "warehouse_code": "WH-1",
            "items": [
                {"sku": "SKU-A", "quantity": 2},
                {"sku": "SKU-B", "quantity": 3},
            ],
        },
    )
    assert response.success is True

    # stock decremented, read back through the RPC rather than the database
    assert stock(stub, token, "SKU-A") == 48
    assert stock(stub, token, "SKU-B") == 3

    with session_factory() as db:
        orders = OrderRepository(db).list()
        items = OrderItemRepository(db).list()
        sales = SalesHistoryRepository(db).list()

        # the order record
        assert len(orders) == 1
        order = orders[0]
        assert order.order_number.startswith("ORD-")
        assert order.status == "COMPLETED"
        # 2 * 24.99 + 3 * 79.50 -- computed by the server, not the client
        assert order.total_amount == Decimal("288.48")

        # the order lines
        assert {item.product_id: item.quantity for item in items} == {
            estate["product_a"]: 2,
            estate["product_b"]: 3,
        }
        assert all(item.order_id == order.id for item in items)

        # the sales history, which demand forecasting later reads
        assert {row.product_id: row.quantity for row in sales} == {
            estate["product_a"]: 2,
            estate["product_b"]: 3,
        }
        assert all(row.order_id == order.id for row in sales)
        # one server timestamp for the whole order
        assert len({row.sold_at for row in sales}) == 1


# ===========================================================================
# Scenario 4 -- a failed multi-item order rolls everything back
# ===========================================================================
def test_scenario_4_one_short_line_rolls_back_the_whole_order(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    session_factory: sessionmaker[DbSession],
) -> None:
    """The first line is available and *is* decremented before the second fails.

    That ordering is what makes this test meaningful: the rollback has real
    work to undo. SKU-A is processed first and succeeds; SKU-B is short. A
    system that committed per line would leave SKU-A at 40 and an order
    header behind. Everything below must be exactly as it started.
    """
    stub = nodes
    token = login(stub, auth_user.username)
    before = database_snapshot(session_factory)
    assert before["stock"][estate["product_b"]] == 6

    with pytest.raises(grpc.RpcError) as exc:
        post(
            stub,
            token,
            "order.place",
            {
                "warehouse_code": "WH-1",
                "items": [
                    {"sku": "SKU-A", "quantity": 10},  # available
                    {"sku": "SKU-B", "quantity": 500},  # not available
                ],
            },
        )

    # A precondition failure, not bad input: the request was well formed.
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "SKU-B" in exc.value.details()

    # Nothing partial survived: no order, no lines, no sales, no decrement.
    assert database_snapshot(session_factory) == before
    assert stock(stub, token, "SKU-A") == 50
    assert stock(stub, token, "SKU-B") == 6


def test_scenario_4_the_server_still_works_afterwards(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    session_factory: sessionmaker[DbSession],
) -> None:
    """A rolled-back transaction must not poison the session or the pool."""
    stub = nodes
    token = login(stub, auth_user.username)

    with pytest.raises(grpc.RpcError):
        post(
            stub,
            token,
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-B", "quantity": 500}]},
        )

    # The very next order, which fits, must succeed.
    assert post(
        stub,
        token,
        "order.place",
        {"warehouse_code": "WH-1", "items": [{"sku": "SKU-B", "quantity": 6}]},
    ).success
    assert stock(stub, token, "SKU-B") == 0


# ===========================================================================
# Scenario 5 -- application -> LLM over gRPC
# ===========================================================================
@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("DEMAND_FORECAST", {"sku": "SKU-A", "warehouse_code": "WH-1"}),
        ("REORDER_RECOMMENDATION", {"sku": "SKU-A", "warehouse_code": "WH-1"}),
        ("INVENTORY_SUMMARY", None),
    ],
)
def test_scenario_5_the_application_sends_the_expected_query_type(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    llm_service: RecordingLLMService,
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    """Asserted on the message the LLM node received, not on the reply.

    The three names are one vocabulary shared by the client's operation
    string, the wire's query_type and the prompt layer's dispatch key. This
    is the test that would catch them drifting apart.
    """
    stub = nodes
    token = login(stub, auth_user.username)

    ask(stub, token, operation, payload)

    assert len(llm_service.requests) == 1
    assert llm_service.last.query_type == operation
    assert llm_service.last.request_id == "req-1"


def test_scenario_5_the_context_crosses_the_wire_with_the_database_figures(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    llm_service: RecordingLLMService,
) -> None:
    """The context is gathered from SQLite by the application server.

    The LLM node cannot open a database, so anything it knows had to arrive
    in this message. Checking it here proves the application server did the
    reading.
    """
    stub = nodes
    token = login(stub, auth_user.username)

    ask(stub, token, "REORDER_RECOMMENDATION", {"sku": "SKU-A", "warehouse_code": "WH-1"})

    context = llm_service.last_context()
    assert context, "the application server sent an empty context"
    flattened = str(context)
    assert "SKU-A" in flattened
    assert "WH-1" in flattened
    # The stock figure the estate fixture committed.
    assert "50" in flattened


def test_scenario_5_the_answer_is_returned_to_the_caller(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
) -> None:
    stub = nodes
    token = login(stub, auth_user.username)

    response = ask(stub, token, "INVENTORY_SUMMARY")

    assert response.success is True
    assert response.request_id == "req-1"
    data = dict(response.data)
    assert data["answer"] == ANSWER
    assert data["model"] == MODEL
    assert data["query_type"] == "INVENTORY_SUMMARY"
    # Stated in the payload, not only in the prose.
    assert data["advisory"] is True
    # The deterministic figures travel back beside the prose, so a reader can
    # tell a measured number from a written one.
    assert data["context"]


def test_scenario_5_an_analysis_writes_nothing(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    session_factory: sessionmaker[DbSession],
) -> None:
    """The LLM path is read-only, and the model has no way to change that."""
    stub = nodes
    token = login(stub, auth_user.username)
    before = database_snapshot(session_factory)

    for operation in ("DEMAND_FORECAST", "REORDER_RECOMMENDATION", "INVENTORY_SUMMARY"):
        payload = None if operation == "INVENTORY_SUMMARY" else {"sku": "SKU-A", "warehouse_code": "WH-1"}
        ask(stub, token, operation, payload)

    assert database_snapshot(session_factory) == before


# ===========================================================================
# Scenario 6 -- the LLM is unavailable
# ===========================================================================
@pytest.mark.parametrize(
    ("far_side", "expected"),
    [
        (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAVAILABLE),
        (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.DEADLINE_EXCEEDED),
        (grpc.StatusCode.INTERNAL, grpc.StatusCode.INTERNAL),
    ],
)
def test_scenario_6_a_failing_llm_node_becomes_a_controlled_status(
    session_factory: sessionmaker[DbSession],
    estate: dict[str, Any],
    auth_user: Any,
    far_side: grpc.StatusCode,
    expected: grpc.StatusCode,
) -> None:
    """The failure is translated, not leaked and not swallowed.

    UNAVAILABLE and DEADLINE_EXCEEDED survive both hops as themselves, which
    is what lets a caller tell "try again" from "it is broken". The
    application server neither returns success with no answer nor collapses
    everything into INTERNAL.
    """
    service = RecordingLLMService(abort_with=far_side)

    for stub in _nodes(session_factory, service):
        token = login(stub, auth_user.username)

        with pytest.raises(grpc.RpcError) as exc:
            ask(stub, token, "INVENTORY_SUMMARY")

        assert exc.value.code() == expected


def test_scenario_6_the_application_stays_alive_and_the_database_is_untouched(
    session_factory: sessionmaker[DbSession], estate: dict[str, Any], auth_user: Any
) -> None:
    """An outage at the LLM node must not reach the rest of the system.

    Three things are checked after the failure: the process is still serving,
    ordinary RPCs still work, and nothing was written. The third is the one
    worth stating -- the context is gathered inside a transaction, and that
    transaction has to roll back cleanly when the call that follows it fails.
    """
    service = RecordingLLMService(abort_with=grpc.StatusCode.UNAVAILABLE)

    for stub in _nodes(session_factory, service):
        token = login(stub, auth_user.username)
        before = database_snapshot(session_factory)

        for _ in range(3):
            with pytest.raises(grpc.RpcError) as exc:
                ask(stub, token, "INVENTORY_SUMMARY")
            assert exc.value.code() == grpc.StatusCode.UNAVAILABLE

        # still serving
        assert get(stub, token, "products").success is True
        assert get(stub, token, "warehouses").success is True

        # orders still work, so the outage did not poison the session pool
        assert post(
            stub,
            token,
            "order.place",
            {"warehouse_code": "WH-1", "items": [{"sku": "SKU-A", "quantity": 1}]},
        ).success

        after = database_snapshot(session_factory)
        # The only change is the order that was placed *after* the outage.
        assert after["orders"] == before["orders"] + 1
        assert after["stock"][estate["product_a"]] == before["stock"][estate["product_a"]] - 1


def test_scenario_6_a_node_that_is_not_listening_is_unavailable(
    session_factory: sessionmaker[DbSession], estate: dict[str, Any], auth_user: Any
) -> None:
    """Nothing is bound to the port at all -- a harder case than a failing node."""
    llm_client = LLMClient("127.0.0.1:1", timeout=1.0)
    app_server, port = create_app_server(
        session_factory, host="127.0.0.1", port=0, max_workers=4, llm_client=llm_client
    )
    app_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = client_pb2_grpc.ClientServiceStub(channel)
            token = login(stub, auth_user.username)
            before = database_snapshot(session_factory)

            with pytest.raises(grpc.RpcError) as exc:
                ask(stub, token, "INVENTORY_SUMMARY")
            assert exc.value.code() == grpc.StatusCode.UNAVAILABLE

            assert get(stub, token, "products").success is True
            assert database_snapshot(session_factory) == before
    finally:
        app_server.stop(None).wait()
        llm_client.close()


# ===========================================================================
# Scenarios 7 and 8 -- isolation and determinism, asserted rather than assumed
# ===========================================================================
def test_the_suite_runs_against_a_temporary_database(
    db_path: Any, session_factory: sessionmaker[DbSession]
) -> None:
    """Never data/node1.db.

    The session-scoped guard in ``tests/conftest.py`` fails the run if the
    node database changes at all; this states the positive half, that the
    file under test is a throwaway under pytest's tmp_path.

    ``session_factory`` is requested so the engine has actually created the
    file by the time it is inspected -- ``db_path`` alone only names it.
    """
    from app.storage.database import get_database_path

    assert db_path != get_database_path()
    assert "node1.db" not in str(db_path)
    assert db_path.exists()
    assert "pytest" in str(db_path)


def test_each_test_starts_from_its_own_database(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    session_factory: sessionmaker[DbSession],
) -> None:
    """No order from any earlier test is visible here."""
    assert database_snapshot(session_factory)["orders"] == 0


def test_the_fake_model_is_deterministic(
    nodes: client_pb2_grpc.ClientServiceStub,
    estate: dict[str, Any],
    auth_user: Any,
    llm_service: RecordingLLMService,
) -> None:
    """Same question, same answer, every time and with no runtime involved."""
    stub = nodes
    token = login(stub, auth_user.username)

    answers = {dict(ask(stub, token, "INVENTORY_SUMMARY").data)["answer"] for _ in range(3)}

    assert answers == {ANSWER}
    assert len(llm_service.requests) == 3


def test_no_test_in_this_file_needs_ollama() -> None:
    """The model runtime is never imported here, let alone reached.

    ``tests/conftest.py`` enforces this for the whole suite by failing the
    run on a real ``ollama.Client`` or a socket to 11434. This states the
    intent locally, where a future edit to this file would see it.
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

    assert not any(name.split(".")[0] in {"ollama", "httpx"} for name in imported)
