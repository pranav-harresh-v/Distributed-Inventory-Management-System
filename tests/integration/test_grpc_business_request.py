"""The whole Phase G chain, over real gRPC, with only the model faked.

Two genuine servers run here: the application node and the LLM node, each on
its own ephemeral port, with a real channel between them and a real channel
from the client. Only Ollama is stubbed.

    client --gRPC--> application server --gRPC--> LLM server --> (fake model)
                           |                          |
                        SQLite                   (no database)

Every other Phase G test mocks one side of a boundary. This one mocks none of
them, which is what makes it able to catch the failures that matter: a
context key the prompt layer never renders, a status code lost in transit, or
a client that could somehow reach the model on its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from decimal import Decimal
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.llm_client import LLMClient
from app.server import create_server as create_app_server
from llm.ollama_client import OllamaTimeoutError, OllamaUnavailableError
from llm.server import create_server as create_llm_server
from proto import client_pb2, client_pb2_grpc
from tests.conftest import TEST_PASSWORD

ANSWER = "Stock is nine days from running out; recommend ordering 40 units."
MODEL = "phi3:mini"


class FakeOllamaClient:
    """The only fake in the chain: everything else is the real thing."""

    def __init__(self, *, raises: BaseException | None = None) -> None:
        self.model = MODEL
        self.raises = raises
        self.prompts: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.prompts.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return ANSWER


@pytest.fixture
def ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def estate(session_factory: sessionmaker[DbSession]) -> None:
    from app.storage.models import (
        ORDER_COMPLETED,
        Inventory,
        Order,
        Product,
        SalesHistory,
        Supplier,
        SupplierProduct,
        User,
        Warehouse,
        utc_now,
    )

    with session_factory() as db:
        user = User(
            username="ops", email="ops@example.com", password_hash="x", role="manager"
        )
        product = Product(sku="SKU-1", name="Widget", unit_price=Decimal("24.99"))
        warehouse = Warehouse(code="WH-1", name="Main", location="Bengaluru")
        db.add_all([user, product, warehouse])
        db.flush()

        db.add(
            Inventory(
                warehouse_id=warehouse.id,
                product_id=product.id,
                quantity=8,
                reorder_point=25,
                safety_stock=10,
            )
        )
        order = Order(
            order_number="ORD-TEST",
            user_id=user.id,
            warehouse_id=warehouse.id,
            status=ORDER_COMPLETED,
            total_amount=Decimal("0.00"),
        )
        supplier = Supplier(name="Acme")
        db.add_all([order, supplier])
        db.flush()
        db.add(
            SupplierProduct(
                supplier_id=supplier.id,
                product_id=product.id,
                unit_cost=Decimal("11.50"),
                lead_time_days=7,
            )
        )
        for day in range(1, 15):
            db.add(
                SalesHistory(
                    order_id=order.id,
                    warehouse_id=warehouse.id,
                    product_id=product.id,
                    quantity=3,
                    unit_price=Decimal("24.99"),
                    sold_at=utc_now() - timedelta(days=day),
                )
            )
        db.commit()


def both_nodes(
    session_factory: sessionmaker[DbSession], ollama: FakeOllamaClient
) -> Iterator[client_pb2_grpc.ClientServiceStub]:
    """Start the LLM node, point an application node at it, and connect."""
    llm_server, llm_port = create_llm_server(
        ollama, host="127.0.0.1", port=0, max_workers=2
    )
    llm_server.start()
    llm_client = LLMClient(f"127.0.0.1:{llm_port}", timeout=5.0)

    app_server, app_port = create_app_server(
        session_factory,
        host="127.0.0.1",
        port=0,
        max_workers=4,
        llm_client=llm_client,
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
def stub(
    session_factory: sessionmaker[DbSession], ollama: FakeOllamaClient
) -> Iterator[client_pb2_grpc.ClientServiceStub]:
    yield from both_nodes(session_factory, ollama)


def login(stub: client_pb2_grpc.ClientServiceStub, username: str) -> str:
    response = stub.login(
        client_pb2.LoginRequest(username=username, password=TEST_PASSWORD)
    )
    return response.session_token


def ask(
    stub: client_pb2_grpc.ClientServiceStub,
    token: str,
    operation: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str = "req-1",
) -> client_pb2.BusinessResponse:
    request = client_pb2.BusinessRequest(
        session_token=token, operation=operation, request_id=request_id
    )
    if payload:
        struct = Struct()
        struct.update(payload)
        request.payload.CopyFrom(struct)
    return stub.processBusinessRequest(request)


@pytest.mark.parametrize(
    ("operation", "payload"),
    [
        ("DEMAND_FORECAST", {"sku": "SKU-1", "warehouse_code": "WH-1"}),
        ("REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}),
        ("INVENTORY_SUMMARY", None),
    ],
)
def test_each_operation_crosses_both_hops(
    stub: client_pb2_grpc.ClientServiceStub,
    estate: None,
    auth_user: Any,
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    response = ask(stub, login(stub, auth_user.username), operation, payload)

    assert response.success is True
    data = dict(response.data)
    assert data["answer"] == ANSWER
    assert data["model"] == MODEL
    assert data["query_type"] == operation
    assert data["advisory"] is True
    assert response.request_id == "req-1"


def test_the_context_survives_every_hop(
    stub: client_pb2_grpc.ClientServiceStub,
    estate: None,
    auth_user: Any,
    ollama: FakeOllamaClient,
) -> None:
    """SQLite -> context -> Struct -> gRPC -> Struct -> prompt, intact."""
    ask(
        stub,
        login(stub, auth_user.username),
        "REORDER_RECOMMENDATION",
        {"sku": "SKU-1", "warehouse_code": "WH-1"},
    )

    prompt = ollama.prompts[-1]["user_prompt"]
    assert "Current quantity on hand: 8" in prompt
    assert "Reorder point: 25" in prompt
    assert "Supplier lead time (days): 7" in prompt
    assert '"unit_cost": "11.50"' in prompt
    # The whole number survived the Struct's doubles.
    assert "8.0" not in prompt


def test_an_operator_question_reaches_the_model(
    stub: client_pb2_grpc.ClientServiceStub,
    estate: None,
    auth_user: Any,
    ollama: FakeOllamaClient,
) -> None:
    ask(
        stub,
        login(stub, auth_user.username),
        "DEMAND_FORECAST",
        {"sku": "SKU-1", "warehouse_code": "WH-1", "question": "Busy weekend ahead?"},
    )

    assert "Busy weekend ahead?" in ollama.prompts[-1]["user_prompt"]


def test_an_unauthenticated_caller_gets_nowhere(
    stub: client_pb2_grpc.ClientServiceStub, estate: None, ollama: FakeOllamaClient
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        ask(stub, "not-a-token", "INVENTORY_SUMMARY")

    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    assert ollama.prompts == []


def test_an_unknown_operation_is_invalid_argument(
    stub: client_pb2_grpc.ClientServiceStub, estate: None, auth_user: Any
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        ask(stub, login(stub, auth_user.username), "NOT_AN_OPERATION")

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_an_invalid_product_is_not_found(
    stub: client_pb2_grpc.ClientServiceStub, estate: None, auth_user: Any
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        ask(
            stub,
            login(stub, auth_user.username),
            "DEMAND_FORECAST",
            {"sku": "NOPE", "warehouse_code": "WH-1"},
        )

    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_a_generated_request_id_comes_back(
    stub: client_pb2_grpc.ClientServiceStub, estate: None, auth_user: Any
) -> None:
    response = ask(
        stub, login(stub, auth_user.username), "INVENTORY_SUMMARY", request_id=""
    )

    assert response.request_id.startswith("REQ-")


def test_an_unreachable_model_surfaces_as_unavailable(
    session_factory: sessionmaker[DbSession], auth_user: Any, estate: None
) -> None:
    """Ollama is down behind the LLM node; the client learns to retry."""
    ollama = FakeOllamaClient(raises=OllamaUnavailableError("Ollama is down."))

    for stub in both_nodes(session_factory, ollama):
        with pytest.raises(grpc.RpcError) as exc:
            ask(stub, login(stub, auth_user.username), "INVENTORY_SUMMARY")

        assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_a_generation_timeout_surfaces_as_deadline_exceeded(
    session_factory: sessionmaker[DbSession], auth_user: Any, estate: None
) -> None:
    """Distinct from UNAVAILABLE across two hops, which is the point of it."""
    ollama = FakeOllamaClient(raises=OllamaTimeoutError("no answer in 60 seconds"))

    for stub in both_nodes(session_factory, ollama):
        with pytest.raises(grpc.RpcError) as exc:
            ask(stub, login(stub, auth_user.username), "INVENTORY_SUMMARY")

        assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_a_dead_llm_node_does_not_take_the_application_down(
    session_factory: sessionmaker[DbSession], auth_user: Any, estate: None
) -> None:
    """Nothing is listening on the LLM port at all: Phase F must still serve."""
    app_server, port = create_app_server(
        session_factory,
        host="127.0.0.1",
        port=0,
        max_workers=4,
        llm_client=LLMClient("127.0.0.1:1", timeout=1.0),
    )
    app_server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = client_pb2_grpc.ClientServiceStub(channel)
            token = login(stub, auth_user.username)

            with pytest.raises(grpc.RpcError) as exc:
                ask(stub, token, "INVENTORY_SUMMARY")
            assert exc.value.code() == grpc.StatusCode.UNAVAILABLE

            # The existing RPCs are untouched by the outage.
            listed = stub.get(
                client_pb2.GetRequest(session_token=token, resource="products")
            )
            assert listed.success is True
            assert len(listed.items) == 1
    finally:
        app_server.stop(None).wait()
