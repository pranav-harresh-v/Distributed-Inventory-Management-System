"""``processBusinessRequest`` routing and the three AI operations (Phase G).

This is where the whole Phase G chain meets Phase F, so the tests split in
two. Half cover the routing itself -- authentication before dispatch, unknown
operations, correlation ids, every failure of the LLM node mapped onto a
status a client can branch on. The other half cover the guarantees the
project is graded against and which nothing else asserts end to end:

* **The client never reaches the model.** It sends an operation name and a
  product; this node authenticates it, reads its own database, builds the
  context and makes the call. A mocked LLM client records what it was given,
  which is how the tests can show the context was assembled server-side.
* **No advice moves stock.** The model is made to answer with a sentence
  claiming an order was placed, and the inventory is checked afterwards. It
  must be untouched: nothing parses that prose into a quantity.

The LLM client is mocked throughout -- no LLM node, no Ollama, no model.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.api.llm_client import (
    LLMAnswer,
    LLMClientError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
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
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

ANSWER = "Recommend ordering 40 units of SKU-1 before the lead time bites."
MODEL = "phi3:mini"


class FakeContext:
    """Minimal ServicerContext stand-in; ``abort`` raises, as the real one does."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise _Aborted(code, details)


class _Aborted(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


class FakeLLMClient:
    """Records what the application server asked, and answers or fails."""

    def __init__(
        self, *, answer: str = ANSWER, raises: BaseException | None = None
    ) -> None:
        self.answer = answer
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def get_llm_answer(
        self,
        request_id: str,
        query_type: str,
        query: str = "",
        context: Any = None,
    ) -> LLMAnswer:
        self.calls.append(
            {
                "request_id": request_id,
                "query_type": query_type,
                "query": query,
                "context": context,
            }
        )
        if self.raises is not None:
            raise self.raises
        return LLMAnswer(request_id=request_id, answer=self.answer, model=MODEL)

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


@pytest.fixture
def estate(session_factory: sessionmaker[DbSession]) -> dict[str, int]:
    """A product with stock, sales history and a supplier, committed."""
    with session_factory() as db:
        user = User(
            username="ops", email="ops@example.com", password_hash="x", role="manager"
        )
        product = Product(sku="SKU-1", name="Widget", unit_price=Decimal("24.99"))
        bare = Product(sku="SKU-2", name="Gizmo", unit_price=Decimal("5.00"))
        warehouse = Warehouse(code="WH-1", name="Main", location="Bengaluru")
        db.add_all([user, product, bare, warehouse])
        db.flush()

        db.add_all(
            [
                Inventory(
                    warehouse_id=warehouse.id,
                    product_id=product.id,
                    quantity=8,
                    reorder_point=25,
                    safety_stock=10,
                ),
                Inventory(
                    warehouse_id=warehouse.id,
                    product_id=bare.id,
                    quantity=50,
                    reorder_point=5,
                ),
            ]
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
        return {"warehouse_id": warehouse.id, "product_id": product.id}


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def service(
    session_factory: sessionmaker[DbSession], llm: FakeLLMClient
) -> ClientService:
    return ClientService(session_factory, llm_client=llm)


@pytest.fixture
def token(service: ClientService, auth_user: User) -> str:
    response = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    )
    return response.session_token


def _struct(values: dict[str, Any]) -> Struct:
    struct = Struct()
    struct.update(values)
    return struct


def business_request(
    token: str,
    operation: str,
    payload: dict[str, Any] | None = None,
    *,
    request_id: str = "req-1",
) -> client_pb2.BusinessRequest:
    request = client_pb2.BusinessRequest(
        session_token=token, operation=operation, request_id=request_id
    )
    if payload:
        struct = Struct()
        struct.update(payload)
        request.payload.CopyFrom(struct)
    return request


def run(
    service: ClientService, request: client_pb2.BusinessRequest
) -> client_pb2.BusinessResponse:
    return service.processBusinessRequest(request, FakeContext())


def expect_abort(
    service: ClientService, request: client_pb2.BusinessRequest
) -> FakeContext:
    context = FakeContext()
    with pytest.raises(_Aborted):
        service.processBusinessRequest(request, context)
    return context


# --- the three operations --------------------------------------------------
def test_demand_forecast_returns_advice(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    response = run(
        service,
        business_request(
            token,
            "DEMAND_FORECAST",
            {"sku": "SKU-1", "warehouse_code": "WH-1", "horizon_days": 7},
        ),
    )

    assert response.success is True
    data = dict(response.data)
    assert data["query_type"] == "DEMAND_FORECAST"
    assert data["answer"] == ANSWER
    assert data["model"] == MODEL
    assert data["advisory"] is True
    assert llm.last_call["query_type"] == "DEMAND_FORECAST"


def test_reorder_recommendation_returns_advice(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    response = run(
        service,
        business_request(
            token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}
        ),
    )

    context = llm.last_call["context"]
    assert context["current_quantity"] == 8
    assert context["below_reorder_point"] is True
    assert context["baseline_reorder_quantity"] > 0
    assert dict(response.data)["advisory"] is True


def test_inventory_summary_returns_advice(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    response = run(service, business_request(token, "INVENTORY_SUMMARY"))

    assert response.success is True
    assert llm.last_call["context"]["totals"]["stock_lines"] == 2


def test_the_deterministic_context_comes_back_with_the_prose(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    """The auditable half: a reader can tell a measured number from a written one."""
    response = run(
        service,
        business_request(
            token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}
        ),
    )

    context = dict(response.data)["context"]
    assert context["current_quantity"] == 8
    assert context["baseline_rationale"]


def test_an_operator_question_reaches_the_model(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    run(
        service,
        business_request(
            token,
            "DEMAND_FORECAST",
            {
                "sku": "SKU-1",
                "warehouse_code": "WH-1",
                "question": "Should I worry about the weekend?",
            },
        ),
    )

    assert llm.last_call["query"] == "Should I worry about the weekend?"
    # Held apart from the builder's arguments, which reject unknown fields.
    assert "question" not in llm.last_call["context"]


# --- the client never reaches the model ------------------------------------
def test_the_application_server_builds_the_context_itself(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    """The client sends a SKU; everything the model sees came from SQLite."""
    run(
        service,
        business_request(
            token, "DEMAND_FORECAST", {"sku": "SKU-1", "warehouse_code": "WH-1"}
        ),
    )

    context = llm.last_call["context"]
    assert context["product"]["name"] == "Widget"
    assert context["recent_sales_total"] == 42
    assert context["historical_daily_sales"]["days_with_sales"] == 14


def test_the_client_contract_offers_no_route_to_the_llm() -> None:
    """There is no LLM service on the client-facing contract to call."""
    services = client_pb2.DESCRIPTOR.services_by_name

    assert set(services) == {"ClientService"}
    methods = {method.name for method in services["ClientService"].methods}
    assert methods == {
        "login",
        "logout",
        "post",
        "get",
        "processBusinessRequest",
    }


# --- no advice moves stock -------------------------------------------------
def test_an_answer_claiming_an_order_changes_nothing(
    service: ClientService,
    token: str,
    estate: dict[str, int],
    session_factory: sessionmaker[DbSession],
) -> None:
    """The model asserts an action happened. The database must disagree."""
    service._llm_client = FakeLLMClient(
        answer="I have placed a restock order for 500 units and updated stock."
    )

    run(
        service,
        business_request(
            token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}
        ),
    )

    with session_factory() as db:
        record = (
            db.query(Inventory)
            .filter_by(
                warehouse_id=estate["warehouse_id"], product_id=estate["product_id"]
            )
            .one()
        )
        assert record.quantity == 8
        assert db.query(SupplierProduct).count() == 1
        assert db.query(Order).count() == 1


# --- authentication --------------------------------------------------------
def test_an_invalid_token_is_unauthenticated(
    service: ClientService, estate: dict[str, int]
) -> None:
    context = expect_abort(
        service, business_request("not-a-token", "INVENTORY_SUMMARY")
    )

    assert context.code == grpc.StatusCode.UNAUTHENTICATED


def test_authentication_runs_before_the_operation_is_resolved(
    service: ClientService, estate: dict[str, int]
) -> None:
    """Otherwise an anonymous caller could discover which operations exist."""
    context = expect_abort(service, business_request("not-a-token", "NONSENSE"))

    assert context.code == grpc.StatusCode.UNAUTHENTICATED


def test_an_unauthenticated_request_never_reaches_the_model(
    service: ClientService, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    expect_abort(service, business_request("not-a-token", "INVENTORY_SUMMARY"))

    assert llm.calls == []


def test_a_revoked_session_cannot_run_an_operation(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    service.logout(client_pb2.LogoutRequest(session_token=token), FakeContext())

    context = expect_abort(service, business_request(token, "INVENTORY_SUMMARY"))

    assert context.code == grpc.StatusCode.UNAUTHENTICATED


# --- request validation ----------------------------------------------------
@pytest.mark.parametrize(
    "operation", ["", "   ", "demand_forecast", "FORECAST", "order.place"]
)
def test_an_unknown_operation_is_rejected(
    service: ClientService, token: str, estate: dict[str, int], operation: str
) -> None:
    context = expect_abort(service, business_request(token, operation))

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "Supported values are" in context.details


def test_an_invalid_product_is_not_found(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    context = expect_abort(
        service,
        business_request(
            token, "DEMAND_FORECAST", {"sku": "NOPE", "warehouse_code": "WH-1"}
        ),
    )

    assert context.code == grpc.StatusCode.NOT_FOUND
    assert "product" in context.details


def test_an_invalid_warehouse_is_not_found(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    context = expect_abort(
        service,
        business_request(
            token, "DEMAND_FORECAST", {"sku": "SKU-1", "warehouse_code": "NOPE"}
        ),
    )

    assert context.code == grpc.StatusCode.NOT_FOUND
    assert "warehouse" in context.details


def test_an_unknown_payload_field_is_invalid(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    context = expect_abort(
        service,
        business_request(
            token,
            "DEMAND_FORECAST",
            {"sku": "SKU-1", "warehouse_code": "WH-1", "magic": True},
        ),
    )

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT


def test_a_bad_request_never_reaches_the_model(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    expect_abort(
        service,
        business_request(
            token, "DEMAND_FORECAST", {"sku": "NOPE", "warehouse_code": "WH-1"}
        ),
    )

    assert llm.calls == []


# --- sparse data still answers ---------------------------------------------
def test_a_product_with_no_sales_still_gets_an_answer(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    """Missing history is described to the model, not treated as an error."""
    response = run(
        service,
        business_request(
            token, "DEMAND_FORECAST", {"sku": "SKU-2", "warehouse_code": "WH-1"}
        ),
    )

    assert response.success is True
    context = llm.last_call["context"]
    assert context["recent_sales_total"] == 0
    assert context["baseline_forecast"] is None
    assert context["data_quality"] == "none"


def test_a_product_with_no_supplier_still_gets_an_answer(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    response = run(
        service,
        business_request(
            token, "REORDER_RECOMMENDATION", {"sku": "SKU-2", "warehouse_code": "WH-1"}
        ),
    )

    assert response.success is True
    context = llm.last_call["context"]
    assert context["supplier"] is None
    assert context["baseline_reorder_quantity"] is None


# --- correlation ids -------------------------------------------------------
def test_the_request_id_is_echoed_back(
    service: ClientService, token: str, estate: dict[str, int], llm: FakeLLMClient
) -> None:
    response = run(
        service, business_request(token, "INVENTORY_SUMMARY", request_id="req-99")
    )

    assert response.request_id == "req-99"
    assert llm.last_call["request_id"] == "req-99"


@pytest.mark.parametrize("request_id", ["", "   "])
def test_a_missing_request_id_is_generated(
    service: ClientService,
    token: str,
    estate: dict[str, int],
    llm: FakeLLMClient,
    request_id: str,
) -> None:
    """Every answer stays correlatable, even from a caller that sent nothing."""
    response = run(
        service, business_request(token, "INVENTORY_SUMMARY", request_id=request_id)
    )

    assert response.request_id.startswith("REQ-")
    assert llm.last_call["request_id"] == response.request_id


def test_generated_request_ids_are_unique(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    ids = {
        run(
            service, business_request(token, "INVENTORY_SUMMARY", request_id="")
        ).request_id
        for _ in range(5)
    }

    assert len(ids) == 5


# --- LLM node failures -----------------------------------------------------
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (LLMUnavailableError("node down"), grpc.StatusCode.UNAVAILABLE),
        (LLMTimeoutError("no answer in 65s"), grpc.StatusCode.DEADLINE_EXCEEDED),
        (LLMRequestError("unknown query_type"), grpc.StatusCode.INTERNAL),
        (LLMClientError("something else"), grpc.StatusCode.INTERNAL),
    ],
)
def test_llm_failures_map_onto_useful_statuses(
    session_factory: sessionmaker[DbSession],
    auth_user: User,
    estate: dict[str, int],
    failure: Exception,
    expected: grpc.StatusCode,
) -> None:
    service = ClientService(session_factory, llm_client=FakeLLMClient(raises=failure))
    login = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    )

    context = expect_abort(
        service, business_request(login.session_token, "INVENTORY_SUMMARY")
    )

    assert context.code == expected


def test_an_unavailable_llm_node_leaves_the_database_alone(
    session_factory: sessionmaker[DbSession],
    auth_user: User,
    estate: dict[str, int],
) -> None:
    """A Phase F invariant: a failed operation changes nothing."""
    service = ClientService(
        session_factory, llm_client=FakeLLMClient(raises=LLMUnavailableError("down"))
    )
    login = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    )

    expect_abort(
        service,
        business_request(
            login.session_token,
            "REORDER_RECOMMENDATION",
            {"sku": "SKU-1", "warehouse_code": "WH-1"},
        ),
    )

    with session_factory() as db:
        record = (
            db.query(Inventory)
            .filter_by(
                warehouse_id=estate["warehouse_id"], product_id=estate["product_id"]
            )
            .one()
        )
        assert record.quantity == 8


# --- Phase F is untouched --------------------------------------------------
def test_the_existing_rpcs_work_without_an_llm_client(
    session_factory: sessionmaker[DbSession], auth_user: User, estate: dict[str, int]
) -> None:
    """A servicer constructed the Phase F way behaves exactly as before.

    It must also not open a channel it will never use, which is why the LLM
    client is built on first AI request rather than in ``__init__``.
    """
    service = ClientService(session_factory)

    login = service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    )
    assert login.success is True

    created = service.post(
        client_pb2.PostRequest(
            session_token=login.session_token,
            resource="warehouse.create",
            payload=_struct({"code": "WH-9", "name": "Spare", "location": "Chennai"}),
        ),
        FakeContext(),
    )
    assert created.success is True

    listed = service.get(
        client_pb2.GetRequest(session_token=login.session_token, resource="warehouses"),
        FakeContext(),
    )
    assert len(listed.items) == 2
    assert service._llm_client is None


def test_post_and_get_are_unaffected(
    service: ClientService, token: str, estate: dict[str, int]
) -> None:
    listed = service.get(
        client_pb2.GetRequest(session_token=token, resource="products"), FakeContext()
    )

    assert listed.success is True
    assert len(listed.items) == 2
