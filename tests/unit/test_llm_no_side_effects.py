"""What an AI operation must never do to the database.

The project's LLM rule is one sentence -- the model is advisory and may not
change state -- and the rest of the suite verifies the pieces that implement
it: builders that only read, prose that is never parsed into a quantity, a
servicer with no repository imports. This module checks the rule itself, at
the only place it can actually be broken: after a complete
``processBusinessRequest`` has run against a real database.

Every mutable table is counted before and after. That is deliberately
broader than checking stock alone, because the interesting failure is not a
quantity moving -- it is a restock or an order appearing because someone
later wired "the model recommended it" to a write. ``restock_orders`` in
particular is asserted here and nowhere else in the suite.

The model is made hostile on purpose: it answers by claiming it has already
placed the order. The database has to disagree with it.
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
from app.api.llm_client import LLMAnswer
from app.storage.models import (
    ORDER_COMPLETED,
    Inventory,
    Order,
    OrderItem,
    Product,
    RestockOrder,
    SalesHistory,
    Supplier,
    SupplierProduct,
    User,
    Warehouse,
    utc_now,
)
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

#: An answer that asserts, in plain language, that it has already acted. If
#: any part of the system ever believes a model, this is the string that
#: would prove it.
ACTING_ANSWER = (
    "Done -- I have placed a restock order for 500 units with Acme, "
    "created the purchase order and updated the stock level to 508."
)

#: Every table an AI operation could conceivably write to.
MUTABLE_TABLES = (
    Inventory,
    Order,
    OrderItem,
    RestockOrder,
    SalesHistory,
    Product,
    Supplier,
    SupplierProduct,
    Warehouse,
)

OPERATIONS = (
    ("DEMAND_FORECAST", {"sku": "SKU-1", "warehouse_code": "WH-1"}),
    ("REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}),
    ("INVENTORY_SUMMARY", None),
)


class FakeContext:
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


class ActingLLMClient:
    """A model that claims to have changed the world."""

    def __init__(self, answer: str = ACTING_ANSWER) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def get_llm_answer(
        self,
        request_id: str,
        query_type: str,
        query: str = "",
        context: Any = None,
    ) -> LLMAnswer:
        self.calls.append({"query_type": query_type, "context": context})
        return LLMAnswer(request_id=request_id, answer=self.answer, model="phi3:mini")

    @property
    def last_context(self) -> dict[str, Any]:
        return self.calls[-1]["context"]


@pytest.fixture
def estate(session_factory: sessionmaker[DbSession]) -> dict[str, Any]:
    """Two warehouses with stock, sales history, a supplier and one order."""
    with session_factory() as db:
        user = User(
            username="ops", email="ops@example.com", password_hash="x", role="manager"
        )
        widget = Product(sku="SKU-1", name="Widget", unit_price=Decimal("24.99"))
        gizmo = Product(sku="SKU-2", name="Gizmo", unit_price=Decimal("5.00"))
        main = Warehouse(code="WH-1", name="Main", location="Bengaluru")
        depot = Warehouse(code="WH-2", name="Depot", location="Pune")
        db.add_all([user, widget, gizmo, main, depot])
        db.flush()

        db.add_all(
            [
                Inventory(
                    warehouse_id=main.id,
                    product_id=widget.id,
                    quantity=8,
                    reorder_point=25,
                    safety_stock=10,
                ),
                Inventory(
                    warehouse_id=depot.id, product_id=gizmo.id, quantity=60,
                    reorder_point=5,
                ),
            ]
        )
        order = Order(
            order_number="ORD-EXISTING",
            user_id=user.id,
            warehouse_id=main.id,
            status=ORDER_COMPLETED,
            total_amount=Decimal("0.00"),
        )
        supplier = Supplier(name="Acme")
        db.add_all([order, supplier])
        db.flush()
        db.add(
            SupplierProduct(
                supplier_id=supplier.id,
                product_id=widget.id,
                unit_cost=Decimal("11.50"),
                lead_time_days=7,
            )
        )
        for day in range(1, 15):
            db.add(
                SalesHistory(
                    order_id=order.id,
                    warehouse_id=main.id,
                    product_id=widget.id,
                    quantity=3,
                    unit_price=Decimal("24.99"),
                    sold_at=utc_now() - timedelta(days=day),
                )
            )
        db.commit()
        return {"warehouse_id": main.id, "product_id": widget.id}


@pytest.fixture
def llm() -> ActingLLMClient:
    return ActingLLMClient()


@pytest.fixture
def service(
    session_factory: sessionmaker[DbSession], llm: ActingLLMClient
) -> ClientService:
    return ClientService(session_factory, llm_client=llm)


@pytest.fixture
def token(service: ClientService, auth_user: User) -> str:
    return service.login(
        client_pb2.LoginRequest(username=auth_user.username, password=TEST_PASSWORD),
        FakeContext(),
    ).session_token


def snapshot(session_factory: sessionmaker[DbSession]) -> dict[str, Any]:
    """Row counts for every mutable table, plus the stock levels themselves.

    Counts alone would miss an update in place, which is the mutation a stock
    system actually has to fear.
    """
    with session_factory() as db:
        counts = {model.__name__: db.query(model).count() for model in MUTABLE_TABLES}
        levels = {
            (record.warehouse_id, record.product_id): record.quantity
            for record in db.query(Inventory).all()
        }
    return {"counts": counts, "levels": levels}


def run(
    service: ClientService, token: str, operation: str, payload: dict[str, Any] | None
) -> client_pb2.BusinessResponse:
    request = client_pb2.BusinessRequest(
        session_token=token, operation=operation, request_id="req-1"
    )
    if payload:
        struct = Struct()
        struct.update(payload)
        request.payload.CopyFrom(struct)
    return service.processBusinessRequest(request, FakeContext())


# --- nothing is written ----------------------------------------------------
@pytest.mark.parametrize(("operation", "payload"), OPERATIONS)
def test_an_operation_writes_to_no_table(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    before = snapshot(session_factory)

    response = run(service, token, operation, payload)

    assert response.success is True
    assert snapshot(session_factory) == before


@pytest.mark.parametrize(("operation", "payload"), OPERATIONS)
def test_no_restock_order_is_ever_created(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    """The table a reorder recommendation would write to, if it ever wrote.

    ``restock_orders`` exists in the schema and is unused: recommending a
    restock and recording one are separate acts, and only the second is a
    write. Asserted for all three operations because the guarantee is about
    the LLM path, not about one handler.
    """
    run(service, token, operation, payload)

    with session_factory() as db:
        assert db.query(RestockOrder).count() == 0


@pytest.mark.parametrize(("operation", "payload"), OPERATIONS)
def test_no_customer_order_is_ever_created(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    run(service, token, operation, payload)

    with session_factory() as db:
        assert db.query(Order).count() == 1
        assert db.query(Order).one().order_number == "ORD-EXISTING"
        assert db.query(OrderItem).count() == 0


@pytest.mark.parametrize(("operation", "payload"), OPERATIONS)
def test_stock_levels_are_untouched(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
    operation: str,
    payload: dict[str, Any] | None,
) -> None:
    """Even though the answer says the level was updated to 508."""
    run(service, token, operation, payload)

    with session_factory() as db:
        record = (
            db.query(Inventory)
            .filter_by(
                warehouse_id=estate["warehouse_id"], product_id=estate["product_id"]
            )
            .one()
        )
        assert record.quantity == 8


def test_the_claim_reaches_the_client_but_changes_nothing(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    """The prose is returned verbatim -- and is inert.

    The system does not sanitise a model's claims; it makes them harmless by
    never giving the text a path to a write, and by returning the measured
    figures beside it so a reader can check.
    """
    before = snapshot(session_factory)

    response = run(
        service, token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"}
    )

    data = dict(response.data)
    assert data["answer"] == ACTING_ANSWER
    assert data["advisory"] is True
    assert data["context"]["current_quantity"] == 8
    assert snapshot(session_factory) == before


def test_repeating_an_operation_stays_inert(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    session_factory: sessionmaker[DbSession],
) -> None:
    """Ten runs, no accumulation: there is no write path to leak through."""
    before = snapshot(session_factory)

    for _ in range(10):
        run(service, token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"})

    assert snapshot(session_factory) == before


# --- the context still carries the real data -------------------------------
def test_the_summary_spans_every_warehouse_through_the_rpc(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    llm: ActingLLMClient,
) -> None:
    """Multi-warehouse data reaches the model through the orchestration path.

    The builder is covered directly elsewhere; what this pins down is that
    the RPC does not quietly scope the summary to one warehouse.
    """
    run(service, token, "INVENTORY_SUMMARY", None)

    context = llm.last_context
    assert context["totals"]["warehouses"] == 2
    assert context["totals"]["warehouses_included"] == 2
    assert context["totals"]["total_units"] == 68
    assert {item["code"] for item in context["warehouses"]["items"]} == {"WH-1", "WH-2"}
    assert context["scope"] == "all warehouses"


def test_the_summary_can_still_be_scoped_to_one_warehouse(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    llm: ActingLLMClient,
) -> None:
    run(service, token, "INVENTORY_SUMMARY", {"warehouse_code": "WH-2"})

    context = llm.last_context
    assert context["scope"] == "WH-2"
    assert context["totals"]["total_units"] == 60


def test_the_reorder_context_carries_inventory_and_supplier(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    llm: ActingLLMClient,
) -> None:
    run(service, token, "REORDER_RECOMMENDATION", {"sku": "SKU-1", "warehouse_code": "WH-1"})

    context = llm.last_context
    assert context["current_quantity"] == 8
    assert context["reorder_point"] == 25
    assert context["below_reorder_point"] is True
    assert context["lead_time_days"] == 7
    assert context["supplier"]["supplier_name"] == "Acme"
    assert context["supplier"]["unit_cost"] == "11.50"


def test_the_forecast_context_carries_database_history(
    service: ClientService,
    token: str,
    estate: dict[str, Any],
    llm: ActingLLMClient,
) -> None:
    """14 days of 3 units each, read from sales_history rather than supplied."""
    run(service, token, "DEMAND_FORECAST", {"sku": "SKU-1", "warehouse_code": "WH-1"})

    context = llm.last_context
    assert context["recent_sales_total"] == 42
    assert context["historical_daily_sales"]["days_with_sales"] == 14
    assert context["baseline_forecast"] is not None
