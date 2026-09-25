"""The LLM gRPC servicer (Phase G).

Every test drives the servicer with a mocked Ollama adapter, so the suite
never needs a daemon or a model. What is being pinned down here is the
translation layer: protobuf in, prompt out, generated text back, and -- the
part most worth testing -- the mapping from each failure onto the status code
an application node will branch on.

Two guarantees get particular attention:

* **No traceback reaches a client.** An unexpected exception must arrive as a
  bare INTERNAL with a fixed message, with the detail left in this node's log.
* **The node cannot act.** It is asserted by import list rather than by
  intention: a servicer that cannot import a repository cannot write a row.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct

from llm import service as service_module
from llm.ollama_client import (
    OllamaGenerationError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from llm.prompts import DEMAND_FORECAST, INVENTORY_SUMMARY, REORDER_RECOMMENDATION
from llm.service import INTERNAL_ERROR_MESSAGE, LLMService
from proto import llm_pb2

ANSWER = "Recommend ordering 40 units of SKU-1 before the 7-day lead time bites."
MODEL = "phi3:mini"


class FakeContext:
    """Minimal ServicerContext stand-in.

    ``abort`` raises, exactly as grpc's real context does; without that the
    servicer would carry on past a failure and the tests would not reflect
    real behaviour. Copied from the Phase E/F suite so both read alike.
    """

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


class FakeOllamaClient:
    """Records what it was asked to generate, and answers or fails on cue."""

    def __init__(
        self, *, answer: str = ANSWER, raises: BaseException | None = None
    ) -> None:
        self.model = MODEL
        self.answer = answer
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.answer

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


def make_request(
    *,
    request_id: str = "req-1",
    query_type: str = DEMAND_FORECAST,
    query: str = "How much will we sell?",
    context: dict[str, Any] | None = None,
) -> llm_pb2.GetLLMAnswerRequest:
    request = llm_pb2.GetLLMAnswerRequest(
        request_id=request_id, query_type=query_type, query=query
    )
    if context:
        struct = Struct()
        struct.update(context)
        request.context.CopyFrom(struct)
    return request


def call(
    request: llm_pb2.GetLLMAnswerRequest, client: FakeOllamaClient | None = None
) -> tuple[llm_pb2.GetLLMAnswerResponse, FakeContext, FakeOllamaClient]:
    fake = client or FakeOllamaClient()
    context = FakeContext()
    response = LLMService(fake).getLLMAnswer(request, context)
    return response, context, fake


def expect_abort(
    request: llm_pb2.GetLLMAnswerRequest, client: FakeOllamaClient | None = None
) -> FakeContext:
    fake = client or FakeOllamaClient()
    context = FakeContext()
    with pytest.raises(_Aborted):
        LLMService(fake).getLLMAnswer(request, context)
    return context


# --- the three analyses ----------------------------------------------------
def test_demand_forecast_succeeds() -> None:
    response, _, fake = call(
        make_request(
            query_type=DEMAND_FORECAST,
            context={"product": {"sku": "SKU-1"}, "recent_daily_average": 4},
        )
    )

    assert response.answer == ANSWER
    assert response.model == MODEL
    assert "SKU-1" in fake.last_call["user_prompt"]


def test_reorder_recommendation_succeeds() -> None:
    response, _, fake = call(
        make_request(
            query_type=REORDER_RECOMMENDATION,
            context={"current_quantity": 12, "reorder_point": 25},
        )
    )

    assert response.answer == ANSWER
    assert "Current quantity on hand: 12" in fake.last_call["user_prompt"]


def test_inventory_summary_succeeds() -> None:
    response, _, fake = call(
        make_request(
            query_type=INVENTORY_SUMMARY,
            context={"totals": {"products": 12, "units": 480}},
        )
    )

    assert response.answer == ANSWER
    assert "Inventory totals" in fake.last_call["user_prompt"]


@pytest.mark.parametrize(
    "query_type", [DEMAND_FORECAST, REORDER_RECOMMENDATION, INVENTORY_SUMMARY]
)
def test_each_analysis_sends_its_own_prompt(query_type: str) -> None:
    _, _, fake = call(make_request(query_type=query_type, context={"product": {}}))

    assert fake.last_call["system_prompt"].startswith("You are an advisory")
    assert fake.last_call["user_prompt"].startswith("TASK:")


# --- the response contract -------------------------------------------------
@pytest.mark.parametrize("request_id", ["req-1", "abc-123", "a" * 64])
def test_the_request_id_comes_back_unchanged(request_id: str) -> None:
    """A node matches replies to in-flight requests by this value alone."""
    response, _, _ = call(make_request(request_id=request_id))

    assert response.request_id == request_id


def test_the_configured_model_is_reported() -> None:
    response, _, _ = call(make_request())

    assert response.model == MODEL


def test_only_the_generated_text_is_returned() -> None:
    response, _, _ = call(make_request(), FakeOllamaClient(answer="  spaced  "))

    assert response.answer == "  spaced  "


# --- request validation ----------------------------------------------------
@pytest.mark.parametrize("request_id", ["", "   ", "\t\n"])
def test_a_missing_request_id_is_rejected(request_id: str) -> None:
    context = expect_abort(make_request(request_id=request_id))

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "request_id" in context.details


def test_a_missing_request_id_never_reaches_the_model() -> None:
    fake = FakeOllamaClient()

    expect_abort(make_request(request_id=""), fake)

    assert fake.calls == []


@pytest.mark.parametrize(
    "query_type", ["", "   ", "demand_forecast", "FORECAST", "NONSENSE"]
)
def test_an_unsupported_query_type_is_rejected(query_type: str) -> None:
    context = expect_abort(make_request(query_type=query_type))

    assert context.code == grpc.StatusCode.INVALID_ARGUMENT
    assert "Supported values are" in context.details


def test_an_unsupported_query_type_never_reaches_the_model() -> None:
    fake = FakeOllamaClient()

    expect_abort(make_request(query_type="NONSENSE"), fake)

    assert fake.calls == []


def test_an_empty_context_is_still_a_valid_request() -> None:
    """Every context field is optional; the model is told the data is absent."""
    response, _, fake = call(make_request(context=None))

    assert response.answer == ANSWER
    assert "No data was supplied" in fake.last_call["user_prompt"]


def test_a_nested_struct_context_survives_conversion() -> None:
    """Struct -> dict goes through protobuf's own MessageToDict."""
    _, _, fake = call(
        make_request(
            query_type=REORDER_RECOMMENDATION,
            context={
                "supplier": {"name": "Acme", "unit_cost": "11.50"},
                "recent_demand": [3, 5, 4],
            },
        )
    )

    user_prompt = fake.last_call["user_prompt"]
    assert "Acme" in user_prompt
    assert '"unit_cost": "11.50"' in user_prompt
    # Struct stores numbers as doubles; the prompt layer spells integral
    # ones as integers again, so the model does not see "[3.0, 5.0, 4.0]".
    assert "[3, 5, 4]" in user_prompt


# --- failure mapping -------------------------------------------------------
def test_an_unreachable_runtime_is_unavailable() -> None:
    context = expect_abort(
        make_request(),
        FakeOllamaClient(raises=OllamaUnavailableError("Could not reach Ollama.")),
    )

    assert context.code == grpc.StatusCode.UNAVAILABLE
    assert "Could not reach Ollama." in context.details


def test_a_timeout_is_deadline_exceeded_not_unavailable() -> None:
    """Distinguishing the two tells an operator which knob to reach for."""
    context = expect_abort(
        make_request(),
        FakeOllamaClient(raises=OllamaTimeoutError("did not answer within 60 seconds")),
    )

    assert context.code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert "60 seconds" in context.details


def test_a_generation_failure_is_internal() -> None:
    context = expect_abort(
        make_request(),
        FakeOllamaClient(raises=OllamaGenerationError("Model 'phi3:mini' is missing.")),
    )

    assert context.code == grpc.StatusCode.INTERNAL
    assert "phi3:mini" in context.details


def test_an_unexpected_failure_leaks_nothing() -> None:
    """The one path where the real message is withheld from the caller."""
    context = expect_abort(
        make_request(), FakeOllamaClient(raises=RuntimeError("secret internal detail"))
    )

    assert context.code == grpc.StatusCode.INTERNAL
    assert context.details == INTERNAL_ERROR_MESSAGE
    assert "secret internal detail" not in context.details
    assert "Traceback" not in context.details


# --- logging ---------------------------------------------------------------
def test_a_request_is_logged_by_id_and_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="llm.service"):
        call(make_request(request_id="req-42", query_type=INVENTORY_SUMMARY))

    logged = caplog.text
    assert "req-42" in logged
    assert INVENTORY_SUMMARY in logged
    assert "completed" in logged


def test_a_failure_is_logged_with_its_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="llm.service"):
        expect_abort(
            make_request(request_id="req-43"),
            FakeOllamaClient(raises=OllamaUnavailableError("down")),
        )

    assert "req-43" in caplog.text


def test_business_data_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Prices, volumes and supplier terms belong in the database, not a log."""
    with caplog.at_level(logging.DEBUG, logger="llm.service"):
        call(
            make_request(
                query="What about ACME-SECRET-TERMS?",
                context={"supplier": {"name": "ACME-SECRET-TERMS"}},
            )
        )

    assert "ACME-SECRET-TERMS" not in caplog.text


# --- the boundary ----------------------------------------------------------
FORBIDDEN_ROOTS = {"app", "sqlalchemy", "sqlite3", "raft"}


def imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_the_servicer_cannot_reach_a_database_or_raft() -> None:
    """It cannot write a row because it cannot import anything that can."""
    for module in imported_modules(inspect.getsource(service_module)):
        assert module.split(".")[0] not in FORBIDDEN_ROOTS, module


def test_no_module_in_the_llm_package_can() -> None:
    package = Path(service_module.__file__).parent

    for path in package.glob("*.py"):
        for module in imported_modules(path.read_text(encoding="utf-8")):
            assert module.split(".")[0] not in FORBIDDEN_ROOTS, f"{path.name}: {module}"


def test_ollama_is_only_ever_reached_through_the_adapter() -> None:
    """The servicer talks to the adapter, never to the runtime itself."""
    imported = imported_modules(inspect.getsource(service_module))

    assert "ollama" not in imported
    assert "httpx" not in imported
