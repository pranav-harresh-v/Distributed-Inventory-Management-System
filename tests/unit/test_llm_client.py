"""The application server's client for the LLM node (Phase G).

Every test drives a mocked stub, so nothing here opens a socket or needs the
LLM node running. What is pinned down is the boundary in both directions:
what goes onto the wire, what comes back, and what each gRPC status becomes
by the time domain code sees it.

The architectural assertions at the bottom are the ones that matter most for
the project's constraints. The application server must reach a model *only*
through the LLM node, so this package may not import ``ollama`` and may not
know Ollama's port exists. Both are checked against the source rather than
taken on trust.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import grpc
import pytest

from app.api import llm_client as llm_client_module
from app.api.llm_client import (
    DEFAULT_PORT,
    RPC_TIMEOUT_MARGIN_SECONDS,
    LLMAnswer,
    LLMClient,
    LLMClientError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    get_llm_target,
    get_request_timeout,
)
from proto import llm_pb2

ANSWER = "Recommend ordering 40 units before the lead time bites."
MODEL = "phi3:mini"


class FakeRpcError(grpc.RpcError):
    """A gRPC failure with the status the real one carries."""

    def __init__(self, code: grpc.StatusCode, details: str = "boom") -> None:
        super().__init__(details)
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


class FakeStub:
    """Records the request it was given and answers or fails on cue."""

    def __init__(
        self,
        *,
        response: llm_pb2.GetLLMAnswerResponse | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self.response = response
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def getLLMAnswer(
        self, request: llm_pb2.GetLLMAnswerRequest, timeout: float | None = None
    ) -> llm_pb2.GetLLMAnswerResponse:
        self.calls.append({"request": request, "timeout": timeout})
        if self.raises is not None:
            raise self.raises
        if self.response is not None:
            return self.response
        return llm_pb2.GetLLMAnswerResponse(
            request_id=request.request_id, answer=ANSWER, model=MODEL
        )

    @property
    def last_request(self) -> llm_pb2.GetLLMAnswerRequest:
        return self.calls[-1]["request"]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LLM_GRPC_HOST", "LLM_GRPC_PORT", "LLM_REQUEST_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def build(**kwargs: Any) -> tuple[LLMClient, FakeStub]:
    stub = FakeStub(**kwargs)
    return LLMClient(stub=stub), stub


# --- configuration ---------------------------------------------------------
def test_the_default_target_is_the_llm_node_on_50052() -> None:
    assert get_llm_target() == "127.0.0.1:50052"
    assert DEFAULT_PORT == 50052


def test_the_target_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_GRPC_HOST", "10.0.0.7")
    monkeypatch.setenv("LLM_GRPC_PORT", "6000")

    assert get_llm_target() == "10.0.0.7:6000"
    assert LLMClient(stub=FakeStub()).target == "10.0.0.7:6000"


def test_an_explicit_target_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_GRPC_HOST", "10.0.0.7")

    assert LLMClient("localhost:1234", stub=FakeStub()).target == "localhost:1234"


def test_the_timeout_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "30")

    assert get_request_timeout() == 30.0


@pytest.mark.parametrize("value", ["not-a-number", "0", "-5"])
def test_an_unusable_timeout_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="LLM_REQUEST_TIMEOUT_SECONDS"):
        LLMClient(stub=FakeStub())


# --- request construction --------------------------------------------------
def test_the_request_carries_every_field() -> None:
    client, stub = build()

    client.get_llm_answer(
        "req-1", "DEMAND_FORECAST", "How many?", {"product": {"sku": "SKU-1"}}
    )

    request = stub.last_request
    assert request.request_id == "req-1"
    assert request.query_type == "DEMAND_FORECAST"
    assert request.query == "How many?"


def test_every_call_carries_a_finite_deadline() -> None:
    """An RPC without one parks a servicer thread forever if the far side wedges."""
    client, stub = build()

    client.get_llm_answer("req-1", "DEMAND_FORECAST")

    timeout = stub.calls[-1]["timeout"]
    assert timeout is not None
    assert timeout == client.rpc_deadline
    assert 0 < timeout < float("inf")


def test_the_deadline_outlasts_the_generation_budget() -> None:
    """So the LLM node's own timeout wins and the failure stays attributable."""
    client = LLMClient(timeout=30.0, stub=FakeStub())

    assert client.rpc_deadline == 30.0 + RPC_TIMEOUT_MARGIN_SECONDS
    assert client.rpc_deadline > client.timeout


def test_a_query_is_optional() -> None:
    client, stub = build()

    client.get_llm_answer("req-1", "INVENTORY_SUMMARY")

    assert stub.last_request.query == ""


# --- context conversion ----------------------------------------------------
def test_a_nested_context_becomes_a_struct() -> None:
    client, stub = build()

    client.get_llm_answer(
        "req-1",
        "REORDER_RECOMMENDATION",
        context={
            "current_quantity": 12,
            "supplier": {"name": "Acme", "unit_cost": "11.50"},
            "recent_demand": [3, 5, 4],
            "in_stock": True,
            "note": None,
        },
    )

    context = dict(stub.last_request.context)
    assert context["current_quantity"] == 12
    assert context["supplier"]["name"] == "Acme"
    assert list(context["recent_demand"]) == [3, 5, 4]
    assert context["in_stock"] is True
    assert context["note"] is None


def test_money_crosses_as_an_exact_string() -> None:
    """A double cannot hold 11.50 exactly, which is why it travels as text."""
    client, stub = build()

    client.get_llm_answer("req-1", "DEMAND_FORECAST", context={"unit_price": "1234.57"})

    assert dict(stub.last_request.context)["unit_price"] == "1234.57"


@pytest.mark.parametrize("context", [None, {}])
def test_an_absent_context_is_allowed(context: dict[str, Any] | None) -> None:
    client, stub = build()

    client.get_llm_answer("req-1", "INVENTORY_SUMMARY", context=context)

    assert len(stub.last_request.context) == 0


def test_a_decimal_in_the_context_is_reported_clearly() -> None:
    """Struct rejects Decimal; the caller gets told why, not a bare TypeError."""
    client, _ = build()

    with pytest.raises(LLMRequestError, match="JSON-compatible"):
        client.get_llm_answer(
            "req-1", "DEMAND_FORECAST", context={"unit_price": Decimal("11.50")}
        )


# --- successful response ---------------------------------------------------
def test_a_successful_answer_is_returned_without_protobuf() -> None:
    """Domain code should never grow an import from `proto`."""
    client, _ = build()

    result = client.get_llm_answer("req-1", "DEMAND_FORECAST")

    assert isinstance(result, LLMAnswer)
    assert result.answer == ANSWER
    assert result.model == MODEL


@pytest.mark.parametrize("request_id", ["req-1", "abc-123", "a" * 64])
def test_the_request_id_propagates_end_to_end(request_id: str) -> None:
    client, stub = build()

    result = client.get_llm_answer(request_id, "DEMAND_FORECAST")

    assert stub.last_request.request_id == request_id
    assert result.request_id == request_id


def test_a_mismatched_request_id_warns_but_keeps_the_answer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A far-side bug, but not a reason to throw away a good answer."""
    stub = FakeStub(
        response=llm_pb2.GetLLMAnswerResponse(
            request_id="someone-else", answer=ANSWER, model=MODEL
        )
    )
    client = LLMClient(stub=stub)

    with caplog.at_level(logging.WARNING, logger="app.api.llm_client"):
        result = client.get_llm_answer("req-1", "DEMAND_FORECAST")

    assert result.answer == ANSWER
    assert "req-1" in caplog.text


# --- failure mapping -------------------------------------------------------
def test_unavailable_becomes_an_unavailable_error() -> None:
    client, _ = build(
        raises=FakeRpcError(grpc.StatusCode.UNAVAILABLE, "failed to connect")
    )

    with pytest.raises(LLMUnavailableError, match="unavailable"):
        client.get_llm_answer("req-1", "DEMAND_FORECAST")


def test_deadline_exceeded_becomes_a_timeout_error() -> None:
    client, _ = build(raises=FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED))

    with pytest.raises(LLMTimeoutError, match="did not answer"):
        client.get_llm_answer("req-1", "DEMAND_FORECAST")


def test_invalid_argument_becomes_a_request_error() -> None:
    client, _ = build(
        raises=FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT, "Unknown query_type")
    )

    with pytest.raises(LLMRequestError, match="Unknown query_type"):
        client.get_llm_answer("req-1", "NONSENSE")


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNIMPLEMENTED,
        grpc.StatusCode.PERMISSION_DENIED,
    ],
)
def test_any_other_status_becomes_the_base_error(code: grpc.StatusCode) -> None:
    client, _ = build(raises=FakeRpcError(code))

    with pytest.raises(LLMClientError):
        client.get_llm_answer("req-1", "DEMAND_FORECAST")


def test_no_grpc_exception_reaches_the_caller() -> None:
    """Domain code catches LLMClientError, never grpc.RpcError."""
    for code in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INVALID_ARGUMENT,
        grpc.StatusCode.INTERNAL,
    ):
        client, _ = build(raises=FakeRpcError(code))
        with pytest.raises(LLMClientError):
            client.get_llm_answer("req-1", "DEMAND_FORECAST")


def test_every_failure_type_shares_one_base() -> None:
    for error in (LLMUnavailableError, LLMTimeoutError, LLMRequestError):
        assert issubclass(error, LLMClientError)


def test_a_failure_is_logged_with_its_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _ = build(raises=FakeRpcError(grpc.StatusCode.UNAVAILABLE))

    with (
        caplog.at_level(logging.WARNING, logger="app.api.llm_client"),
        pytest.raises(LLMUnavailableError),
    ):
        client.get_llm_answer("req-77", "DEMAND_FORECAST")

    assert "req-77" in caplog.text


# --- lifecycle -------------------------------------------------------------
def test_closing_an_injected_stub_is_harmless() -> None:
    client, _ = build()

    client.close()
    client.close()


def test_the_client_works_as_a_context_manager() -> None:
    with LLMClient(stub=FakeStub()) as client:
        assert client.get_llm_answer("req-1", "DEMAND_FORECAST").answer == ANSWER


def test_constructing_a_real_client_does_not_require_the_llm_node() -> None:
    """grpc channels connect lazily, so this must not block or raise."""
    client = LLMClient("127.0.0.1:1")
    try:
        assert client.target == "127.0.0.1:1"
    finally:
        client.close()


# --- the architectural boundary --------------------------------------------
def imported_modules(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def application_sources() -> list[Path]:
    root = Path(llm_client_module.__file__).resolve().parents[1]
    return sorted(root.rglob("*.py"))


def test_the_application_never_imports_ollama() -> None:
    """The app server reaches a model only through the LLM node, over gRPC."""
    for path in application_sources():
        for module in imported_modules(path.read_text(encoding="utf-8")):
            root = module.split(".")[0]
            assert root not in {"ollama", "httpx"}, f"{path.name}: {module}"


def code_constants(source: str) -> Iterator[Any]:
    """Every literal in real code, with docstrings and comments left out.

    These modules *discuss* Ollama's port in prose precisely because they
    must never dial it, so a raw text search would flag the documentation
    that promises the opposite. Comments never enter the syntax tree;
    docstrings do, and are skipped explicitly.
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

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) not in docstrings:
            yield node.value


def test_the_application_never_names_the_ollama_port() -> None:
    """11434 belongs to the LLM node alone; this side talks to 50052."""
    for path in application_sources():
        for constant in code_constants(path.read_text(encoding="utf-8")):
            if isinstance(constant, str):
                assert "11434" not in constant, f"{path.name}: {constant!r}"
            elif isinstance(constant, int):
                assert constant != 11434, path.name


def test_this_module_holds_no_database_code() -> None:
    for module in imported_modules(
        Path(llm_client_module.__file__).read_text(encoding="utf-8")
    ):
        root = module.split(".")[0]
        assert root not in {"sqlalchemy", "sqlite3"}, module
