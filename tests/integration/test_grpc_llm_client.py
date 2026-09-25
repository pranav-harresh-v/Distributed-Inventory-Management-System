"""The application server's LLM client against the real LLM node.

The unit tests on either side mock the other, which leaves one gap: both
could be internally consistent and still disagree with each other. These
tests run the genuine servicer from :mod:`llm.server` and the genuine client
from :mod:`app.api.llm_client` over a real channel, so a field the caller
sets and the servicer never reads would show up here.

Only the Ollama adapter is faked. No daemon, no model, and the server binds
port 0 so it cannot collide with a running node.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from app.api.llm_client import (
    LLMClient,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.ollama_client import OllamaTimeoutError, OllamaUnavailableError
from llm.prompts import DEMAND_FORECAST, INVENTORY_SUMMARY, REORDER_RECOMMENDATION
from llm.server import create_server

ANSWER = "Stock covers roughly nine days of demand at the current rate."
MODEL = "phi3:mini"


class FakeOllamaClient:
    def __init__(self, *, raises: BaseException | None = None) -> None:
        self.model = MODEL
        self.raises = raises
        self.prompts: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> str:
        self.prompts.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return ANSWER


def connected(client: FakeOllamaClient) -> Iterator[LLMClient]:
    server, port = create_server(client, host="127.0.0.1", port=0, max_workers=2)
    server.start()
    try:
        with LLMClient(f"127.0.0.1:{port}", timeout=5.0) as llm:
            yield llm
    finally:
        server.stop(None).wait()


@pytest.fixture
def fake_ollama() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def llm(fake_ollama: FakeOllamaClient) -> Iterator[LLMClient]:
    yield from connected(fake_ollama)


@pytest.mark.parametrize(
    "query_type", [DEMAND_FORECAST, REORDER_RECOMMENDATION, INVENTORY_SUMMARY]
)
def test_each_analysis_round_trips(llm: LLMClient, query_type: str) -> None:
    result = llm.get_llm_answer(
        "req-1", query_type, "What should we do?", {"product": {"sku": "SKU-1"}}
    )

    assert result.request_id == "req-1"
    assert result.answer == ANSWER
    assert result.model == MODEL


def test_the_context_reaches_the_prompt_intact(
    llm: LLMClient, fake_ollama: FakeOllamaClient
) -> None:
    """Dict -> Struct -> dict -> prompt, across a process boundary."""
    llm.get_llm_answer(
        "req-1",
        REORDER_RECOMMENDATION,
        context={
            "current_quantity": 12,
            "supplier": {"name": "Acme", "unit_cost": "11.50"},
        },
    )

    user_prompt = fake_ollama.prompts[-1]["user_prompt"]
    assert "Current quantity on hand: 12" in user_prompt
    assert '"unit_cost": "11.50"' in user_prompt


def test_an_unknown_query_type_is_rejected_by_the_far_side(llm: LLMClient) -> None:
    with pytest.raises(LLMRequestError, match="Supported values are"):
        llm.get_llm_answer("req-1", "NOT_AN_ANALYSIS")


def test_a_missing_request_id_is_rejected_by_the_far_side(llm: LLMClient) -> None:
    with pytest.raises(LLMRequestError, match="request_id"):
        llm.get_llm_answer("", DEMAND_FORECAST)


def test_an_unreachable_runtime_surfaces_as_unavailable() -> None:
    client = FakeOllamaClient(raises=OllamaUnavailableError("Ollama is down."))

    for llm in connected(client):
        with pytest.raises(LLMUnavailableError):
            llm.get_llm_answer("req-1", DEMAND_FORECAST)


def test_a_generation_timeout_surfaces_as_a_timeout() -> None:
    """The far side's DEADLINE_EXCEEDED, not this side's deadline firing."""
    client = FakeOllamaClient(raises=OllamaTimeoutError("no answer in 60 seconds"))

    for llm in connected(client):
        with pytest.raises(LLMTimeoutError):
            llm.get_llm_answer("req-1", DEMAND_FORECAST)


def test_a_node_that_is_not_running_is_unavailable() -> None:
    """Nothing is listening on this port; the client must not hang."""
    with (
        LLMClient("127.0.0.1:1", timeout=1.0) as llm,
        pytest.raises(LLMUnavailableError),
    ):
        llm.get_llm_answer("req-1", DEMAND_FORECAST)
