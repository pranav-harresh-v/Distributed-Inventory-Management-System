"""The LLM node over a real gRPC channel.

The unit tests call the servicer directly; these prove the other half -- that
``create_server`` binds, registers the generated servicer and produces status
codes an application node can actually branch on once they have crossed a
socket. A code that is right in-process but lost in transit would not show up
anywhere else.

The Ollama adapter is still faked. Nothing here starts a daemon or loads a
model, and the server binds port 0 so it cannot collide with a running node.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import grpc
import pytest
from google.protobuf.struct_pb2 import Struct

from llm.ollama_client import OllamaTimeoutError, OllamaUnavailableError
from llm.prompts import DEMAND_FORECAST, INVENTORY_SUMMARY, REORDER_RECOMMENDATION
from llm.server import create_server
from proto import llm_pb2, llm_pb2_grpc

ANSWER = "Demand looks steady at about four units a day."
MODEL = "phi3:mini"


class FakeOllamaClient:
    def __init__(self, *, raises: BaseException | None = None) -> None:
        self.model = MODEL
        self.raises = raises

    def generate(self, **_kwargs: Any) -> str:
        if self.raises is not None:
            raise self.raises
        return ANSWER


def serve(client: FakeOllamaClient) -> Iterator[llm_pb2_grpc.LLMServiceStub]:
    server, port = create_server(client, host="127.0.0.1", port=0, max_workers=2)
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            yield llm_pb2_grpc.LLMServiceStub(channel)
    finally:
        server.stop(None).wait()


@pytest.fixture
def stub() -> Iterator[llm_pb2_grpc.LLMServiceStub]:
    yield from serve(FakeOllamaClient())


def request_for(
    query_type: str, context: dict[str, Any] | None = None
) -> llm_pb2.GetLLMAnswerRequest:
    request = llm_pb2.GetLLMAnswerRequest(
        request_id="req-1", query_type=query_type, query="What should we do?"
    )
    if context:
        struct = Struct()
        struct.update(context)
        request.context.CopyFrom(struct)
    return request


@pytest.mark.parametrize(
    "query_type", [DEMAND_FORECAST, REORDER_RECOMMENDATION, INVENTORY_SUMMARY]
)
def test_each_analysis_answers_over_the_wire(
    stub: llm_pb2_grpc.LLMServiceStub, query_type: str
) -> None:
    response = stub.getLLMAnswer(request_for(query_type, {"product": {"sku": "SKU-1"}}))

    assert response.request_id == "req-1"
    assert response.answer == ANSWER
    assert response.model == MODEL


def test_the_server_binds_an_ephemeral_port() -> None:
    """Port 0 must report the port actually bound, as app/server.py does."""
    server, port = create_server(FakeOllamaClient(), host="127.0.0.1", port=0)
    try:
        assert port > 0
    finally:
        server.stop(None).wait()


def test_a_missing_request_id_is_invalid_argument(
    stub: llm_pb2_grpc.LLMServiceStub,
) -> None:
    request = request_for(DEMAND_FORECAST)
    request.request_id = ""

    with pytest.raises(grpc.RpcError) as exc:
        stub.getLLMAnswer(request)

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_an_unknown_query_type_is_invalid_argument(
    stub: llm_pb2_grpc.LLMServiceStub,
) -> None:
    with pytest.raises(grpc.RpcError) as exc:
        stub.getLLMAnswer(request_for("NOT_A_REAL_ANALYSIS"))

    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "Supported values are" in exc.value.details()


def test_an_unreachable_runtime_is_unavailable() -> None:
    client = FakeOllamaClient(raises=OllamaUnavailableError("Could not reach Ollama."))

    for stub in serve(client):
        with pytest.raises(grpc.RpcError) as exc:
            stub.getLLMAnswer(request_for(DEMAND_FORECAST))

        assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_a_timeout_is_deadline_exceeded() -> None:
    client = FakeOllamaClient(raises=OllamaTimeoutError("no answer in 60 seconds"))

    for stub in serve(client):
        with pytest.raises(grpc.RpcError) as exc:
            stub.getLLMAnswer(request_for(DEMAND_FORECAST))

        assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_an_unexpected_failure_carries_no_traceback() -> None:
    client = FakeOllamaClient(raises=RuntimeError("secret internal detail"))

    for stub in serve(client):
        with pytest.raises(grpc.RpcError) as exc:
            stub.getLLMAnswer(request_for(DEMAND_FORECAST))

        assert exc.value.code() == grpc.StatusCode.INTERNAL
        assert "secret internal detail" not in exc.value.details()
        assert "Traceback" not in exc.value.details()
