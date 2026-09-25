"""gRPC implementation of ``inventory.llm.v1.LLMService``.

This layer only translates: protobuf in, prompt out, model call, protobuf
back, with failures mapped onto gRPC status codes. It holds no state between
requests and reaches no database -- every fact an answer rests on arrives
inside the request, gathered from SQLite by the application server.

It mirrors :mod:`app.api.client_service` on purpose, including the ``_abort``
helper and the error-to-status table, so both servicers fail the same way.

What this service cannot do is as much the point as what it can. It cannot
create a product, move stock, place an order, record a restock or reach Raft,
because it imports nothing capable of any of that; a test asserts the import
list rather than trusting the claim.
"""

from __future__ import annotations

import logging
from typing import Any, Final, NoReturn

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from llm import prompts
from llm.ollama_client import (
    OllamaClient,
    OllamaGenerationError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from llm.prompts import PromptError
from proto import llm_pb2, llm_pb2_grpc

_logger = logging.getLogger(__name__)

#: Returned for a failure that is not one of the known categories below.
#: Deliberately opaque: a traceback belongs in this node's log, never on the
#: wire.
INTERNAL_ERROR_MESSAGE: Final[str] = "Internal error while generating an answer."

__all__ = ["LLMService"]


def _abort(
    context: grpc.ServicerContext, code: grpc.StatusCode, message: str
) -> NoReturn:
    """Terminate the RPC with ``code``.

    ``abort`` raises, so this never returns; the explicit ``raise`` after it
    documents that and keeps type checkers honest if a context implementation
    ever fails to raise.
    """
    context.abort(code, message)
    raise RuntimeError("unreachable: ServicerContext.abort must raise")


def _struct_to_dict(payload: Struct) -> dict[str, Any]:
    """Convert an inbound Struct to plain Python.

    ``MessageToDict`` rather than hand-written traversal, matching
    ``app.api.client_service``: it already handles nesting, nulls and the
    JSON-type mapping, and an ad-hoc parser would only differ from the
    application server's in ways nobody would notice until a payload broke.
    """
    return MessageToDict(payload)


class LLMService(llm_pb2_grpc.LLMServiceServicer):
    """Answers inventory questions for application nodes.

    The Ollama adapter is injected so tests can drive the whole RPC without a
    runtime, the same way :class:`~app.api.client_service.ClientService` takes
    a session factory. Production passes nothing and gets a client configured
    from the environment.

    ``json_output`` selects the structured reply shape declared by
    :mod:`llm.prompts`. It defaults to prose because the wire contract returns
    ``answer`` as text; the Streamlit phase can turn it on when it wants
    fields rather than paragraphs.
    """

    def __init__(
        self, client: OllamaClient | None = None, *, json_output: bool = False
    ) -> None:
        self._client = client or OllamaClient()
        self._json_output = json_output

    # Named for the generated servicer base class, which takes the spelling
    # from proto/llm.proto. gRPC dispatches on that name, so a PascalCase
    # method here would simply never be called.
    def getLLMAnswer(
        self, request: llm_pb2.GetLLMAnswerRequest, context: grpc.ServicerContext
    ) -> llm_pb2.GetLLMAnswerResponse:
        request_id = request.request_id.strip()
        query_type = request.query_type.strip()

        # Correlation id first: without one, a reply cannot be matched to its
        # request and nothing downstream can be traced.
        if not request_id:
            _abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "'request_id' is required.",
            )

        # Only the id and the query type are logged. The query and context
        # carry business data -- prices, volumes, supplier terms -- and a log
        # is the wrong place for it.
        _logger.info("llm request %s started (query_type=%s)", request_id, query_type)

        try:
            prompt = prompts.build_prompt(
                query_type,
                _struct_to_dict(request.context),
                query=request.query,
                json_output=self._json_output,
            )
        except PromptError as exc:
            # Covers an unsupported query type and a context that is not an
            # object: both are the caller's to fix.
            _logger.warning("llm request %s rejected: %s", request_id, exc)
            _abort(context, grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        try:
            answer = self._client.generate(
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.user_prompt,
                json_output=prompt.json_output,
            )
        except OllamaTimeoutError as exc:
            # Checked before OllamaUnavailableError, which it subclasses.
            _logger.warning("llm request %s timed out: %s", request_id, exc)
            _abort(context, grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))
        except OllamaUnavailableError as exc:
            _logger.warning("llm request %s could not reach the model: %s", request_id, exc)
            _abort(context, grpc.StatusCode.UNAVAILABLE, str(exc))
        except OllamaGenerationError as exc:
            # The runtime answered, but with an error or with nothing. Its
            # message names the model and reason, and carries no traceback.
            _logger.error("llm request %s failed to generate: %s", request_id, exc)
            _abort(context, grpc.StatusCode.INTERNAL, str(exc))
        except Exception:
            _logger.exception("llm request %s failed unexpectedly", request_id)
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

        _logger.info(
            "llm request %s completed (query_type=%s, %d characters)",
            request_id,
            query_type,
            len(answer),
        )

        return llm_pb2.GetLLMAnswerResponse(
            request_id=request_id,
            answer=answer,
            model=self._client.model,
        )
