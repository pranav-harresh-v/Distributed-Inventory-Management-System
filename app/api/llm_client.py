"""Client for the separate LLM node's ``inventory.llm.v1.LLMService``.

The outbound counterpart to :mod:`app.api.client_service`, which is why it
lives beside it: both modules are this node's gRPC boundary, one accepting
calls and one making them.

This is the application server's only route to a model. It holds a stub and
nothing else -- no ``ollama`` import, no HTTP client, no knowledge that port
11434 exists. Swapping the model runtime is the LLM node's business, and a
test asserts that this package cannot reach it directly.

Two details are worth reading before changing anything here.

**Every call is deadlined.** An RPC without a timeout blocks a servicer
thread forever when the far side wedges, and this node only has eight. The
deadline is derived from the same ``LLM_REQUEST_TIMEOUT_SECONDS`` the LLM node
uses for generation, plus a margin: see :data:`RPC_TIMEOUT_MARGIN_SECONDS`.

**Protobuf leaves at the boundary.** Callers receive an :class:`LLMAnswer`,
not a generated message, so domain code never grows an import from ``proto``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, Self

import grpc
from google.protobuf.struct_pb2 import Struct

from proto import llm_pb2, llm_pb2_grpc

_logger = logging.getLogger(__name__)

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 50052
DEFAULT_TIMEOUT_SECONDS: Final[float] = 60.0

#: The same variables the LLM node binds to, so one setting configures both
#: ends of the link rather than two that can disagree.
HOST_ENV_VAR: Final[str] = "LLM_GRPC_HOST"
PORT_ENV_VAR: Final[str] = "LLM_GRPC_PORT"
TIMEOUT_ENV_VAR: Final[str] = "LLM_REQUEST_TIMEOUT_SECONDS"

#: Added to the configured timeout to get this client's RPC deadline.
#:
#: Both processes read ``LLM_REQUEST_TIMEOUT_SECONDS``: the LLM node uses it
#: as its generation budget, and this client as its deadline. Identical values
#: would race -- whichever fires first decides the error -- and the caller
#: would usually see a bare local DEADLINE_EXCEEDED instead of the LLM node's
#: own message naming the budget it blew. The margin lets the far end lose
#: first, so the failure is attributable.
RPC_TIMEOUT_MARGIN_SECONDS: Final[float] = 5.0

__all__ = [
    "DEFAULT_PORT",
    "LLMAnswer",
    "LLMClient",
    "LLMClientError",
    "LLMRequestError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "get_llm_target",
    "get_request_timeout",
]


class LLMClientError(Exception):
    """Base class for failures reaching the LLM node.

    Separate from :class:`~app.domain.inventory.CatalogError` on purpose: an
    unreachable model is an infrastructure fault, not a business-rule
    violation, and the two map onto different gRPC statuses when the
    application server reports back to its own client.
    """


class LLMUnavailableError(LLMClientError):
    """The LLM node could not be reached, or has no model runtime behind it."""


class LLMTimeoutError(LLMClientError):
    """The LLM node did not answer within the deadline."""


class LLMRequestError(LLMClientError):
    """The LLM node rejected the request.

    Raised for an unsupported query type, a missing request id or a context
    that cannot become a Struct. Every one of those is a bug on this side of
    the wire, not a condition a user can fix.
    """


class LLMResponseError(LLMClientError):
    """The LLM node answered, but not with a usable answer.

    The RPC succeeded -- there is no gRPC status to read the failure from --
    yet the `answer` field came back blank. That makes it a protocol
    violation by the far side rather than a transport fault, which is why it
    is neither :class:`LLMUnavailableError` (retrying changes nothing) nor
    :class:`LLMRequestError` (this side sent a well-formed request).

    The current LLM node cannot produce this: ``llm/ollama_client.py``
    already rejects an empty completion before it can be returned. The guard
    exists because the application server must not depend on that remaining
    true. Trusting the answer would be the worst of the three failure modes
    in this module -- an empty advisory reaching a user as ``success = true``
    is indistinguishable from real advice, whereas an error is not.
    """


@dataclass(frozen=True)
class LLMAnswer:
    """One answer, with the protobuf message left behind at the boundary."""

    request_id: str
    answer: str
    model: str


def _env_int(name: str, default: int) -> int:
    """Read an integer setting.

    A local copy of the helper in ``app/server.py`` rather than an import:
    ``app.server`` imports the servicer, the servicer will import this module,
    and the cycle would only announce itself at startup.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def get_llm_target() -> str:
    """``host:port`` of the LLM node."""
    host = os.environ.get(HOST_ENV_VAR, "").strip() or DEFAULT_HOST
    return f"{host}:{_env_int(PORT_ENV_VAR, DEFAULT_PORT)}"


def get_request_timeout() -> float:
    """Generation budget configured for the LLM node, in seconds."""
    raw = os.environ.get(TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{TIMEOUT_ENV_VAR} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{TIMEOUT_ENV_VAR} must be positive, got {value}")
    return value


def _to_struct(context: Any) -> Struct:
    """Convert a plain mapping to a protobuf Struct.

    A Struct holds JSON types only. It rejects :class:`~decimal.Decimal`
    outright and stores numbers as doubles, which is why this project sends
    money as exact decimal strings -- the same rule the client-facing contract
    follows. A caller that forgets gets told which key broke rather than a
    bare TypeError from inside protobuf.
    """
    struct = Struct()
    if not context:
        return struct
    try:
        struct.update(context)
    except (TypeError, ValueError, AttributeError) as exc:
        raise LLMRequestError(
            "'context' must contain only JSON-compatible values. Decimals and "
            f"datetimes must be converted to strings first: {exc}"
        ) from exc
    return struct


class LLMClient:
    """Calls the LLM node over gRPC.

    The channel is created eagerly but connects lazily, so constructing a
    client costs nothing and does not require the LLM node to be running. The
    stub is injectable for tests, as the session factory is on
    :class:`~app.api.client_service.ClientService`.
    """

    def __init__(
        self,
        target: str | None = None,
        *,
        timeout: float | None = None,
        stub: Any = None,
    ) -> None:
        self._target = target if target is not None else get_llm_target()
        self._timeout = timeout if timeout is not None else get_request_timeout()
        self._channel: grpc.Channel | None = None

        if stub is not None:
            self._stub = stub
        else:
            self._channel = grpc.insecure_channel(self._target)
            self._stub = llm_pb2_grpc.LLMServiceStub(self._channel)

    @property
    def target(self) -> str:
        return self._target

    @property
    def timeout(self) -> float:
        """Generation budget the LLM node is expected to honour."""
        return self._timeout

    @property
    def rpc_deadline(self) -> float:
        """Deadline applied to the RPC itself, always finite."""
        return self._timeout + RPC_TIMEOUT_MARGIN_SECONDS

    def get_llm_answer(
        self,
        request_id: str,
        query_type: str,
        query: str = "",
        context: Any = None,
    ) -> LLMAnswer:
        """Ask the LLM node one question.

        ``context`` carries the facts the answer must rest on, gathered from
        this node's database by the caller; it is converted to a Struct here.
        ``request_id`` correlates the reply and is echoed back by the node.

        Raises :class:`LLMUnavailableError` if the node cannot be reached,
        :class:`LLMTimeoutError` if it overruns the deadline,
        :class:`LLMRequestError` if it rejects the request,
        :class:`LLMResponseError` if it returns a blank answer, and
        :class:`LLMClientError` for anything else.
        """
        request = llm_pb2.GetLLMAnswerRequest(
            request_id=request_id, query_type=query_type, query=query
        )
        request.context.CopyFrom(_to_struct(context))

        try:
            response = self._stub.getLLMAnswer(request, timeout=self.rpc_deadline)
        except grpc.RpcError as exc:
            raise self._translate(exc, request_id) from exc

        if response.request_id != request_id:
            # gRPC pairs a reply with its call on one stream, so this cannot
            # be a crossed response -- it means the far side echoed the wrong
            # id. Worth knowing about, not worth failing a good answer over.
            _logger.warning(
                "llm reply for %s carried request_id %r",
                request_id,
                response.request_id,
            )

        # A successful RPC is not yet a usable answer. Checked here, at the
        # boundary, so no caller downstream has to decide whether empty prose
        # counts as advice.
        if not response.answer.strip():
            _logger.error(
                "llm reply for %s carried no answer (model=%r)",
                request_id,
                response.model,
            )
            raise LLMResponseError(
                f"The LLM service at {self._target} returned an empty answer."
            )

        # An absent model tag is attribution metadata, not content. Warned
        # about rather than raised on, for the same reason as a mismatched
        # request id above: it is not worth discarding a good answer over.
        if not response.model.strip():
            _logger.warning("llm reply for %s named no model", request_id)

        return LLMAnswer(
            request_id=response.request_id,
            answer=response.answer,
            model=response.model,
        )

    def _translate(self, exc: grpc.RpcError, request_id: str) -> LLMClientError:
        """Map a gRPC status onto this module's exceptions."""
        code = exc.code() if callable(getattr(exc, "code", None)) else None
        details = exc.details() if callable(getattr(exc, "details", None)) else None
        detail = details or "no detail"

        _logger.warning("llm request %s failed: %s (%s)", request_id, code, detail)

        if code == grpc.StatusCode.UNAVAILABLE:
            return LLMUnavailableError(
                f"The LLM service at {self._target} is unavailable: {detail}"
            )
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return LLMTimeoutError(
                f"The LLM service at {self._target} did not answer within "
                f"{self.rpc_deadline:g} seconds."
            )
        if code == grpc.StatusCode.INVALID_ARGUMENT:
            return LLMRequestError(f"The LLM service rejected the request: {detail}")
        return LLMClientError(f"The LLM service failed ({code}): {detail}")

    def close(self) -> None:
        """Release the channel. Safe to call on an injected stub."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
