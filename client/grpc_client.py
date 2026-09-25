"""gRPC transport for the Streamlit client.

This module is the client's **only** way to reach anything. It holds no
Streamlit import, which is what makes it testable without a browser, and it
holds no database import, which is what makes the architecture true rather
than merely intended:

    browser --HTTP--> Streamlit :8501 --gRPC--> application server :50051
                                                       |         |
                                                    SQLite    gRPC :50052 --> LLM

The client never opens SQLite, never imports a repository, and has no address
for the LLM node. AI answers arrive through ``processBusinessRequest`` on the
application server, which is the node that owns both the data and the LLM
channel. A Streamlit process that could dial :50052 itself would be able to
ask a model about inventory it had not been authenticated to read.

Two conventions are inherited rather than reinvented:

* **Resources, not methods.** ``proto/client.proto`` is generic: ``post`` and
  ``get`` dispatch on a ``resource`` string and carry payloads as
  ``google.protobuf.Struct``. The helpers at the bottom of this file name
  those strings in one place so a typo is a missing function rather than a
  runtime ``INVALID_ARGUMENT``.
* **Money is a string.** A Struct stores numbers as doubles, so prices cross
  the wire as exact decimal strings. This module passes them through
  untouched; it never parses one into a float to display it.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any, Final, Self

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from proto import client_pb2, client_pb2_grpc

#: Where the application server listens. Matches ``app/server.py``'s defaults
#: and reads the same environment variables, so one setting moves both ends.
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 50051

HOST_ENV_VAR: Final[str] = "INVENTORY_GRPC_HOST"
PORT_ENV_VAR: Final[str] = "INVENTORY_GRPC_PORT"

#: Ordinary CRUD is fast and local. The AI operations are not: generation on
#: a CPU-bound Phi-3 can take the better part of a minute, and the
#: application server already allows for that plus a margin. A single short
#: deadline here would abort a request the backend was still happily serving.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 15.0
BUSINESS_TIMEOUT_SECONDS: Final[float] = 120.0

__all__ = [
    "AI_OPERATIONS",
    "AuthenticationError",
    "ConflictError",
    "InvalidRequestError",
    "InventoryClient",
    "NotFoundError",
    "PreconditionError",
    "ServerError",
    "Session",
    "TransportError",
    "TransportTimeout",
    "get_server_target",
    "restore_whole_numbers",
]

#: The three analyses the application server exposes. Spelled exactly as
#: ``app.api.business_service.BUSINESS_ACTIONS`` spells them -- matching is
#: exact on the server, so a near-miss here is an INVALID_ARGUMENT.
AI_OPERATIONS: Final[tuple[str, ...]] = (
    "DEMAND_FORECAST",
    "REORDER_RECOMMENDATION",
    "INVENTORY_SUMMARY",
)


# ---------------------------------------------------------------------------
# Errors
#
# One exception per gRPC status the application server actually returns. The
# UI then renders a category rather than re-deriving meaning from a status
# code, and an unmapped status still arrives as something catchable.
# ---------------------------------------------------------------------------
class ClientError(Exception):
    """Base class for anything that went wrong talking to the server."""

    def __init__(self, message: str, code: grpc.StatusCode | None = None) -> None:
        super().__init__(message)
        self.code = code


class AuthenticationError(ClientError):
    """UNAUTHENTICATED: no session, a rejected one, or one that expired."""


class InvalidRequestError(ClientError):
    """INVALID_ARGUMENT: the request was malformed or named nothing known."""


class NotFoundError(ClientError):
    """NOT_FOUND: the product, warehouse or stock row does not exist."""


class ConflictError(ClientError):
    """ALREADY_EXISTS: a unique value -- a SKU, a code -- is taken."""


class PreconditionError(ClientError):
    """FAILED_PRECONDITION: well formed, but the warehouse cannot satisfy it.

    In practice this is insufficient stock. It is deliberately distinct from
    :class:`InvalidRequestError`: the request was correct and may well succeed
    later, so the UI should say "not enough stock", not "bad input".
    """


class TransportTimeout(ClientError):
    """DEADLINE_EXCEEDED: the server did not answer in time."""


class TransportError(ClientError):
    """UNAVAILABLE: the application server is not reachable."""


class ServerError(ClientError):
    """INTERNAL or any status not mapped above."""


#: gRPC status -> exception. Mirrors the server's own mapping tables in
#: ``app/api/client_service.py``, read from the other side of the wire.
_STATUS_ERRORS: Final[dict[grpc.StatusCode, type[ClientError]]] = {
    grpc.StatusCode.UNAUTHENTICATED: AuthenticationError,
    grpc.StatusCode.INVALID_ARGUMENT: InvalidRequestError,
    grpc.StatusCode.NOT_FOUND: NotFoundError,
    grpc.StatusCode.ALREADY_EXISTS: ConflictError,
    grpc.StatusCode.FAILED_PRECONDITION: PreconditionError,
    grpc.StatusCode.DEADLINE_EXCEEDED: TransportTimeout,
    grpc.StatusCode.UNAVAILABLE: TransportError,
}

#: Shown instead of the server's own text for the two infrastructure faults,
#: whose real messages are channel diagnostics rather than anything a user of
#: an inventory system can act on.
_FALLBACK_MESSAGES: Final[dict[grpc.StatusCode, str]] = {
    grpc.StatusCode.UNAVAILABLE: (
        "The application server is not reachable. Start it with "
        "`python -m app.server` and try again."
    ),
    grpc.StatusCode.DEADLINE_EXCEEDED: (
        "The application server did not answer in time. If this was an AI "
        "request, the LLM node may still be loading the model."
    ),
}


def translate(error: grpc.RpcError) -> ClientError:
    """Turn a gRPC failure into this module's vocabulary.

    Kept a free function so tests can exercise the mapping without a channel.
    """
    code = error.code() if callable(getattr(error, "code", None)) else None
    details = error.details() if callable(getattr(error, "details", None)) else None

    message = _FALLBACK_MESSAGES.get(code) or details or "The request failed."
    return _STATUS_ERRORS.get(code, ServerError)(message, code)


# ---------------------------------------------------------------------------
# Configuration and conversion
# ---------------------------------------------------------------------------
def get_server_target() -> str:
    """The application server to dial, from the environment or the default."""
    host = os.environ.get(HOST_ENV_VAR, "").strip() or DEFAULT_HOST
    raw_port = os.environ.get(PORT_ENV_VAR, "").strip()
    if not raw_port:
        return f"{host}:{DEFAULT_PORT}"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"{PORT_ENV_VAR} must be an integer, got {raw_port!r}") from exc
    return f"{host}:{port}"


def to_struct(payload: dict[str, Any] | None) -> Struct:
    """Build a Struct, dropping keys the caller left empty.

    Empty means "not supplied" here, because a Streamlit text input returns
    `""` rather than nothing when the operator skips it. The domain layer
    rejects unknown *and* malformed fields, so sending `description: ""` for
    an untouched box would turn a blank form field into a stored empty
    string. `False` and `0` are values, not blanks, and are kept.
    """
    struct = Struct()
    if payload:
        struct.update({k: v for k, v in payload.items() if v != "" and v is not None})
    return struct


def restore_whole_numbers(value: Any) -> Any:
    """Undo the Struct round trip that turns every integer into a double.

    ``google.protobuf.Struct`` has one numeric type, so a quantity of 150
    leaves the server as an int and arrives as ``150.0``. Rendered straight
    into a table that reads "150.0 units in stock", and a reorder point of
    "40.0" -- wrong for values the schema stores as INTEGER.

    Restoring the integral ones is lossless and is done for the same reason
    ``llm/prompts.py`` does it before showing figures to the model: the wire
    format's limitation should not become something a reader has to decode.

    Money is unaffected: prices cross as exact decimal strings precisely
    because a double cannot hold them, so they never enter this path as
    numbers. Booleans are excluded explicitly -- ``bool`` is a subclass of
    ``int`` and ``True.is_integer()`` would otherwise turn a flag into 1.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: restore_whole_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [restore_whole_numbers(item) for item in value]
    return value


def _items_to_dicts(items: Any) -> list[dict[str, Any]]:
    """Project repeated DataItems onto plain dicts for rendering."""
    return [restore_whole_numbers(MessageToDict(item.data)) for item in items]


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class Session:
    """What a successful login returns. Note what is absent: the password.

    The credential is used to obtain this and is then out of scope -- it is
    never stored on the session, never held in Streamlit state, and never
    sent again. Every later call authenticates with ``token`` alone.
    """

    __slots__ = ("email", "expires_at", "role", "token", "username")

    def __init__(
        self,
        token: str,
        username: str,
        role: str,
        email: str = "",
        expires_at: str = "",
    ) -> None:
        self.token = token
        self.username = username
        self.role = role
        self.email = email
        self.expires_at = expires_at


class InventoryClient:
    """A connection to one application node.

    ``stub`` is injectable for the same reason ``ClientService`` takes a
    session factory: it lets the whole client be tested without a server, a
    port or a browser.
    """

    def __init__(
        self,
        target: str | None = None,
        *,
        stub: Any = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._target = target or get_server_target()
        self._timeout = timeout
        self._channel: grpc.Channel | None = None
        if stub is not None:
            self._stub = stub
        else:
            # Channels connect lazily, so constructing this does not require
            # the server to be up -- the UI can render a login page and
            # report the outage only when someone tries to use it.
            self._channel = grpc.insecure_channel(self._target)
            self._stub = client_pb2_grpc.ClientServiceStub(self._channel)

    @property
    def target(self) -> str:
        return self._target

    # -- session --------------------------------------------------------
    def login(self, username: str, password: str) -> Session:
        """Exchange a username and password for a session token.

        The password is a local argument and nothing more: it is not stored
        on the returned object and not retained by this client.
        """
        try:
            response = self._stub.login(
                client_pb2.LoginRequest(username=username, password=password),
                timeout=self._timeout,
            )
        except grpc.RpcError as exc:
            raise translate(exc) from exc

        return Session(
            token=response.session_token,
            username=response.user.username,
            role=response.user.role,
            email=response.user.email,
            expires_at=response.expires_at.ToDatetime().isoformat(timespec="seconds")
            if response.HasField("expires_at")
            else "",
        )

    def logout(self, token: str) -> str:
        """Revoke the session on the server.

        Called before the local state is cleared. Dropping the token locally
        would only make the browser forget it; the row would stay valid on
        the node until it expired, so a copied token would still work.
        """
        try:
            response = self._stub.logout(
                client_pb2.LogoutRequest(session_token=token), timeout=self._timeout
            )
        except grpc.RpcError as exc:
            raise translate(exc) from exc
        return response.message

    # -- generic verbs --------------------------------------------------
    def post(
        self, token: str, resource: str, payload: dict[str, Any] | None = None
    ) -> str:
        """Run a write action. Returns the server's confirmation message."""
        request = client_pb2.PostRequest(session_token=token, resource=resource)
        request.payload.CopyFrom(to_struct(payload))
        try:
            response = self._stub.post(request, timeout=self._timeout)
        except grpc.RpcError as exc:
            raise translate(exc) from exc
        return response.message

    def get(
        self, token: str, resource: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Run a read action. Returns the rows as plain dicts."""
        request = client_pb2.GetRequest(session_token=token, resource=resource)
        request.params.CopyFrom(to_struct(params))
        try:
            response = self._stub.get(request, timeout=self._timeout)
        except grpc.RpcError as exc:
            raise translate(exc) from exc
        return _items_to_dicts(response.items)

    def business_request(
        self, token: str, operation: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Ask the application server for one AI-assisted analysis.

        The request goes to :50051, never to the LLM node. The application
        server authenticates the caller, reads its own SQLite, computes every
        figure deterministically, and only then asks the model to explain
        them. The reply carries both halves, so the UI can show the measured
        numbers beside the prose and label the prose as advisory.
        """
        request = client_pb2.BusinessRequest(session_token=token, operation=operation)
        request.payload.CopyFrom(to_struct(payload))
        try:
            response = self._stub.processBusinessRequest(
                request, timeout=BUSINESS_TIMEOUT_SECONDS
            )
        except grpc.RpcError as exc:
            raise translate(exc) from exc

        answer = restore_whole_numbers(MessageToDict(response.data))
        answer.setdefault("request_id", response.request_id)
        return answer

    # -- resource helpers -----------------------------------------------
    #
    # Thin by design: each one names a resource string registered in
    # `POST_ACTIONS` / `GET_ACTIONS` on the server. Keeping them here means
    # the UI never spells a resource, and adding a backend action costs one
    # function rather than a scattered literal.
    def list_products(self, token: str, **params: Any) -> list[dict[str, Any]]:
        return self.get(token, "products", params)

    def create_product(self, token: str, **fields: Any) -> str:
        return self.post(token, "product.create", fields)

    def update_product(self, token: str, **fields: Any) -> str:
        return self.post(token, "product.update", fields)

    def list_warehouses(self, token: str, **params: Any) -> list[dict[str, Any]]:
        return self.get(token, "warehouses", params)

    def create_warehouse(self, token: str, **fields: Any) -> str:
        return self.post(token, "warehouse.create", fields)

    def list_warehouse_inventory(
        self, token: str, **params: Any
    ) -> list[dict[str, Any]]:
        return self.get(token, "inventory.warehouse", params)

    def list_low_stock(self, token: str, **params: Any) -> list[dict[str, Any]]:
        return self.get(token, "inventory.low_stock", params)

    def set_initial_stock(self, token: str, **fields: Any) -> str:
        return self.post(token, "inventory.set_initial_stock", fields)

    def add_stock(self, token: str, **fields: Any) -> str:
        return self.post(token, "inventory.add_stock", fields)

    def list_suppliers(self, token: str, **params: Any) -> list[dict[str, Any]]:
        return self.get(token, "suppliers", params)

    def create_supplier(self, token: str, **fields: Any) -> str:
        return self.post(token, "supplier.create", fields)

    def associate_supplier_product(self, token: str, **fields: Any) -> str:
        return self.post(token, "supplier.associate_product", fields)

    def place_order(
        self, token: str, warehouse_code: str, items: list[dict[str, Any]]
    ) -> str:
        """Place one order.

        ``items`` is a list of ``{"sku": ..., "quantity": ...}``. The server
        consolidates duplicate lines, decrements stock with a guarded
        conditional UPDATE and writes the sales rows, all in one transaction.
        Nothing about that arithmetic happens here: this client sends the
        request and reports what came back.
        """
        return self.post(
            token, "order.place", {"warehouse_code": warehouse_code, "items": items}
        )

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
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
