"""gRPC implementation of ``inventory.client.v1.ClientService``.

This layer only translates: protobuf in, domain call, protobuf out, with
domain errors mapped onto gRPC status codes. Every authentication decision
belongs to :mod:`app.domain.auth`.

Error-reporting rule: clients are told *what* category of thing went wrong
and never *why* internally. Database errors are logged server-side with a
full traceback and reported as a bare ``INTERNAL``; no SQL, exception text,
credential, hash or stack trace crosses the wire.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final, NoReturn

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.domain import auth, inventory, orders
from app.domain.inventory import CatalogError
from app.storage.database import SessionLocal, session_scope
from app.storage.models import User
from proto import client_pb2, client_pb2_grpc

_logger = logging.getLogger(__name__)

#: Returned for any unexpected server-side failure. Deliberately opaque.
INTERNAL_ERROR_MESSAGE: Final[str] = "Internal server error."

#: Phase E exposes post/get/processBusinessRequest as authenticated stubs;
#: the operations behind them arrive in a later phase.
_NOT_IMPLEMENTED_TEMPLATE: Final[str] = (
    "'{rpc}' is authenticated but not implemented yet."
)

__all__ = ["ClientService", "build_user_info"]


def _abort(
    context: grpc.ServicerContext, code: grpc.StatusCode, message: str
) -> NoReturn:
    """Terminate the RPC with ``code``.

    ``abort`` raises, so this never returns; the explicit ``raise`` after it
    documents that and keeps type checkers honest if a context
    implementation ever fails to raise.
    """
    context.abort(code, message)
    raise RuntimeError("unreachable: ServicerContext.abort must raise")


def build_user_info(user: User) -> client_pb2.UserInfo:
    """Project a user row onto the wire message.

    Only the five safe columns are copied. ``password_hash`` has no field to
    land in, which is the point.
    """
    return client_pb2.UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
    )


def _to_timestamp(value) -> Timestamp:
    timestamp = Timestamp()
    timestamp.FromDatetime(value)
    return timestamp


def _struct_to_dict(payload: Struct) -> dict[str, Any]:
    """Convert an inbound Struct to plain Python.

    Numbers arrive as doubles and are re-typed by the domain validators;
    nothing here assumes a shape.
    """
    return MessageToDict(payload)


def _data_item(record_id: int, payload: dict[str, Any]) -> client_pb2.DataItem:
    """Build one DataItem from an already-serialized row.

    ``DataItem.id`` is the canonical identifier, and ``data`` carries the
    Struct-safe projection built by the domain layer -- the ORM object
    itself never reaches protobuf.
    """
    item = client_pb2.DataItem(id=str(record_id))
    item.data.update(payload)
    return item


def _resolve_action(resource: str, actions: Mapping[str, Any]) -> Any:
    """Look up the handler for ``resource``.

    Raises :class:`~app.domain.inventory.ValidationError` -- an unroutable
    resource is invalid input -- so it maps to INVALID_ARGUMENT through the
    same table as every other domain failure. Only ever called after the
    caller has been authenticated.
    """
    handler = actions.get(resource)
    if handler is None:
        known = ", ".join(sorted(actions))
        label = resource or "<empty>"
        raise inventory.ValidationError(
            f"Unknown resource '{label}'. Supported values are: {known}."
        )
    return handler


def _catalog_status(error: CatalogError) -> grpc.StatusCode:
    """Map a domain failure onto its gRPC status code."""
    for error_type, code in _CATALOG_STATUS_CODES.items():
        if isinstance(error, error_type):
            return code
    return grpc.StatusCode.INVALID_ARGUMENT


# --------------------------------------------------------------------------
# Action routing
#
# `resource` selects the operation, keeping the Phase E wire contract
# unchanged: the proto stays generic and each phase registers handlers here.
# --------------------------------------------------------------------------
def _create_product(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    product = inventory.create_product(db, payload)
    return f"Product '{product.sku}' created."


def _update_product(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    product = inventory.update_product(db, payload)
    return f"Product '{product.sku}' updated."


def _get_product(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    product = inventory.get_product_from_params(db, params)
    return [_data_item(product.id, inventory.serialize_product(product))]


def _list_products(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    return [
        _data_item(product.id, inventory.serialize_product(product))
        for product in inventory.list_products(db, params)
    ]


def _create_warehouse(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    warehouse = inventory.create_warehouse(db, payload)
    return f"Warehouse '{warehouse.code}' created."


def _get_warehouse(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    warehouse = inventory.get_warehouse_from_params(db, params)
    return [_data_item(warehouse.id, inventory.serialize_warehouse(warehouse))]


def _list_warehouses(
    db: DbSession, params: dict[str, Any]
) -> list[client_pb2.DataItem]:
    return [
        _data_item(warehouse.id, inventory.serialize_warehouse(warehouse))
        for warehouse in inventory.list_warehouses(db, params)
    ]


def _set_initial_stock(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    record = inventory.set_initial_stock(db, payload)
    return (
        f"Initial stock set for '{record.product.sku}' in "
        f"'{record.warehouse.code}': {record.quantity}."
    )


def _add_stock(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    record = inventory.add_stock(db, payload)
    return (
        f"Stock for '{record.product.sku}' in '{record.warehouse.code}' "
        f"is now {record.quantity}."
    )


def _get_stock(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    record = inventory.get_stock(db, params)
    return [_data_item(record.id, inventory.serialize_inventory(record))]


def _list_warehouse_inventory(
    db: DbSession, params: dict[str, Any]
) -> list[client_pb2.DataItem]:
    return [
        _data_item(record.id, inventory.serialize_inventory(record))
        for record in inventory.list_warehouse_inventory(db, params)
    ]


def _list_low_stock(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    return [
        _data_item(record.id, inventory.serialize_inventory(record))
        for record in inventory.list_low_stock(db, params)
    ]


def _create_supplier(db: DbSession, payload: dict[str, Any], _actor: User) -> str:
    supplier = inventory.create_supplier(db, payload)
    return f"Supplier '{supplier.name}' created."


def _associate_supplier_product(
    db: DbSession, payload: dict[str, Any], _actor: User
) -> str:
    link = inventory.associate_supplier_product(db, payload)
    return (
        f"Supplier '{link.supplier.name}' linked to product '{link.product.sku}' "
        f"at {link.unit_cost} with a {link.lead_time_days}-day lead time."
    )


def _list_suppliers(db: DbSession, params: dict[str, Any]) -> list[client_pb2.DataItem]:
    return [
        _data_item(supplier.id, inventory.serialize_supplier(supplier))
        for supplier in inventory.list_suppliers(db, params)
    ]


def _place_order(db: DbSession, payload: dict[str, Any], actor: User) -> str:
    order = orders.place_order(db, actor, payload)
    return (
        f"Order {order.order_number} placed: {len(order.items)} line(s), "
        f"total {order.total_amount}, status {order.status}."
    )


#: resource -> write handler. Each returns the message for StatusResponse.
POST_ACTIONS: Final[dict[str, Callable[[DbSession, dict[str, Any], User], str]]] = {
    "product.create": _create_product,
    "product.update": _update_product,
    "warehouse.create": _create_warehouse,
    "inventory.set_initial_stock": _set_initial_stock,
    "inventory.add_stock": _add_stock,
    "supplier.create": _create_supplier,
    "supplier.associate_product": _associate_supplier_product,
    "order.place": _place_order,
}

#: resource -> read handler. Each returns rows to serialize into DataItems.
GET_ACTIONS: Final[
    dict[str, Callable[[DbSession, dict[str, Any]], list[client_pb2.DataItem]]]
] = {
    "product": _get_product,
    "products": _list_products,
    "warehouse": _get_warehouse,
    "warehouses": _list_warehouses,
    "inventory.stock": _get_stock,
    "inventory.warehouse": _list_warehouse_inventory,
    "inventory.low_stock": _list_low_stock,
    "suppliers": _list_suppliers,
}

_CATALOG_STATUS_CODES: Final[dict[type[CatalogError], grpc.StatusCode]] = {
    # Listed before the broader classes: the request was well formed, the
    # warehouse simply cannot satisfy it right now.
    orders.InsufficientStockError: grpc.StatusCode.FAILED_PRECONDITION,
    inventory.NotFoundError: grpc.StatusCode.NOT_FOUND,
    inventory.DuplicateError: grpc.StatusCode.ALREADY_EXISTS,
    inventory.ValidationError: grpc.StatusCode.INVALID_ARGUMENT,
}


class ClientService(client_pb2_grpc.ClientServiceServicer):
    """Client-facing RPCs for one application node.

    The session factory is injected so tests can bind the service to a
    temporary database; production passes nothing and gets Phase D's
    ``SessionLocal``.
    """

    def __init__(self, session_factory: sessionmaker[DbSession] | None = None) -> None:
        self._session_factory = session_factory or SessionLocal

    # -- shared authentication ---------------------------------------------
    @contextmanager
    def _authenticated(
        self, raw_token: str, context: grpc.ServicerContext
    ) -> Iterator[tuple[DbSession, auth.AuthenticatedContext]]:
        """Run a block inside a transaction, with the caller authenticated.

        The single place every authenticated RPC validates a session, so the
        rules cannot drift between methods. The block sees an open
        transaction that commits on clean exit and rolls back otherwise.
        """
        try:
            with session_scope(self._session_factory) as db:
                authenticated = auth.authenticate(db, raw_token)
                yield db, authenticated
        except auth.AuthError:
            _abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                auth.GENERIC_SESSION_FAILURE_MESSAGE,
            )
        except SQLAlchemyError:
            _logger.exception("database failure during session validation")
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

    def _authenticated_stub(
        self, raw_token: str, context: grpc.ServicerContext, rpc_name: str
    ) -> NoReturn:
        """Authenticate, then report the RPC as not yet implemented.

        Order matters: an invalid token must fail as UNAUTHENTICATED even
        though the operation behind it does not exist yet, so a caller can
        never use an unimplemented RPC to probe whether a token is valid.
        """
        with self._authenticated(raw_token, context):
            pass
        _abort(
            context,
            grpc.StatusCode.UNIMPLEMENTED,
            _NOT_IMPLEMENTED_TEMPLATE.format(rpc=rpc_name),
        )

    # -- RPCs ---------------------------------------------------------------
    def login(
        self, request: client_pb2.LoginRequest, context: grpc.ServicerContext
    ) -> client_pb2.LoginResponse:
        # Whitespace-only usernames are malformed input. Passwords are not
        # stripped or rejected for whitespace -- spaces are legal in a
        # password, and only a genuinely empty one is malformed.
        if not request.username.strip() or not request.password:
            _abort(
                context,
                grpc.StatusCode.INVALID_ARGUMENT,
                "Both 'username' and 'password' are required.",
            )

        try:
            with session_scope(self._session_factory) as db:
                result = auth.login(db, request.username, request.password)
                response = client_pb2.LoginResponse(
                    success=True,
                    session_token=result.raw_token,
                    message="Login successful.",
                    user=build_user_info(result.user),
                    expires_at=_to_timestamp(result.expires_at),
                )
        except auth.InvalidCredentialsError:
            _abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                auth.GENERIC_AUTH_FAILURE_MESSAGE,
            )
        except SQLAlchemyError:
            _logger.exception("database failure during login")
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

        _logger.info("login succeeded for user id=%s", result.user.id)
        return response

    def logout(
        self, request: client_pb2.LogoutRequest, context: grpc.ServicerContext
    ) -> client_pb2.StatusResponse:
        with self._authenticated(request.session_token, context) as (db, authenticated):
            auth.revoke_session(db, authenticated)
            user_id = authenticated.user.id

        _logger.info("logout revoked one session for user id=%s", user_id)
        return client_pb2.StatusResponse(success=True, message="Logged out.")

    def post(
        self, request: client_pb2.PostRequest, context: grpc.ServicerContext
    ) -> client_pb2.StatusResponse:
        try:
            # Authenticate before resolving the action. The reverse order
            # would let an unauthenticated caller discover which resources
            # exist by comparing INVALID_ARGUMENT against UNAUTHENTICATED.
            with self._authenticated(request.session_token, context) as (
                db,
                authenticated,
            ):
                handler = _resolve_action(request.resource, POST_ACTIONS)
                message = handler(
                    db, _struct_to_dict(request.payload), authenticated.user
                )
        except CatalogError as exc:
            _abort(context, _catalog_status(exc), str(exc))

        return client_pb2.StatusResponse(success=True, message=message)

    def get(
        self, request: client_pb2.GetRequest, context: grpc.ServicerContext
    ) -> client_pb2.GetResponse:
        try:
            # Authenticated first, for the same reason as `post`.
            with self._authenticated(request.session_token, context) as (db, _):
                handler = _resolve_action(request.resource, GET_ACTIONS)
                # Serialized inside the transaction: rows are read here and
                # only plain values leave the block.
                items = handler(db, _struct_to_dict(request.params))
        except CatalogError as exc:
            _abort(context, _catalog_status(exc), str(exc))

        return client_pb2.GetResponse(
            success=True,
            message=f"{len(items)} record(s).",
            items=items,
        )

    def processBusinessRequest(
        self, request: client_pb2.BusinessRequest, context: grpc.ServicerContext
    ) -> client_pb2.BusinessResponse:
        self._authenticated_stub(
            request.session_token, context, "processBusinessRequest"
        )
