"""The operations behind ``processBusinessRequest``.

The third dispatch table of the API layer, alongside ``POST_ACTIONS`` and
``GET_ACTIONS`` in :mod:`app.api.client_service`, and registered the same way:
an operation is a name, a context builder and the analysis it asks the model
for. It lives in its own module because these operations differ from the
other two tables in one important respect -- each one leaves this node to
call another service -- and that is worth keeping visible.

What an AI operation is, and is not:

* It **reads**. Every builder here runs against this node's own SQLite
  through the Phase F domain layer, and none of them writes. A reorder
  recommendation is advice about a restock, never a restock.
* The numbers are **ours**. The daily rate, the baseline forecast and the
  baseline reorder quantity are computed deterministically before the model
  is asked anything, and they are returned to the client beside the prose.
  Nothing parses the model's answer back into a quantity, so the system
  cannot act on a number a model invented.
* The answer is **advisory**, and says so: every response carries
  ``advisory: true`` next to the text.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

from sqlalchemy.orm import Session as DbSession

from app.api.llm_client import LLMAnswer
from app.domain import analytics, forecasting
from app.domain.inventory import ValidationError, require_mapping

__all__ = [
    "BUSINESS_ACTIONS",
    "BusinessOperation",
    "new_request_id",
    "prepare",
    "serialize_answer",
]

#: Optional free-text question an operator may attach to any operation. Held
#: apart from the rest of the payload because the context builders reject
#: fields they do not act on, and this one is for the model rather than for
#: them.
QUESTION_FIELD: Final[str] = "question"

#: ``REQ-`` plus 12 hex characters, matching the shape of an order number.
#: Server-generated when the client does not supply one, so every answer can
#: be correlated with the request that produced it even from a caller that
#: never bothered.
_REQUEST_ID_PREFIX: Final[str] = "REQ-"


@dataclass(frozen=True)
class BusinessOperation:
    """One AI-assisted operation: what to gather, and what to ask."""

    query_type: str
    build_context: Callable[[DbSession, dict[str, Any]], dict[str, Any]]
    description: str


#: operation -> handler. Named exactly as the LLM query types they map onto,
#: so the string a client sends, the analysis the prompt layer builds and the
#: field in the reply are all one value rather than three that can drift.
BUSINESS_ACTIONS: Final[dict[str, BusinessOperation]] = {
    "DEMAND_FORECAST": BusinessOperation(
        query_type="DEMAND_FORECAST",
        build_context=forecasting.build_demand_forecast_context,
        description="Expected demand for one product in one warehouse.",
    ),
    "REORDER_RECOMMENDATION": BusinessOperation(
        query_type="REORDER_RECOMMENDATION",
        build_context=forecasting.build_reorder_context,
        description="Whether to restock one product, and roughly how much.",
    ),
    "INVENTORY_SUMMARY": BusinessOperation(
        query_type="INVENTORY_SUMMARY",
        build_context=analytics.build_inventory_summary_context,
        description="Management summary of stock across the node.",
    ),
}


def new_request_id() -> str:
    """Generate a correlation id for a request that arrived without one."""
    return f"{_REQUEST_ID_PREFIX}{uuid.uuid4().hex[:12].upper()}"


def resolve_operation(operation: str) -> BusinessOperation:
    """Look up the handler for ``operation``.

    Raises :class:`~app.domain.inventory.ValidationError` -- an unroutable
    operation is invalid input -- so it maps to INVALID_ARGUMENT through the
    same table as every other domain failure, exactly as ``_resolve_action``
    does for ``post`` and ``get``. Only ever called after the caller has been
    authenticated.
    """
    handler = BUSINESS_ACTIONS.get(operation)
    if handler is None:
        known = ", ".join(sorted(BUSINESS_ACTIONS))
        label = operation or "<empty>"
        raise ValidationError(
            f"Unknown operation '{label}'. Supported values are: {known}."
        )
    return handler


def prepare(
    db: DbSession, operation: str, payload: Any
) -> tuple[BusinessOperation, dict[str, Any], str]:
    """Resolve the operation and gather its context from the database.

    Everything here happens inside the caller's transaction; the model call
    that follows deliberately does not. Returns the operation, the context to
    send, and the operator's question if they asked one.
    """
    handler = resolve_operation(operation)
    data = require_mapping(payload or {}, "payload")

    question = data.get(QUESTION_FIELD, "")
    if not isinstance(question, str):
        raise ValidationError(f"'{QUESTION_FIELD}' must be a string.")

    arguments = {key: value for key, value in data.items() if key != QUESTION_FIELD}
    return handler, handler.build_context(db, arguments), question.strip()


def serialize_answer(
    operation: BusinessOperation, answer: LLMAnswer, context: dict[str, Any]
) -> dict[str, Any]:
    """Build the reply payload.

    The deterministic context travels back with the prose on purpose. It is
    the auditable half of the answer -- a client can show the figures the
    advice was based on, and a reader can tell a measured number from a
    written one. It is also what a later Streamlit view renders without
    asking the model anything a second time.
    """
    return {
        "query_type": operation.query_type,
        "answer": answer.answer,
        "model": answer.model,
        # Stated in the payload, not only in the prose: nothing in this reply
        # has changed any inventory, and no part of it will.
        "advisory": True,
        "context": context,
    }
