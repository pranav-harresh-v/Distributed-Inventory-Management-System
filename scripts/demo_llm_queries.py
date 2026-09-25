"""Live demonstration of the Phase G LLM node answering with Phi-3.

Runs the three Milestone 1 analyses against a running LLM service and prints
what the model said. This is the one place in the project where a real model
actually generates: the automated suite fakes the runtime on purpose, so it
can pass on a machine with no Ollama installed.

Usage (with the LLM node already running)::

    python -m llm.server                        # in another terminal
    python scripts/demo_llm_queries.py
    python scripts/demo_llm_queries.py --address 127.0.0.1:50052
    python scripts/demo_llm_queries.py --only REORDER_RECOMMENDATION
    python scripts/demo_llm_queries.py --show-context

What this demonstrates, and what it deliberately does not:

* It speaks **gRPC to the LLM node and nothing else**. It does not import
  ``ollama``, does not know Ollama's port, and could not reach the model any
  other way. Proving that link works is the entire point.
* It **touches no database**. The contexts below are mock data in exactly the
  shape ``app.domain.forecasting`` and ``app.domain.analytics`` build from
  SQLite, which is what lets this run without a seeded node -- and what the
  assignment means by "demand prediction based on historical data (mock)".
  Nothing here can create an order, a restock record or a stock movement,
  because it holds no session and never calls the application server.

Exits 0 when every query is answered, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

# Make the project importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.llm_client import (
    LLMAnswer,
    LLMClient,
    LLMClientError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.storage.models import utc_now

DEFAULT_ADDRESS = "127.0.0.1:50052"

#: Phi-3 Mini on a CPU is not fast. A first call also pays for loading the
#: model into memory, which can dominate everything after it.
DEFAULT_TIMEOUT_SECONDS = 120.0

RULE = "=" * 78
THIN_RULE = "-" * 78


def _recent_days(units: list[int]) -> list[dict[str, Any]]:
    """Turn a run of daily quantities into dated points ending yesterday."""
    today = utc_now().date()
    span = len(units)
    return [
        {"date": (today - timedelta(days=span - index)).isoformat(), "units": value}
        for index, value in enumerate(units)
    ]


#: Fourteen days of a product selling roughly ten a day and accelerating.
_DAILY_UNITS = [7, 6, 9, 8, 7, 10, 9, 11, 10, 13, 12, 14, 13, 16]

PRODUCT = {"sku": "SKU-1042", "name": "USB-C Charging Cable 2m"}
WAREHOUSE = {"code": "WH-BLR", "name": "Bengaluru Main"}

DEMAND_FORECAST_CONTEXT: dict[str, Any] = {
    "product": PRODUCT,
    "warehouse": WAREHOUSE,
    "forecast_horizon_days": 14,
    "historical_daily_sales": {
        "window_days": 30,
        "days_with_sales": len(_DAILY_UNITS),
        "days_shown": len(_DAILY_UNITS),
        "days": _recent_days(_DAILY_UNITS),
    },
    "recent_sales_total": sum(_DAILY_UNITS),
    "recent_daily_average": round(sum(_DAILY_UNITS) / 30, 3),
    "trend": {"direction": "rising", "change_percent": 42.9},
    "baseline_forecast": 69,
    "data_quality": "adequate",
    "history_window_days": 30,
}

REORDER_CONTEXT: dict[str, Any] = {
    "product": PRODUCT,
    "warehouse": WAREHOUSE,
    "current_quantity": 42,
    "reorder_point": 120,
    "safety_stock": 60,
    "below_reorder_point": True,
    "recent_demand": {
        "window_days": 30,
        "units": sum(_DAILY_UNITS),
        "daily_average": round(sum(_DAILY_UNITS) / 30, 3),
        "trend": "rising",
        "data_quality": "adequate",
    },
    "forecast_demand": 35,
    "lead_time_days": 7,
    # Money is an exact decimal string, never a number: a double cannot hold
    # 112.50 exactly, and the same rule applies on every wire in this project.
    "supplier": {
        "supplier_name": "Acme Components",
        "unit_cost": "112.50",
        "lead_time_days": 7,
        "minimum_order_qty": 50,
        "is_preferred": True,
    },
    "baseline_reorder_quantity": 103,
    "baseline_rationale": (
        "5.0 units/day over a 7-day lead time, plus 60 safety stock, "
        "less 42 on hand"
    ),
}

INVENTORY_SUMMARY_CONTEXT: dict[str, Any] = {
    "totals": {
        "products_in_catalogue": 48,
        "warehouses": 3,
        "warehouses_included": 3,
        "stock_lines": 96,
        "total_units": 14_820,
        "stock_value": "1842355.00",
        "low_stock_lines": 5,
        "out_of_stock_lines": 2,
    },
    "warehouses": {
        "count": 3,
        "items": [
            {
                "code": "WH-BLR",
                "name": "Bengaluru Main",
                "location": "Bengaluru",
                "product_lines": 42,
                "total_units": 8_140,
                "stock_value": "1012400.00",
                "low_stock_lines": 3,
                "out_of_stock_lines": 1,
            },
            {
                "code": "WH-PNQ",
                "name": "Pune Depot",
                "location": "Pune",
                "product_lines": 31,
                "total_units": 4_310,
                "stock_value": "561955.00",
                "low_stock_lines": 1,
                "out_of_stock_lines": 0,
            },
            {
                "code": "WH-MAA",
                "name": "Chennai Depot",
                "location": "Chennai",
                "product_lines": 23,
                "total_units": 2_370,
                "stock_value": "268000.00",
                "low_stock_lines": 1,
                "out_of_stock_lines": 1,
            },
        ],
    },
    "low_stock_items": {
        "count": 5,
        "shown": 3,
        "items": [
            {
                "sku": "SKU-1042",
                "name": "USB-C Charging Cable 2m",
                "warehouse_code": "WH-BLR",
                "quantity": 42,
                "reorder_point": 120,
            },
            {
                "sku": "SKU-2277",
                "name": "Laptop Stand Aluminium",
                "warehouse_code": "WH-BLR",
                "quantity": 18,
                "reorder_point": 60,
            },
            {
                "sku": "SKU-3310",
                "name": "Wireless Mouse Ergonomic",
                "warehouse_code": "WH-PNQ",
                "quantity": 25,
                "reorder_point": 50,
            },
        ],
    },
    "out_of_stock_items": {
        "count": 2,
        "shown": 2,
        "items": [
            {
                "sku": "SKU-4801",
                "name": "HDMI 2.1 Cable 3m",
                "warehouse_code": "WH-BLR",
                "quantity": 0,
                "reorder_point": 40,
            },
            {
                "sku": "SKU-5520",
                "name": "Mechanical Keyboard TKL",
                "warehouse_code": "WH-MAA",
                "quantity": 0,
                "reorder_point": 25,
            },
        ],
    },
    "sales_metrics": {
        "window_days": 30,
        "units_sold": 3_942,
        "products_sold": 37,
        "top_products": [
            {"sku": "SKU-1042", "name": "USB-C Charging Cable 2m", "units": 145},
            {"sku": "SKU-3310", "name": "Wireless Mouse Ergonomic", "units": 121},
            {"sku": "SKU-4801", "name": "HDMI 2.1 Cable 3m", "units": 98},
        ],
    },
    "suppliers": {
        "supplier_count": 6,
        "products_with_supplier": 41,
        "supplier_product_links": 53,
    },
    "scope": "all warehouses",
}

QUERIES: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "DEMAND_FORECAST",
        "How much of this product should we expect to sell over the horizon?",
        DEMAND_FORECAST_CONTEXT,
    ),
    (
        "REORDER_RECOMMENDATION",
        "Should we restock this product now, and roughly how many units?",
        REORDER_CONTEXT,
    ),
    (
        "INVENTORY_SUMMARY",
        "Summarise the inventory position for the operations meeting.",
        INVENTORY_SUMMARY_CONTEXT,
    ),
)


def wrap(text: str, width: int = 76, indent: str = "  ") -> str:
    """Wrap the model's prose without collapsing its own line breaks."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        current = ""
        for word in paragraph.split():
            if current and len(current) + len(word) + 1 > width:
                lines.append(f"{indent}{current}")
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(f"{indent}{current}")
    return "\n".join(lines)


def report(answer: LLMAnswer, query_type: str, elapsed: float) -> None:
    print(f"  Request ID : {answer.request_id}")
    print(f"  Query type : {query_type}")
    print(f"  Model      : {answer.model}")
    print(f"  Generated  : {elapsed:.1f}s")
    print()
    print("  Phi-3 answer:")
    print(wrap(answer.answer))


def explain_failure(error: LLMClientError, address: str) -> None:
    """Say what went wrong and what to do about it."""
    print(f"  FAILED: {error}")
    print()
    if isinstance(error, LLMTimeoutError):
        print("  The LLM node was reached but did not answer in time.")
        print("  Phi-3 Mini on a CPU is slow, and the first request also pays")
        print("  for loading the model. Try a longer budget:")
        print("      python scripts/demo_llm_queries.py --timeout 300")
        print("  If every request times out, check the node's own log for")
        print("  whether Ollama is responding at all.")
    elif isinstance(error, LLMUnavailableError):
        print(f"  Nothing answered at {address}. Check, in order:")
        print("    1. Ollama is running        ->  ollama list")
        print("    2. the model is pulled      ->  ollama pull phi3:mini")
        print("    3. the LLM node is running  ->  python -m llm.server")
        print("       (it should log: LLMService listening on 127.0.0.1:50052)")
        print("    4. the address matches      ->  --address host:port")
    elif isinstance(error, LLMRequestError):
        print("  The LLM node rejected the request. That is a bug in this")
        print("  script rather than a setup problem: the query type or the")
        print("  context shape no longer matches what the node accepts.")
    else:
        print("  The LLM node failed to produce an answer. Its log will have")
        print("  the detail -- most often the model tag is not pulled:")
        print("      ollama pull phi3:mini")


def run(address: str, timeout: float, only: str | None, show_context: bool) -> int:
    queries = [item for item in QUERIES if only is None or item[0] == only]
    if not queries:
        print(f"No such query type: {only}")
        print(f"Choose one of: {', '.join(item[0] for item in QUERIES)}")
        return 1

    print(RULE)
    print("Phase G -- LLM service demonstration")
    print(RULE)
    print(f"LLM service : {address}  (gRPC)")
    print(f"Timeout     : {timeout:g}s per query")
    print(f"Queries     : {len(queries)}")
    print()
    print("The context below is mock data in the shape the application server")
    print("builds from SQLite. This script holds no database session and")
    print("cannot change any inventory state.")

    answered = 0
    attempted = 0
    aborted = False

    with LLMClient(address, timeout=timeout) as client:
        for index, (query_type, question, context) in enumerate(queries, start=1):
            attempted += 1
            print()
            print(RULE)
            print(f"{index}. {query_type}")
            print(THIN_RULE)
            print(f"  Question   : {question}")
            if show_context:
                print("  Context    :")
                print(wrap(json.dumps(context, indent=2), width=100, indent="    "))
            print()

            # No request id is supplied, so the client generates one -- the
            # same path an application node takes for a caller that sends
            # none.
            request_id = f"DEMO-{query_type[:4]}-{index}"
            started = time.monotonic()
            try:
                answer = client.get_llm_answer(
                    request_id, query_type, question, context
                )
            except LLMClientError as error:
                explain_failure(error, address)
                # A dead node fails every remaining query the same way, so
                # stop rather than printing the same advice three times.
                if isinstance(error, LLMUnavailableError):
                    aborted = True
                    break
                continue

            answered += 1
            report(answer, query_type, time.monotonic() - started)

    print()
    print(RULE)
    print(f"Answered {answered} of {len(queries)} queries.")
    if aborted:
        print(f"Stopped after {attempted}: the LLM service is not reachable.")
    print(RULE)
    return 0 if answered == len(queries) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask the Phase G LLM node the three Milestone 1 questions."
    )
    parser.add_argument(
        "--address",
        default=DEFAULT_ADDRESS,
        help=f"LLM service address (default: {DEFAULT_ADDRESS})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"seconds to wait per query (default: {DEFAULT_TIMEOUT_SECONDS:g})",
    )
    parser.add_argument(
        "--only",
        choices=[item[0] for item in QUERIES],
        help="run a single query instead of all three",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print the context sent to the model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the client's own log lines as well as the printed report",
    )
    arguments = parser.parse_args()

    # The gRPC client logs a warning of its own for every failure. That is
    # right for a server log and wrong for a demonstration, where it would
    # print a raw status above the explanation written for the reader.
    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.ERROR,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    try:
        return run(
            arguments.address,
            arguments.timeout,
            arguments.only,
            arguments.show_context,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
