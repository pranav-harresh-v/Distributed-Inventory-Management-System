"""LLM node gRPC server.

Run from the repository root::

    python -m llm.server

Serves :class:`~llm.service.LLMService` on 127.0.0.1:50052 by default -- a
different port from the application server's 50051, so both run side by side
on one machine during the demo. Configuration comes from the environment,
following ``app/server.py``.

This process is deliberately the boring half of the node: it binds a socket,
registers one servicer and waits. Everything worth testing lives in
:mod:`llm.service`, which needs no socket at all.

Starting does not require Ollama to be running. A node that refused to start
without a model would be harder to diagnose than one that starts and reports
UNAVAILABLE per request, and it would make the application server's own
degradation path untestable.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from concurrent import futures
from types import FrameType
from typing import Final

import grpc

from llm.ollama_client import OllamaClient
from llm.service import LLMService
from proto import llm_pb2_grpc

_logger = logging.getLogger("llm.server")

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 50052
DEFAULT_SHUTDOWN_GRACE_SECONDS: Final[float] = 5.0

#: Small on purpose, and not configurable from the environment. Generation is
#: CPU-bound against a single local model, so extra workers do not add
#: throughput -- they just queue inside Ollama while multiplying memory
#: pressure. Raise it through ``create_server`` if a deployment proves
#: otherwise.
DEFAULT_MAX_WORKERS: Final[int] = 4

HOST_ENV_VAR: Final[str] = "LLM_GRPC_HOST"
PORT_ENV_VAR: Final[str] = "LLM_GRPC_PORT"

#: Shared with the application server rather than duplicated, as documented
#: in .env.example.
LOG_LEVEL_ENV_VAR: Final[str] = "INVENTORY_LOG_LEVEL"

__all__ = ["create_server", "main", "serve"]


def _env_int(name: str, default: int) -> int:
    """Read an integer setting.

    A copy of the helper in ``app/server.py`` rather than an import of it:
    the LLM node must not depend on the application package, which is the
    rule that keeps it unable to reach a database. Five duplicated lines are
    the cheaper half of that trade.
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


def create_server(
    client: OllamaClient | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    max_workers: int | None = None,
    json_output: bool = False,
) -> tuple[grpc.Server, int]:
    """Build a server bound to ``host:port``.

    Returns the server and the port actually bound, which matters when the
    caller passes port 0 for an ephemeral one -- the integration tests do
    exactly that, so they never collide with a running LLM node.
    """
    bind_host = host if host is not None else os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    bind_port = port if port is not None else _env_int(PORT_ENV_VAR, DEFAULT_PORT)
    workers = max_workers if max_workers is not None else DEFAULT_MAX_WORKERS

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMService(client, json_output=json_output), server
    )
    bound_port = server.add_insecure_port(f"{bind_host}:{bind_port}")
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {bind_host}:{bind_port}")
    return server, bound_port


def serve() -> int:
    """Start the server and block until a shutdown signal arrives."""
    logging.basicConfig(
        level=os.environ.get(LOG_LEVEL_ENV_VAR, "INFO"),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    client = OllamaClient()
    server, port = create_server(client)
    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    server.start()

    _logger.info("LLMService listening on %s:%s", host, port)
    _logger.info("Ollama runtime: %s", client.host)
    _logger.info("Model: %s (timeout %gs)", client.model, client.timeout)

    def _request_shutdown(signum: int, _frame: FrameType | None) -> None:
        _logger.info("Signal %s received, shutting down.", signal.Signals(signum).name)
        # Stop accepting new RPCs and let in-flight generations finish.
        server.stop(DEFAULT_SHUTDOWN_GRACE_SECONDS)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_shutdown)

    server.wait_for_termination()
    _logger.info("Server stopped.")
    return 0


def main() -> int:
    try:
        return serve()
    except KeyboardInterrupt:
        return 0
    except Exception:
        _logger.exception("Server failed to start")
        return 1


if __name__ == "__main__":
    sys.exit(main())
