"""Application node gRPC server.

Run from the repository root::

    python -m app.server

Serves :class:`~app.api.client_service.ClientService` on 127.0.0.1:50051 by
default. Configuration comes from the environment so a second or third node
can run from the same checkout later without a code change; the database
path uses Phase D's own ``INVENTORY_DB_PATH`` variable.
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
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.api.llm_client import LLMClient
from app.storage.database import SessionLocal, get_database_path, init_db
from proto import client_pb2_grpc

_logger = logging.getLogger("app.server")

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 50051
DEFAULT_MAX_WORKERS: Final[int] = 8
DEFAULT_SHUTDOWN_GRACE_SECONDS: Final[float] = 5.0

HOST_ENV_VAR: Final[str] = "INVENTORY_GRPC_HOST"
PORT_ENV_VAR: Final[str] = "INVENTORY_GRPC_PORT"
MAX_WORKERS_ENV_VAR: Final[str] = "INVENTORY_GRPC_MAX_WORKERS"

__all__ = ["create_server", "main", "serve"]


def _env_int(name: str, default: int) -> int:
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
    session_factory: sessionmaker[DbSession] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    max_workers: int | None = None,
    llm_client: LLMClient | None = None,
) -> tuple[grpc.Server, int]:
    """Build a server bound to ``host:port``.

    Returns the server and the port actually bound, which matters when the
    caller passes port 0 to get an ephemeral port (the integration tests do
    exactly that, so they never collide with a running dev server).

    The worker count stays below the SQLAlchemy connection budget: Phase D's
    engine uses QueuePool with pool_size 5 and max_overflow 10, so 15
    connections are available to 8 workers and a request never waits on the
    pool.
    """
    bind_host = host if host is not None else os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    bind_port = port if port is not None else _env_int(PORT_ENV_VAR, DEFAULT_PORT)
    workers = (
        max_workers
        if max_workers is not None
        else _env_int(MAX_WORKERS_ENV_VAR, DEFAULT_MAX_WORKERS)
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    client_pb2_grpc.add_ClientServiceServicer_to_server(
        ClientService(session_factory or SessionLocal, llm_client=llm_client), server
    )
    bound_port = server.add_insecure_port(f"{bind_host}:{bind_port}")
    if bound_port == 0:
        raise RuntimeError(f"Could not bind gRPC server to {bind_host}:{bind_port}")
    return server, bound_port


def serve() -> int:
    """Start the server and block until a shutdown signal arrives."""
    logging.basicConfig(
        level=os.environ.get("INVENTORY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Idempotent, and lets a fresh checkout serve without a separate step.
    init_db()

    server, port = create_server()
    host = os.environ.get(HOST_ENV_VAR, DEFAULT_HOST)
    server.start()

    _logger.info("ClientService listening on %s:%s", host, port)
    _logger.info("Node database: %s", get_database_path())

    def _request_shutdown(signum: int, _frame: FrameType | None) -> None:
        _logger.info("Signal %s received, shutting down.", signal.Signals(signum).name)
        # Stop accepting new RPCs and let in-flight ones finish.
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
