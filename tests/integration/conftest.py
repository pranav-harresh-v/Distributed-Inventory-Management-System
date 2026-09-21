"""Fixtures that run a real gRPC server against a temporary database."""

from __future__ import annotations

from collections.abc import Iterator

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.server import create_server
from proto import client_pb2_grpc


@pytest.fixture
def grpc_server(session_factory: sessionmaker[DbSession]) -> Iterator[int]:
    """Start a server on an ephemeral port bound to the test database.

    Port 0 lets the OS choose, so the suite never collides with a dev server
    already listening on 50051.
    """
    server, port = create_server(
        session_factory, host="127.0.0.1", port=0, max_workers=4
    )
    server.start()
    try:
        yield port
    finally:
        server.stop(None).wait()


@pytest.fixture
def stub(grpc_server: int) -> Iterator[client_pb2_grpc.ClientServiceStub]:
    """A client stub connected to the test server over a real channel."""
    with grpc.insecure_channel(f"127.0.0.1:{grpc_server}") as channel:
        grpc.channel_ready_future(channel).result(timeout=10)
        yield client_pb2_grpc.ClientServiceStub(channel)
