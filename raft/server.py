"""Raft node process: one member of the cluster.

Run from the repository root, one process per member::

    $env:RAFT_NODE_ID = "1"; python -m raft.server

(or ``RAFT_NODE_ID=1 python -m raft.server`` in a POSIX shell). Configuration is
read from the ``RAFT_`` variables documented in :mod:`raft.config`; with only
``RAFT_NODE_ID`` set, the node joins the local three-member development cluster
on 127.0.0.1:50101-50103 and keeps its election state in ``data/raft_node<id>.db``.

This process is deliberately separate from the application server: it opens no
inventory database, imports nothing from ``app``, ``client`` or ``llm``, and
serves only ``inventory.raft.v1.RaftService`` on its own port with its own gRPC
thread pool, so a slow client request can never delay a heartbeat.

Threads in one node, for the record:

* the gRPC server's pool (this module) answers incoming Raft RPCs;
* one ticker thread (:class:`raft.node.RaftNode`) owns the timers: it starts
  elections when a follower or candidate times out, and sends heartbeats when
  the node leads. Those are mutually exclusive by role, so one thread serves
  both without ever making either wait for the other;
* a small outbound pool (also in the node) makes the actual RPC calls to peers,
  in parallel, so one dead peer cannot delay a vote request or heartbeat to the
  others.

Shutdown is graceful on SIGINT/SIGTERM (and Ctrl+Break on Windows): the node
stops first, then the server drains in-flight RPCs, then the peer channels
close. An abrupt kill skips all of that, which is safe by design: the term and
vote are already on disk before they are ever used.
"""

from __future__ import annotations

import logging
import os
import random
import signal
import sys
import threading
from collections.abc import Callable
from concurrent import futures
from types import FrameType
from typing import Final

import grpc

from proto import raft_pb2_grpc
from raft.config import ConfigError, RaftConfig, load_config
from raft.node import Clock, NodeStatus, RaftNode
from raft.peers import GrpcPeerTransport
from raft.service import RaftService
from raft.storage import RaftStateError, RaftStateStore

_logger = logging.getLogger("raft.server")

#: Raft RPCs are tiny and answered from memory, so a handful of workers is
#: plenty; this pool is private to the Raft server.
DEFAULT_MAX_WORKERS: Final[int] = 8
DEFAULT_SHUTDOWN_GRACE_SECONDS: Final[float] = 2.0
LOG_LEVEL_ENV_VAR: Final[str] = "INVENTORY_LOG_LEVEL"

__all__ = ["RaftMember", "create_server", "main", "serve"]


def create_server(
    node: RaftNode,
    *,
    host: str,
    port: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
    executor: futures.ThreadPoolExecutor | None = None,
) -> tuple[grpc.Server, int]:
    """Build (but do not start) a gRPC server for ``node`` bound to ``host:port``.

    Returns the server and the port actually bound. It binds only the address it
    is given, with its own thread pool (pass ``executor`` to own and shut it
    down yourself), and refuses to share a port: gRPC's default ``SO_REUSEPORT``
    would let a second process bind the same port silently on Linux, so it is
    switched off and a clash is an error.
    """
    pool = executor or futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=f"raft{node.status().node_id}-grpc"
    )
    server = grpc.server(pool, options=(("grpc.so_reuseport", 0),))
    raft_pb2_grpc.add_RaftServiceServicer_to_server(RaftService(node), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"Could not bind the Raft gRPC server to {host}:{port}")
    return server, bound_port


def log_status_change(status: NodeStatus) -> None:
    """Node listener: one log line per role or term change."""
    _logger.info(
        "node %s: now %s in term %s (leader=%s, voted_for=%s)",
        status.node_id,
        status.role.value,
        status.current_term,
        status.leader_id if status.leader_id is not None else "unknown",
        status.voted_for if status.voted_for is not None else "none",
    )


class RaftMember:
    """Everything one cluster member consists of, started and stopped together:
    state store, peer transport, election node and gRPC server.

    Tests build these in-process (with a custom ``clock``/``rng``/``listener``
    if they like); ``serve()`` builds exactly one from the environment.
    """

    def __init__(
        self,
        config: RaftConfig,
        *,
        clock: Clock | None = None,
        rng: random.Random | None = None,
        listener: Callable[[NodeStatus], None] | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self.config = config
        self._stopped = False
        self.transport = GrpcPeerTransport(config.peers)
        # Owned here so stop() can shut the pool down and leave no threads behind.
        self._executor = futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=f"raft{config.node_id}-grpc"
        )

        def on_change(status: NodeStatus) -> None:
            log_status_change(status)
            if listener is not None:
                listener(status)

        try:
            # Loading the state here means a corrupt file stops this member
            # before it binds a port or contacts anyone.
            self.node = RaftNode(
                config,
                RaftStateStore(config.state_path, config.node_id),
                self.transport,
                clock=clock,
                rng=rng,
                listener=on_change,
            )
            self.server, self.port = create_server(
                self.node, host=config.host, port=config.port, executor=self._executor
            )
        except BaseException:
            self.transport.close()
            self._executor.shutdown(wait=False)
            raise

    def start(self) -> None:
        """Begin answering RPCs, then begin the election timer."""
        self.server.start()
        self.node.start()

    def stop(self, grace: float = DEFAULT_SHUTDOWN_GRACE_SECONDS) -> None:
        """Graceful shutdown; safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        self.node.stop()  # no more elections or heartbeats; RPCs now refused
        self.server.stop(grace).wait()  # let in-flight RPCs finish
        self._executor.shutdown(wait=True)  # release the gRPC worker threads
        self.transport.close()  # release every peer channel


# --------------------------------------------------------------------------
# Process entry point
# --------------------------------------------------------------------------
def serve() -> int:
    """Run one Raft member until a shutdown signal arrives."""
    logging.basicConfig(
        level=os.environ.get(LOG_LEVEL_ENV_VAR, "INFO"),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        config = load_config()
        member = RaftMember(config)
    except (ConfigError, RaftStateError) as exc:
        # A bad config or a corrupt state file is a refusal to start, not a crash.
        _logger.error("Raft node cannot start: %s", exc)
        return 2

    persisted = member.node.status()
    _logger.info(
        "node %s: starting as follower with persisted term=%s voted_for=%s",
        config.node_id,
        persisted.current_term,
        persisted.voted_for if persisted.voted_for is not None else "none",
    )
    member.start()
    _logger.info(
        "node %s: RaftService listening on %s:%s (members: %s)",
        config.node_id,
        config.host,
        member.port,
        ", ".join(f"{m.node_id}={m.address}" for m in config.members),
    )
    _logger.info("node %s: election state file: %s", config.node_id, config.state_path)

    shutdown = threading.Event()

    def _request_shutdown(signum: int, _frame: FrameType | None) -> None:
        _logger.info("node %s: signal %s received, shutting down.",
                     config.node_id, signal.Signals(signum).name)
        shutdown.set()

    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):  # Ctrl+Break: how Windows asks politely
        signals.append(signal.SIGBREAK)
    for sig in signals:
        signal.signal(sig, _request_shutdown)

    # Wake regularly: on Windows a bare Event.wait() can delay signal handlers.
    while not shutdown.wait(0.2):
        pass

    member.stop()
    _logger.info("node %s: Raft node stopped.", config.node_id)
    return 0


def main() -> int:
    try:
        return serve()
    except KeyboardInterrupt:
        return 0
    except Exception:
        _logger.exception("Raft node failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
