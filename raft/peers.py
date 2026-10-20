"""Outbound gRPC to the other Raft members.

:class:`GrpcPeerTransport` is the production implementation of
:class:`raft.node.PeerTransport`: the node asks it to call a peer and gets the
peer's reply, or ``None`` if there was none. It follows the shape of
``app/api/llm_client.LLMClient`` (a lazily opened channel, a deadline on every
call, an explicit ``close()``) without importing anything from the LLM code.

The contract with the node is deliberately forgiving: **no ordinary network
failure raises.** A peer that is down, slow or unreachable simply yields
``None``, which Raft treats as a message that was lost, something it tolerates
by design. The first failure to a peer is logged once as a warning, repeats
stay at debug level, and the recovery is logged once, so a crashed peer does
not flood the log at one line per heartbeat.

Every call carries a deadline (the node passes half the minimum election
timeout), so a hung peer can occupy a worker for a bounded time only.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

import grpc

from proto import raft_pb2, raft_pb2_grpc
from raft.config import Member

__all__ = ["GrpcPeerTransport", "PeerClient", "fetch_state"]

_logger = logging.getLogger("raft.peers")

#: gRPC's default reconnect backoff grows towards two minutes. After a peer
#: restarts, that would leave the rest of the cluster ignoring it for far longer
#: than an election takes, and a restarted follower that never hears a heartbeat
#: times out and starts a needless election. The backoff is therefore capped at
#: a quarter of a second, a few heartbeat intervals, which costs a dead peer
#: nothing more than four cheap local connection attempts per second.
CHANNEL_OPTIONS: tuple[tuple[str, int], ...] = (
    ("grpc.initial_reconnect_backoff_ms", 50),
    ("grpc.min_reconnect_backoff_ms", 50),
    ("grpc.max_reconnect_backoff_ms", 250),
)


class PeerClient:
    """A channel and stub for one peer, opened on first use.

    Thread-safe: the node calls it from several worker threads at once, and
    ``close()`` may arrive while calls are in flight (those return ``None``).
    """

    def __init__(self, peer_id: int, address: str) -> None:
        self._peer_id = peer_id
        self._address = address
        self._lock = threading.Lock()
        self._channel: grpc.Channel | None = None
        self._stub: raft_pb2_grpc.RaftServiceStub | None = None
        self._closed = False
        self._reachable: bool | None = None  # None until the first call

    @property
    def peer_id(self) -> int:
        return self._peer_id

    @property
    def address(self) -> str:
        return self._address

    @property
    def closed(self) -> bool:
        return self._closed

    def request_vote(
        self, request: raft_pb2.RequestVoteRequest, timeout: float
    ) -> raft_pb2.RequestVoteResponse | None:
        stub = self._get_stub()
        if stub is None:
            return None
        return self._call(lambda: stub.RequestVote(request, timeout=timeout))

    def append_entries(
        self, request: raft_pb2.AppendEntriesRequest, timeout: float
    ) -> raft_pb2.AppendEntriesResponse | None:
        stub = self._get_stub()
        if stub is None:
            return None
        return self._call(lambda: stub.AppendEntries(request, timeout=timeout))

    def close(self) -> None:
        """Close the channel. Idempotent; later calls return ``None``."""
        with self._lock:
            self._closed = True
            channel, self._channel, self._stub = self._channel, None, None
        if channel is not None:
            channel.close()

    # -- internals ----------------------------------------------------------
    def _get_stub(self) -> raft_pb2_grpc.RaftServiceStub | None:
        with self._lock:
            if self._closed:
                return None
            if self._stub is None:
                self._channel = grpc.insecure_channel(
                    self._address, options=CHANNEL_OPTIONS
                )
                self._stub = raft_pb2_grpc.RaftServiceStub(self._channel)
            return self._stub

    def _call(self, invoke):
        try:
            response = invoke()
        except grpc.RpcError as exc:
            self._note_failure(exc)
            return None
        except ValueError:
            # gRPC raises ValueError for a call on a channel closed mid-flight.
            return None
        self._note_success()
        return response

    def _note_failure(self, exc: grpc.RpcError) -> None:
        code = exc.code() if hasattr(exc, "code") else None
        with self._lock:
            first = self._reachable is not False and not self._closed
            self._reachable = False
        if first:
            _logger.warning(
                "peer %s (%s) is not answering: %s", self._peer_id, self._address, code
            )
        else:
            _logger.debug("peer %s still not answering: %s", self._peer_id, code)

    def _note_success(self) -> None:
        with self._lock:
            recovered = self._reachable is False
            self._reachable = True
        if recovered:
            _logger.info("peer %s (%s) is answering again", self._peer_id, self._address)


class GrpcPeerTransport:
    """The node's view of its peers: ``request_vote`` and ``append_entries``
    by peer id, plus ``close()`` to release every channel."""

    def __init__(self, peers: Iterable[Member]) -> None:
        self._clients = {m.node_id: PeerClient(m.node_id, m.address) for m in peers}

    @property
    def clients(self) -> dict[int, PeerClient]:
        return dict(self._clients)

    def request_vote(
        self, peer_id: int, request: raft_pb2.RequestVoteRequest, timeout: float
    ) -> raft_pb2.RequestVoteResponse | None:
        client = self._clients.get(peer_id)
        return client.request_vote(request, timeout) if client else None

    def append_entries(
        self, peer_id: int, request: raft_pb2.AppendEntriesRequest, timeout: float
    ) -> raft_pb2.AppendEntriesResponse | None:
        client = self._clients.get(peer_id)
        return client.append_entries(request, timeout) if client else None

    def close(self) -> None:
        for client in self._clients.values():
            client.close()


def fetch_state(address: str, timeout: float = 1.0) -> raft_pb2.GetStateResponse | None:
    """Ask one node what it currently believes, or ``None`` if it cannot be
    reached within ``timeout``. Read-only: ``GetState`` changes nothing on the
    node. Used by ``scripts/raft_status.py`` and by the tests."""
    with grpc.insecure_channel(address, options=CHANNEL_OPTIONS) as channel:
        stub = raft_pb2_grpc.RaftServiceStub(channel)
        try:
            return stub.GetState(raft_pb2.GetStateRequest(), timeout=timeout)
        except grpc.RpcError:
            return None
