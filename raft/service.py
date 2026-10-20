"""gRPC front door of a Raft node: ``inventory.raft.v1.RaftService``.

This layer only translates. A protobuf request goes straight to the matching
:class:`~raft.node.RaftNode` handler and the node's protobuf reply comes back
out; every election rule (terms, votes, log freshness, step-down) lives in the
node and is not repeated here. That is what lets the rules be unit-tested
without a socket, and the transport be tested without the rules.

It mirrors ``llm/service.py``: a guard turns an unexpected exception into a
logged ``INTERNAL`` status with an opaque message, so a bug never leaks a
traceback across the wire or takes the server thread down.
"""

from __future__ import annotations

import functools
import logging
from typing import Final, NoReturn

import grpc

from proto import raft_pb2, raft_pb2_grpc
from raft.node import RaftNode, Role

__all__ = ["RaftService"]

_logger = logging.getLogger("raft.service")

INTERNAL_ERROR_MESSAGE: Final[str] = "Internal error in the Raft node."

_ROLE_TO_PROTO: Final[dict[Role, int]] = {
    Role.FOLLOWER: raft_pb2.NODE_ROLE_FOLLOWER,
    Role.CANDIDATE: raft_pb2.NODE_ROLE_CANDIDATE,
    Role.LEADER: raft_pb2.NODE_ROLE_LEADER,
}


def _abort(context: grpc.ServicerContext, code: grpc.StatusCode, message: str) -> NoReturn:
    context.abort(code, message)
    raise RuntimeError("unreachable: ServicerContext.abort must raise")


def _guarded(method):
    """Log an unexpected failure and report it as an opaque INTERNAL status."""

    @functools.wraps(method)
    def wrapper(self, request, context):
        try:
            return method(self, request, context)
        except Exception:
            _logger.exception("Raft RPC %s failed unexpectedly", method.__name__)
            _abort(context, grpc.StatusCode.INTERNAL, INTERNAL_ERROR_MESSAGE)

    return wrapper


class RaftService(raft_pb2_grpc.RaftServiceServicer):
    """Serves one :class:`~raft.node.RaftNode` to its peers and to observers."""

    def __init__(self, node: RaftNode) -> None:
        self._node = node

    # Method names are fixed by proto/raft.proto; gRPC dispatches on them.
    @_guarded
    def RequestVote(self, request, context):
        return self._node.handle_request_vote(request)

    @_guarded
    def AppendEntries(self, request, context):
        return self._node.handle_append_entries(request)

    @_guarded
    def GetState(self, request, context):
        """Read-only: a snapshot of what this node believes. Never changes it."""
        status = self._node.status()
        return raft_pb2.GetStateResponse(
            node_id=status.node_id,
            role=_ROLE_TO_PROTO[status.role],
            current_term=status.current_term,
            leader_id=status.leader_id or 0,  # 0 = no leader known
            voted_for=status.voted_for or 0,  # 0 = no vote cast
            commit_index=status.commit_index,
            last_log_index=status.last_log_index,
        )
