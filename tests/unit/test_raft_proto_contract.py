"""Contract tests for ``proto/raft.proto`` and its generated stubs.

These pin the wire contract Phase 6 (election) and Phase 7 (replication) both
build on, so a later change that would break the protocol fails here first.
They also check that the Raft package is independent of the existing client
and LLM contracts and resolves to this repository's ``raft/`` directory.

The gRPC round trip at the end runs a throwaway servicer on an ephemeral
loopback port. It needs no database, no Ollama and no other service.
"""

from __future__ import annotations

from concurrent import futures
from pathlib import Path

import grpc
import pytest

from proto import raft_pb2, raft_pb2_grpc

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_RPCS = ("RequestVote", "AppendEntries")


def _service() -> object:
    return raft_pb2.DESCRIPTOR.services_by_name["RaftService"]


def _fields(message: type) -> dict[str, int]:
    """Field name -> field number."""
    return {field.name: field.number for field in message.DESCRIPTOR.fields}


# --- stubs and service ------------------------------------------------------
def test_the_raft_stubs_import() -> None:
    assert raft_pb2.DESCRIPTOR.name == "proto/raft.proto"
    assert raft_pb2_grpc.RaftServiceStub is not None
    assert raft_pb2_grpc.RaftServiceServicer is not None


def test_the_package_is_its_own() -> None:
    assert raft_pb2.DESCRIPTOR.package == "inventory.raft.v1"
    assert _service().full_name == "inventory.raft.v1.RaftService"


@pytest.mark.parametrize("rpc", REQUIRED_RPCS)
def test_each_required_rpc_exists_on_service_stub_and_servicer(rpc: str) -> None:
    assert rpc in [method.name for method in _service().methods]
    assert hasattr(raft_pb2_grpc.RaftServiceServicer, rpc)
    # The stub binds its methods in __init__, so inspect a real instance.
    with grpc.insecure_channel("127.0.0.1:1") as channel:
        assert hasattr(raft_pb2_grpc.RaftServiceStub(channel), rpc)


def test_the_optional_observability_rpc_exists() -> None:
    assert "GetState" in [method.name for method in _service().methods]


def test_rpc_request_and_response_types() -> None:
    expected = {
        "RequestVote": ("RequestVoteRequest", "RequestVoteResponse"),
        "AppendEntries": ("AppendEntriesRequest", "AppendEntriesResponse"),
        "GetState": ("GetStateRequest", "GetStateResponse"),
    }
    actual = {
        method.name: (method.input_type.name, method.output_type.name)
        for method in _service().methods
    }
    assert actual == expected


def test_no_rpc_is_streaming() -> None:
    for method in _service().methods:
        assert not method.client_streaming, method.name
        assert not method.server_streaming, method.name


# --- RequestVote -----------------------------------------------------------
def test_request_vote_request_carries_candidate_term_and_log_position() -> None:
    assert set(_fields(raft_pb2.RequestVoteRequest)) == {
        "term",
        "candidate_id",
        "last_log_index",
        "last_log_term",
    }


def test_request_vote_response_carries_term_and_the_decision() -> None:
    assert set(_fields(raft_pb2.RequestVoteResponse)) == {"term", "vote_granted"}


# --- AppendEntries ---------------------------------------------------------
def test_append_entries_request_carries_every_standard_raft_field() -> None:
    assert set(_fields(raft_pb2.AppendEntriesRequest)) == {
        "term",
        "leader_id",
        "prev_log_index",
        "prev_log_term",
        "leader_commit",
        "entries",
    }


def test_append_entries_entries_is_a_repeated_log_entry() -> None:
    field = raft_pb2.AppendEntriesRequest.DESCRIPTOR.fields_by_name["entries"]
    assert field.is_repeated
    assert field.message_type.name == "LogEntry"


def test_log_entry_carries_index_term_and_an_opaque_command() -> None:
    assert set(_fields(raft_pb2.LogEntry)) == {"index", "term", "command"}
    command = raft_pb2.LogEntry.DESCRIPTOR.fields_by_name["command"]
    assert command.type == command.TYPE_BYTES


def test_append_entries_response_reports_term_success_and_matching_info() -> None:
    assert set(_fields(raft_pb2.AppendEntriesResponse)) == {
        "term",
        "success",
        "match_index",
        "conflict_index",
        "conflict_term",
    }


def test_an_empty_heartbeat_has_no_entries_and_round_trips() -> None:
    """Phase 6 sends exactly this: a term and a leader, nothing else."""
    heartbeat = raft_pb2.AppendEntriesRequest(term=3, leader_id=1)
    assert len(heartbeat.entries) == 0
    assert (heartbeat.prev_log_index, heartbeat.prev_log_term) == (0, 0)
    assert heartbeat.leader_commit == 0

    decoded = raft_pb2.AppendEntriesRequest.FromString(heartbeat.SerializeToString())
    assert decoded == heartbeat
    assert decoded.term == 3
    assert decoded.leader_id == 1


def test_a_full_append_request_round_trips_for_phase_seven() -> None:
    request = raft_pb2.AppendEntriesRequest(
        term=5,
        leader_id=2,
        prev_log_index=7,
        prev_log_term=4,
        leader_commit=6,
        entries=[raft_pb2.LogEntry(index=8, term=5, command=b"\x00opaque\xff")],
    )
    decoded = raft_pb2.AppendEntriesRequest.FromString(request.SerializeToString())
    assert decoded == request
    assert decoded.entries[0].command == b"\x00opaque\xff"


# --- GetState --------------------------------------------------------------
def test_get_state_reports_role_term_and_leader() -> None:
    assert set(_fields(raft_pb2.GetStateResponse)) == {
        "node_id",
        "role",
        "current_term",
        "leader_id",
        "voted_for",
        "commit_index",
        "last_log_index",
    }
    assert {value.name for value in raft_pb2.NodeRole.DESCRIPTOR.values} == {
        "NODE_ROLE_UNSPECIFIED",
        "NODE_ROLE_FOLLOWER",
        "NODE_ROLE_CANDIDATE",
        "NODE_ROLE_LEADER",
    }
    # Zero must be the "unset" value, never a real role.
    assert raft_pb2.NodeRole.NODE_ROLE_UNSPECIFIED == 0


# --- encoding rules --------------------------------------------------------
def test_every_field_number_fits_in_one_byte() -> None:
    for message in (
        raft_pb2.LogEntry,
        raft_pb2.RequestVoteRequest,
        raft_pb2.RequestVoteResponse,
        raft_pb2.AppendEntriesRequest,
        raft_pb2.AppendEntriesResponse,
        raft_pb2.GetStateResponse,
    ):
        for name, number in _fields(message).items():
            assert 1 <= number <= 15, f"{message.DESCRIPTOR.name}.{name} = {number}"


# --- independence from the other contracts ---------------------------------
def test_the_raft_contract_imports_no_other_proto() -> None:
    assert list(raft_pb2.DESCRIPTOR.dependencies) == []


def test_the_existing_contracts_are_untouched_by_the_raft_one() -> None:
    from proto import client_pb2, llm_pb2

    assert client_pb2.DESCRIPTOR.package == "inventory.client.v1"
    assert llm_pb2.DESCRIPTOR.package == "inventory.llm.v1"
    client_rpcs = {
        method.name
        for method in client_pb2.DESCRIPTOR.services_by_name["ClientService"].methods
    }
    assert client_rpcs == {
        "login",
        "logout",
        "post",
        "get",
        "processBusinessRequest",
    }
    llm_rpcs = {
        method.name
        for method in llm_pb2.DESCRIPTOR.services_by_name["LLMService"].methods
    }
    assert llm_rpcs == {"getLLMAnswer"}


# --- our package, not someone else's ---------------------------------------
def test_raft_resolves_to_this_repositorys_package() -> None:
    """A third-party ``raft`` on sys.path must not shadow ours."""
    import raft

    assert Path(raft.__file__).resolve() == (PROJECT_ROOT / "raft" / "__init__.py")


# --- a real round trip over gRPC -------------------------------------------
class _FakeRaft(raft_pb2_grpc.RaftServiceServicer):
    # Method names are fixed by the proto, hence PascalCase.
    def RequestVote(self, request, context):
        return raft_pb2.RequestVoteResponse(term=request.term, vote_granted=True)

    def AppendEntries(self, request, context):
        return raft_pb2.AppendEntriesResponse(
            term=request.term, success=True, match_index=request.prev_log_index
        )

    def GetState(self, request, context):
        return raft_pb2.GetStateResponse(
            node_id=1, role=raft_pb2.NODE_ROLE_LEADER, current_term=4, leader_id=1
        )


def test_the_stubs_work_over_a_real_grpc_channel() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    raft_pb2_grpc.add_RaftServiceServicer_to_server(_FakeRaft(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    assert port != 0
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = raft_pb2_grpc.RaftServiceStub(channel)

            vote = stub.RequestVote(
                raft_pb2.RequestVoteRequest(term=2, candidate_id=3), timeout=5
            )
            assert (vote.term, vote.vote_granted) == (2, True)

            beat = stub.AppendEntries(
                raft_pb2.AppendEntriesRequest(term=2, leader_id=3), timeout=5
            )
            assert (beat.term, beat.success) == (2, True)

            state = stub.GetState(raft_pb2.GetStateRequest(), timeout=5)
            assert state.role == raft_pb2.NODE_ROLE_LEADER
            assert state.current_term == 4
    finally:
        server.stop(None).wait()
