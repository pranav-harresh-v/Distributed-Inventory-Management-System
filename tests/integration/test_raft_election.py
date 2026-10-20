"""Raft leader election over real gRPC, three nodes in one process.

Each node is a full :class:`raft.server.RaftMember`: its own state file under
``tmp_path``, its own gRPC server on its own port, its own peer channels and its
own ticker thread. They talk to each other only through gRPC, exactly as three
processes would; the process-level tests (``test_raft_cluster_processes.py``)
then repeat the important scenarios with real operating-system processes.

Nothing here touches ``data/``: every state file is a temporary one. Waiting is
done by polling with a deadline (``wait_until``), never by one fixed sleep, so a
test returns the moment its condition holds and only a genuine failure costs the
full timeout.

Timings are short on purpose (50 ms heartbeat, 300-600 ms election timeout) and
still satisfy the configuration's "timeout at least 5x heartbeat" rule.
"""

from __future__ import annotations

import socket
import threading
import time
from collections import defaultdict
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from proto import raft_pb2, raft_pb2_grpc
from raft.config import Member, RaftConfig
from raft.peers import GrpcPeerTransport, PeerClient, fetch_state
from raft.server import RaftMember
from raft.storage import RaftStateStore
from scripts import raft_status
from tests.integration.raft_support import free_ports, wait_until

HEARTBEAT_MS, ELECTION_MIN_MS, ELECTION_MAX_MS = 50, 300, 600


class GrpcCluster:
    """Three (or more) Raft members that find each other over loopback gRPC."""

    def __init__(self, tmp_path: Path, size: int = 3) -> None:
        self.tmp_path = tmp_path
        ports = free_ports(size)
        self.members = tuple(Member(i + 1, "127.0.0.1", ports[i]) for i in range(size))
        self.running: dict[int, RaftMember] = {}
        self.ever_started: list[RaftMember] = []
        self.leaders_by_term: dict[int, set[int]] = defaultdict(set)
        self._lock = threading.Lock()

    # -- construction -------------------------------------------------------
    def config(self, node_id: int) -> RaftConfig:
        me = next(m for m in self.members if m.node_id == node_id)
        return RaftConfig(
            node_id=node_id,
            members=self.members,
            host=me.host,
            port=me.port,
            state_path=self.state_path(node_id),
            heartbeat_ms=HEARTBEAT_MS,
            election_timeout_min_ms=ELECTION_MIN_MS,
            election_timeout_max_ms=ELECTION_MAX_MS,
            reserved={},
        )

    def state_path(self, node_id: int) -> Path:
        return self.tmp_path / f"raft_node{node_id}.db"

    def address(self, node_id: int) -> str:
        return next(m.address for m in self.members if m.node_id == node_id)

    def peers_argument(self) -> str:
        return ",".join(f"{m.node_id}={m.address}" for m in self.members)

    def _record(self, status) -> None:
        if status.role.value == "leader":
            with self._lock:
                self.leaders_by_term[status.current_term].add(status.node_id)

    def start(self, node_id: int) -> RaftMember:
        member = RaftMember(self.config(node_id), listener=self._record)
        member.start()
        self.running[node_id] = member
        self.ever_started.append(member)
        return member

    def start_all(self) -> None:
        for m in self.members:
            self.start(m.node_id)

    def stop(self, node_id: int, grace: float = 0.5) -> None:
        member = self.running.pop(node_id)
        member.stop(grace)

    def stop_all(self) -> None:
        for node_id in list(self.running):
            self.stop(node_id)

    # -- observation, always through the public gRPC GetState ---------------
    def states(self) -> dict[int, raft_pb2.GetStateResponse]:
        found = {}
        for node_id in self.running:
            state = fetch_state(self.address(node_id), timeout=1.0)
            if state is not None:
                found[node_id] = state
        return found

    def agreed(self) -> tuple[int, int] | None:
        """``(leader_id, term)`` once every running node answers, exactly one is
        leader, and all of them agree on its identity and term; else ``None``."""
        states = self.states()
        if len(states) != len(self.running) or not states:
            return None
        leaders = [s for s in states.values() if s.role == raft_pb2.NODE_ROLE_LEADER]
        if len(leaders) != 1:
            return None
        leader = leaders[0]
        agree = all(
            s.current_term == leader.current_term and s.leader_id == leader.node_id
            for s in states.values()
        )
        return (leader.node_id, leader.current_term) if agree else None

    def assert_election_safety(self) -> None:
        with self._lock:
            snapshot = {t: set(ids) for t, ids in self.leaders_by_term.items()}
        for term, ids in snapshot.items():
            assert len(ids) <= 1, f"term {term} had several leaders: {sorted(ids)}"


@pytest.fixture
def cluster(tmp_path: Path):
    c = GrpcCluster(tmp_path)
    try:
        yield c
    finally:
        c.stop_all()  # always: ports, threads and channels must not leak


# ---------------------------------------------------------------------------
# 1-3. Electing a leader and keeping it
# ---------------------------------------------------------------------------


def test_three_nodes_elect_one_leader_over_grpc(cluster: GrpcCluster) -> None:
    cluster.start_all()

    _, term = wait_until(cluster.agreed, message="one agreed leader")

    states = cluster.states()
    assert sorted(states) == [1, 2, 3]
    assert [s.node_id for s in states.values()] == [1, 2, 3]
    assert term >= 1
    roles = sorted(s.role for s in states.values())
    assert roles == [
        raft_pb2.NODE_ROLE_FOLLOWER,
        raft_pb2.NODE_ROLE_FOLLOWER,
        raft_pb2.NODE_ROLE_LEADER,
    ]
    cluster.assert_election_safety()


def test_each_term_has_at_most_one_observed_leader(cluster: GrpcCluster) -> None:
    cluster.start_all()
    wait_until(cluster.agreed, message="one agreed leader")

    time.sleep(1.5)  # let the cluster run: any election churn would show up here

    assert cluster.leaders_by_term, "no leader was ever recorded"
    cluster.assert_election_safety()


def test_a_stable_leader_keeps_leading_through_heartbeats(cluster: GrpcCluster) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="one agreed leader")

    deadline = time.monotonic() + 2.5  # four-plus maximum election timeouts
    while time.monotonic() < deadline:
        assert cluster.agreed() == (leader_id, term)  # no new term, no new leader
        time.sleep(0.1)

    cluster.assert_election_safety()


def test_get_state_reports_node_term_role_and_leader(cluster: GrpcCluster) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="one agreed leader")

    for node_id, state in cluster.states().items():
        assert state.node_id == node_id
        assert state.current_term == term
        assert state.leader_id == leader_id
        assert state.commit_index == 0 and state.last_log_index == 0
    leader_state = cluster.states()[leader_id]
    assert leader_state.role == raft_pb2.NODE_ROLE_LEADER
    assert leader_state.voted_for == leader_id  # a winner voted for itself


# ---------------------------------------------------------------------------
# 4-6. Failure, restart and higher terms
# ---------------------------------------------------------------------------


def test_a_new_leader_is_elected_after_the_leader_shuts_down(
    cluster: GrpcCluster,
) -> None:
    cluster.start_all()
    old_leader, old_term = wait_until(cluster.agreed, message="first leader")

    cluster.stop(old_leader)

    new_leader, new_term = wait_until(cluster.agreed, message="a new leader")
    assert new_leader != old_leader
    assert new_term > old_term
    cluster.assert_election_safety()


def test_a_restarted_follower_rejoins_without_disturbing_the_leader(
    cluster: GrpcCluster,
) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="first leader")
    follower = next(i for i in cluster.running if i != leader_id)
    persisted = RaftStateStore(cluster.state_path(follower), follower).load()
    assert persisted.current_term == term

    cluster.stop(follower)
    time.sleep(0.4)  # the cluster runs on without it
    restarted = cluster.start(follower)

    assert restarted.node.status().current_term == term  # state came back from disk
    assert wait_until(cluster.agreed, message="the full cluster to agree again") == (
        leader_id,
        term,
    )
    cluster.assert_election_safety()


def test_a_higher_term_deposes_the_leader_and_is_persisted(cluster: GrpcCluster) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="first leader")
    candidate = next(i for i in cluster.running if i != leader_id)

    with grpc.insecure_channel(cluster.address(leader_id)) as channel:
        stub = raft_pb2_grpc.RaftServiceStub(channel)
        reply = stub.RequestVote(
            raft_pb2.RequestVoteRequest(term=term + 10, candidate_id=candidate),
            timeout=2,
        )
    assert reply.vote_granted is True
    assert reply.term == term + 10

    # The leader gave way at once and wrote the new term and its vote to disk.
    deposed = RaftStateStore(cluster.state_path(leader_id), leader_id).load()
    assert deposed.current_term >= term + 10

    _, new_term = wait_until(cluster.agreed, message="a leader in a newer term")
    assert new_term >= term + 10
    cluster.assert_election_safety()


def test_a_stale_term_cannot_depose_the_leader(cluster: GrpcCluster) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="first leader")
    challenger = next(i for i in cluster.running if i != leader_id)

    with grpc.insecure_channel(cluster.address(leader_id)) as channel:
        stub = raft_pb2_grpc.RaftServiceStub(channel)
        reply = stub.RequestVote(
            raft_pb2.RequestVoteRequest(term=term - 1, candidate_id=challenger),
            timeout=2,
        )
        beat = stub.AppendEntries(
            raft_pb2.AppendEntriesRequest(term=term - 1, leader_id=challenger),
            timeout=2,
        )

    assert reply.vote_granted is False and reply.term == term
    assert beat.success is False and beat.term == term
    assert cluster.agreed() == (leader_id, term)


# ---------------------------------------------------------------------------
# 7. RPC timeouts and unreachable peers
# ---------------------------------------------------------------------------


class _SlowRaft(raft_pb2_grpc.RaftServiceServicer):
    """Answers, but only after ``delay`` seconds."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def RequestVote(self, request, context):
        time.sleep(self.delay)
        return raft_pb2.RequestVoteResponse(term=request.term, vote_granted=True)

    def AppendEntries(self, request, context):
        time.sleep(self.delay)
        return raft_pb2.AppendEntriesResponse(term=request.term, success=True)


@pytest.fixture
def slow_peer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    raft_pb2_grpc.add_RaftServiceServicer_to_server(_SlowRaft(delay=1.5), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield port
    finally:
        server.stop(None).wait()


def test_a_peer_that_answers_too_slowly_yields_no_reply_within_the_deadline(
    slow_peer: int,
) -> None:
    client = PeerClient(2, f"127.0.0.1:{slow_peer}")
    try:
        started = time.monotonic()
        vote = client.request_vote(raft_pb2.RequestVoteRequest(term=1, candidate_id=1), 0.15)
        beat = client.append_entries(raft_pb2.AppendEntriesRequest(term=1, leader_id=1), 0.15)
        elapsed = time.monotonic() - started
    finally:
        client.close()

    assert vote is None and beat is None
    assert elapsed < 1.0  # two bounded waits, nowhere near the peer's 1.5 s delay


def test_an_unreachable_peer_yields_no_reply_quickly_and_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dead_port = free_ports(1)[0]  # nothing is listening there
    client = PeerClient(3, f"127.0.0.1:{dead_port}")
    caplog.set_level("DEBUG", logger="raft.peers")
    try:
        started = time.monotonic()
        replies = [
            client.request_vote(raft_pb2.RequestVoteRequest(term=1, candidate_id=1), 0.5)
            for _ in range(3)
        ]
        elapsed = time.monotonic() - started
    finally:
        client.close()

    assert replies == [None, None, None]
    assert elapsed < 3.0
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # one warning for the outage, not one per heartbeat


def test_a_closed_peer_client_returns_none_and_never_raises() -> None:
    client = PeerClient(2, "127.0.0.1:1")
    client.close()
    client.close()  # idempotent

    assert client.closed
    assert client.request_vote(raft_pb2.RequestVoteRequest(term=1, candidate_id=1), 0.2) is None
    assert client.append_entries(raft_pb2.AppendEntriesRequest(term=1, leader_id=1), 0.2) is None


def test_the_transport_ignores_unknown_peers_and_closes_every_channel() -> None:
    transport = GrpcPeerTransport([Member(2, "127.0.0.1", 6002), Member(3, "127.0.0.1", 6003)])

    assert transport.request_vote(9, raft_pb2.RequestVoteRequest(term=1, candidate_id=1), 0.1) is None
    transport.close()
    assert all(client.closed for client in transport.clients.values())


def test_a_cluster_with_one_node_down_still_elects_a_leader(cluster: GrpcCluster) -> None:
    """Two of three is a majority; the dead peer costs only bounded timeouts."""
    cluster.start(1)
    cluster.start(2)  # node 3 never starts: its port refuses connections

    leader_id, _ = wait_until(cluster.agreed, message="a leader with node 3 absent")

    assert leader_id in (1, 2)
    cluster.assert_election_safety()


def test_a_lone_node_never_becomes_leader(cluster: GrpcCluster) -> None:
    member = cluster.start(1)  # nobody to vote for it

    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline:
        assert member.node.status().role.value != "leader"
        time.sleep(0.05)

    assert member.node.status().current_term >= 2  # it kept trying, and failing
    assert not cluster.leaders_by_term


# ---------------------------------------------------------------------------
# 8. Cleanup
# ---------------------------------------------------------------------------


def _raft_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("raft")]


def test_stopping_a_cluster_releases_ports_threads_and_channels(tmp_path: Path) -> None:
    cluster = GrpcCluster(tmp_path)
    cluster.start_all()
    wait_until(cluster.agreed, message="one agreed leader")
    assert _raft_threads()  # ticker, RPC workers and gRPC workers exist while running

    cluster.stop_all()

    wait_until(lambda: not _raft_threads(), timeout=10, message="all raft threads to exit")
    for member in cluster.ever_started:
        assert all(c.closed for c in member.transport.clients.values())
    for m in cluster.members:  # every port can be bound again
        with socket.socket() as probe:
            probe.bind((m.host, m.port))


def test_a_second_member_cannot_share_a_port(cluster: GrpcCluster) -> None:
    cluster.start(1)

    with pytest.raises(RuntimeError, match="(?i)bind"):
        RaftMember(cluster.config(1))

    assert cluster.running[1].node.status().node_id == 1  # the first is unharmed


def test_stopping_a_member_twice_is_harmless(cluster: GrpcCluster) -> None:
    member = cluster.start(1)
    member.stop(0.2)
    member.stop(0.2)
    cluster.running.pop(1)


def test_a_server_binds_only_the_configured_loopback_address(cluster: GrpcCluster) -> None:
    member = cluster.start(1)

    assert member.config.host == "127.0.0.1"
    assert member.port == member.config.port
    assert fetch_state(cluster.address(1), timeout=1.0) is not None


# ---------------------------------------------------------------------------
# The status tool
# ---------------------------------------------------------------------------


def test_the_status_tool_reports_the_leader_and_an_unreachable_node(
    cluster: GrpcCluster, capsys: pytest.CaptureFixture[str]
) -> None:
    cluster.start_all()
    leader_id, term = wait_until(cluster.agreed, message="one agreed leader")
    cluster.stop(next(i for i in cluster.running if i != leader_id))
    wait_until(
        lambda: cluster.agreed() == (leader_id, term), message="the two survivors to agree"
    )

    code = raft_status.main(["--peers", cluster.peers_argument(), "--timeout", "1"])
    out = capsys.readouterr().out

    assert code == 0
    assert f"Leader: node {leader_id} (term {term})." in out
    assert "unreachable" in out
    assert out.count("reachable") >= 3  # two reachable rows, one unreachable, a summary
    assert "2 of 3 nodes reachable." in out


def test_the_status_tool_with_nothing_running_exits_non_zero(
    cluster: GrpcCluster, capsys: pytest.CaptureFixture[str]
) -> None:
    code = raft_status.main(["--peers", cluster.peers_argument(), "--timeout", "0.3"])

    assert code == 2
    assert "0 of 3 nodes reachable." in capsys.readouterr().out


def test_the_status_tool_rejects_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert raft_status.main(["--peers", "not-a-peer-list"]) == 2
    assert raft_status.main(["--peers", "1=127.0.0.1:6001", "--timeout", "0"]) == 2
    assert "Invalid --peers" in capsys.readouterr().err


def test_the_status_tool_flags_two_leaders_in_one_term() -> None:
    members = (Member(1, "127.0.0.1", 6001), Member(2, "127.0.0.1", 6002))
    states = {
        1: raft_pb2.GetStateResponse(node_id=1, role=raft_pb2.NODE_ROLE_LEADER, current_term=4, leader_id=1),
        2: raft_pb2.GetStateResponse(node_id=2, role=raft_pb2.NODE_ROLE_LEADER, current_term=4, leader_id=2),
    }
    assert "SAME term" in raft_status.render(members, states)

    states[2] = raft_pb2.GetStateResponse(
        node_id=2, role=raft_pb2.NODE_ROLE_LEADER, current_term=5, leader_id=2
    )
    verdict = raft_status.render(members, states)
    assert "SAME term" not in verdict
    assert "stale" in verdict
