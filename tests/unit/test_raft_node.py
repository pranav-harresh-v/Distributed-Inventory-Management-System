"""Deterministic tests for the Raft election state machine (``raft.node``).

Almost nothing here sleeps. The node's four collaborators are injected:

* a ``FakeClock`` that only moves when a test advances it,
* a ``StubTransport`` that records calls and returns scripted replies,
* a seeded or fixed random source for election timeouts,
* a dispatcher that runs outbound work inline (or queues it, to deliver replies
  late and prove stale ones are ignored).

Persistence is real: each node gets a ``RaftStateStore`` under ``tmp_path``, so
"the vote reached disk before the reply" and "a restart cannot double-vote" are
checked against the file, not against a mock.

The last section simulates whole clusters in one process (shared fake clock,
in-memory network) to check the properties that matter across nodes: at most
one leader per term, re-election after a crash, no leader without a majority.
Those are *safety* (never violated) versus *liveness* (eventually achieved)
checks, and each test says which it is.

Only two tests use real threads and real time: the ticker must stop cleanly.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from proto import raft_pb2
from raft.config import Member, RaftConfig
from raft.node import NodeStatus, RaftNode, Role
from raft.storage import (
    RaftState,
    RaftStateCorruptError,
    RaftStateIOError,
    RaftStateStore,
)

# ---------------------------------------------------------------------------
# Test doubles and builders
# ---------------------------------------------------------------------------


class FakeClock:
    """Time that moves only when told to."""

    def __init__(self) -> None:
        self.t = 1000.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class MidRng(random.Random):
    """Always picks the middle of the range: a 500-1000 ms timeout is 750 ms."""

    def uniform(self, a: float, b: float) -> float:
        return (a + b) / 2


class StubTransport:
    """Records every call; replies come from the ``on_*`` hooks (default: none)."""

    def __init__(self) -> None:
        self.vote_calls: list[tuple[int, raft_pb2.RequestVoteRequest]] = []
        self.append_calls: list[tuple[int, raft_pb2.AppendEntriesRequest]] = []
        self.on_vote: Callable[[int, raft_pb2.RequestVoteRequest], object] = (
            lambda peer, request: None
        )
        self.on_append: Callable[[int, raft_pb2.AppendEntriesRequest], object] = (
            lambda peer, request: None
        )

    def request_vote(self, peer_id, request, timeout):
        self.vote_calls.append((peer_id, request))
        return self.on_vote(peer_id, request)

    def append_entries(self, peer_id, request, timeout):
        self.append_calls.append((peer_id, request))
        return self.on_append(peer_id, request)


def inline(task: Callable[[], None]) -> None:
    task()


class QueueDispatcher:
    """Holds outbound work until a test decides to run it."""

    def __init__(self) -> None:
        self.queue: list[Callable[[], None]] = []

    def __call__(self, task: Callable[[], None]) -> None:
        self.queue.append(task)

    def run_all(self) -> None:
        pending, self.queue = self.queue, []
        for task in pending:
            task()


class FakeLog:
    """A log with a fixed tail, for testing the vote's freshness rule."""

    def __init__(self, last_index: int, last_term: int) -> None:
        self._index, self._term = last_index, last_term

    def last_index(self) -> int:
        return self._index

    def last_term(self) -> int:
        return self._term

    def term_at(self, index: int) -> int | None:
        if index == 0:
            return 0
        return self._term if index == self._index else None


class FlakyStore:
    """Delegates to a real store but fails chosen ``save`` calls (1-based)."""

    def __init__(self, inner: RaftStateStore, fail_on: set[int]) -> None:
        self.inner, self.fail_on, self.saves = inner, set(fail_on), 0

    def load(self) -> RaftState:
        return self.inner.load()

    def save(self, term: int, voted_for: int | None) -> RaftState:
        self.saves += 1
        if self.saves in self.fail_on:
            raise RaftStateIOError("simulated disk failure")
        return self.inner.save(term, voted_for)


def make_config(tmp_path: Path, node_id: int = 1, size: int = 3, **fields: int) -> RaftConfig:
    members = tuple(Member(i, "127.0.0.1", 6000 + i) for i in range(1, size + 1))
    return RaftConfig(
        node_id=node_id,
        members=members,
        host="127.0.0.1",
        port=6000 + node_id,
        state_path=tmp_path / f"raft_node{node_id}.db",
        reserved={},
        **fields,
    )


@dataclass
class Harness:
    node: RaftNode
    store: RaftStateStore
    transport: StubTransport
    clock: FakeClock
    path: Path
    events: list[NodeStatus] = field(default_factory=list)

    def disk(self) -> RaftState:
        """What a fresh process would read from disk right now."""
        return RaftStateStore(self.path, self.node.status().node_id).load()


def build(
    tmp_path: Path,
    node_id: int = 1,
    size: int = 3,
    *,
    store: object | None = None,
    dispatcher: object = inline,
    log: object | None = None,
    rng: random.Random | None = None,
    transport: StubTransport | None = None,
    clock: FakeClock | None = None,
) -> Harness:
    config = make_config(tmp_path, node_id, size)
    real_store = RaftStateStore(config.state_path, node_id)
    transport = transport or StubTransport()
    clock = clock or FakeClock()
    events: list[NodeStatus] = []
    node = RaftNode(
        config,
        store or real_store,  # type: ignore[arg-type]
        transport,
        clock=clock,
        rng=rng or MidRng(),
        log=log,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        listener=events.append,
    )
    return Harness(node, real_store, transport, clock, config.state_path, events)


def vote_req(term: int, candidate: int, last_index: int = 0, last_term: int = 0):
    return raft_pb2.RequestVoteRequest(
        term=term, candidate_id=candidate, last_log_index=last_index, last_log_term=last_term
    )


def beat(term: int, leader: int, **fields: object):
    return raft_pb2.AppendEntriesRequest(term=term, leader_id=leader, **fields)


def vote_reply(term: int, granted: bool):
    return raft_pb2.RequestVoteResponse(term=term, vote_granted=granted)


def append_reply(term: int, success: bool = True):
    return raft_pb2.AppendEntriesResponse(term=term, success=success)


def expire(h: Harness) -> None:
    """Let the election timeout (750 ms with MidRng) lapse and fire the timer."""
    h.clock.advance(1.0)
    h.node.tick()


def make_leader(h: Harness) -> None:
    """Election in term 1, won with peer 2's vote."""
    expire(h)
    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))
    assert h.node.status().role is Role.LEADER


# ---------------------------------------------------------------------------
# 1. A new node
# ---------------------------------------------------------------------------


def test_a_new_node_is_a_follower_with_no_leader(tmp_path: Path) -> None:
    h = build(tmp_path)

    status = h.node.status()
    assert status.role is Role.FOLLOWER
    assert status.current_term == 0
    assert status.voted_for is None
    assert status.leader_id is None
    assert (status.commit_index, status.last_log_index) == (0, 0)


def test_constructing_a_node_sends_nothing_and_writes_nothing(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.transport.vote_calls == []
    assert h.transport.append_calls == []
    assert not h.path.exists()


def test_a_node_loads_its_persisted_term_and_vote(tmp_path: Path) -> None:
    RaftStateStore(tmp_path / "raft_node1.db", 1).save(7, 3)
    h = build(tmp_path)

    status = h.node.status()
    assert (status.current_term, status.voted_for) == (7, 3)
    assert status.role is Role.FOLLOWER


def test_a_corrupt_state_file_stops_the_node_from_being_built(tmp_path: Path) -> None:
    (tmp_path / "raft_node1.db").write_bytes(b"garbage" * 300)
    with pytest.raises(RaftStateCorruptError, match="Refusing to reset"):
        build(tmp_path)


# ---------------------------------------------------------------------------
# 2. Starting an election
# ---------------------------------------------------------------------------


def test_no_election_starts_before_the_timeout(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.clock.advance(0.74)  # timeout is 0.75
    h.node.tick()

    assert h.node.status().role is Role.FOLLOWER
    assert h.transport.vote_calls == []


def test_a_follower_starts_an_election_after_the_timeout(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.clock.advance(0.76)
    h.node.tick()

    assert h.node.status().role is Role.CANDIDATE


def test_an_election_increments_the_term_and_persists_it(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)

    assert h.node.status().current_term == 1
    assert h.disk() == RaftState(current_term=1, voted_for=1)


def test_a_candidate_votes_for_itself(tmp_path: Path) -> None:
    h = build(tmp_path, node_id=2)
    expire(h)

    assert h.node.status().voted_for == 2
    assert h.disk().voted_for == 2


def test_vote_requests_go_to_every_peer_with_the_right_fields(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)

    assert sorted(peer for peer, _ in h.transport.vote_calls) == [2, 3]
    for _, request in h.transport.vote_calls:
        assert (request.term, request.candidate_id) == (1, 1)
        assert (request.last_log_index, request.last_log_term) == (0, 0)


def test_the_term_and_self_vote_are_on_disk_before_any_request_leaves(
    tmp_path: Path,
) -> None:
    h = build(tmp_path)
    seen: list[RaftState] = []
    h.transport.on_vote = lambda peer, request: seen.append(h.disk())

    expire(h)

    assert len(seen) == 2
    assert all(state == RaftState(1, 1) for state in seen)


def test_a_candidate_whose_round_gets_no_majority_tries_again_later(
    tmp_path: Path,
) -> None:
    h = build(tmp_path)
    expire(h)
    assert h.node.status().current_term == 1

    h.clock.advance(0.5)
    h.node.tick()  # a fresh 750 ms timeout has not yet elapsed
    assert h.node.status().current_term == 1

    h.clock.advance(0.5)
    h.node.tick()
    assert h.node.status().current_term == 2
    assert h.node.status().role is Role.CANDIDATE


def test_a_one_member_cluster_elects_itself_at_once(tmp_path: Path) -> None:
    h = build(tmp_path, size=1)
    expire(h)

    assert h.node.status().role is Role.LEADER
    assert h.transport.vote_calls == []


def test_an_election_that_cannot_be_persisted_does_not_happen(tmp_path: Path) -> None:
    config_path = tmp_path / "raft_node1.db"
    flaky = FlakyStore(RaftStateStore(config_path, 1), fail_on={1})
    h = build(tmp_path, store=flaky)

    expire(h)

    status = h.node.status()
    assert (status.role, status.current_term, status.voted_for) == (Role.FOLLOWER, 0, None)
    assert h.transport.vote_calls == []

    h.clock.advance(0.5)
    h.node.tick()  # backed off by a fresh timeout
    assert h.node.status().current_term == 0
    h.clock.advance(0.5)
    h.node.tick()  # now the retry succeeds
    assert h.node.status().current_term == 1


# ---------------------------------------------------------------------------
# 3. Winning and losing
# ---------------------------------------------------------------------------


def test_a_majority_of_votes_elects_the_leader(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)

    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))

    status = h.node.status()
    assert status.role is Role.LEADER
    assert status.leader_id == 1
    assert status.current_term == 1


def test_the_votes_can_arrive_through_the_transport(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.transport.on_vote = lambda peer, request: vote_reply(request.term, peer == 3)

    expire(h)

    assert h.node.status().role is Role.LEADER


def test_a_new_leader_sends_heartbeats_at_once(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)

    assert sorted(peer for peer, _ in h.transport.append_calls) == [2, 3]
    for _, request in h.transport.append_calls:
        assert (request.term, request.leader_id) == (1, 1)
        assert len(request.entries) == 0
        assert (request.prev_log_index, request.prev_log_term) == (0, 0)
        assert request.leader_commit == 0


def test_a_minority_of_votes_does_not_elect_a_leader(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.transport.on_vote = lambda peer, request: (
        vote_reply(request.term, False) if peer == 2 else None
    )
    expire(h)

    assert h.node.status().role is Role.CANDIDATE


def test_five_nodes_need_three_votes(tmp_path: Path) -> None:
    h = build(tmp_path, size=5)
    expire(h)

    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))
    assert h.node.status().role is Role.CANDIDATE  # self + 1 = 2 of 5

    h.node.handle_request_vote_response(3, 1, vote_reply(1, True))
    assert h.node.status().role is Role.LEADER  # self + 2 = 3 of 5


def test_the_same_peers_vote_is_never_counted_twice(tmp_path: Path) -> None:
    h = build(tmp_path, size=5)
    expire(h)

    for _ in range(5):
        h.node.handle_request_vote_response(2, 1, vote_reply(1, True))

    assert h.node.status().role is Role.CANDIDATE


def test_a_refusal_is_not_a_vote(tmp_path: Path) -> None:
    h = build(tmp_path, size=5)
    expire(h)
    for peer in (2, 3, 4):
        h.node.handle_request_vote_response(peer, 1, vote_reply(1, False))
    assert h.node.status().role is Role.CANDIDATE


def test_replies_from_non_members_do_not_count(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    h.node.handle_request_vote_response(9, 1, vote_reply(1, True))
    h.node.handle_request_vote_response(1, 1, vote_reply(1, True))  # itself, again
    assert h.node.status().role is Role.CANDIDATE


def test_a_missing_reply_changes_nothing(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    before = h.node.status()
    h.node.handle_request_vote_response(2, 1, None)
    assert h.node.status() == before


def test_a_reply_with_a_higher_term_ends_the_candidacy(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.transport.on_vote = lambda peer, request: vote_reply(5, False)
    expire(h)

    status = h.node.status()
    assert (status.role, status.current_term, status.voted_for) == (Role.FOLLOWER, 5, None)
    assert h.disk() == RaftState(5, None)


# ---------------------------------------------------------------------------
# 4. Stale replies
# ---------------------------------------------------------------------------


def test_a_reply_from_an_older_round_is_ignored(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)  # term 1
    expire(h)  # term 2: the first round is over
    assert h.node.status().current_term == 2

    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))

    assert h.node.status().role is Role.CANDIDATE


def test_a_reply_labelled_with_the_wrong_term_is_ignored(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    expire(h)

    # Sent in term 1, but the responder (wrongly or late) says term 2.
    h.node.handle_request_vote_response(2, 1, vote_reply(2, True))
    assert h.node.status().role is Role.CANDIDATE


def test_a_late_vote_cannot_elect_a_node_that_has_stepped_down(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    h.node.handle_append_entries(beat(1, 3))  # someone else won term 1

    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))

    status = h.node.status()
    assert status.role is Role.FOLLOWER
    assert status.leader_id == 3


def test_old_round_requests_that_never_went_out_are_not_sent(tmp_path: Path) -> None:
    queue = QueueDispatcher()
    h = build(tmp_path, dispatcher=queue)
    expire(h)  # term 1; two asks queued, not run
    first_round = list(queue.queue)
    queue.queue.clear()
    expire(h)  # term 2

    for task in first_round:
        task()
    assert h.transport.vote_calls == []

    queue.run_all()
    assert sorted(request.term for _, request in h.transport.vote_calls) == [2, 2]


# ---------------------------------------------------------------------------
# 5. Granting and refusing votes
# ---------------------------------------------------------------------------


def test_a_stale_vote_request_is_refused(tmp_path: Path) -> None:
    RaftStateStore(tmp_path / "raft_node1.db", 1).save(5, None)
    h = build(tmp_path)

    response = h.node.handle_request_vote(vote_req(term=4, candidate=2))

    assert response.vote_granted is False
    assert response.term == 5
    assert h.disk() == RaftState(5, None)


def test_an_eligible_candidate_gets_the_vote(tmp_path: Path) -> None:
    h = build(tmp_path)

    response = h.node.handle_request_vote(vote_req(term=1, candidate=2))

    assert response.vote_granted is True
    assert response.term == 1
    status = h.node.status()
    assert (status.current_term, status.voted_for) == (1, 2)
    assert h.disk() == RaftState(1, 2)


def test_a_vote_is_on_disk_by_the_time_it_is_granted(tmp_path: Path) -> None:
    h = build(tmp_path)
    response = h.node.handle_request_vote(vote_req(term=3, candidate=3))
    assert response.vote_granted
    assert h.disk() == RaftState(3, 3)


def test_no_node_votes_for_two_candidates_in_one_term(tmp_path: Path) -> None:
    h = build(tmp_path)

    first = h.node.handle_request_vote(vote_req(term=1, candidate=2))
    second = h.node.handle_request_vote(vote_req(term=1, candidate=3))

    assert first.vote_granted is True
    assert second.vote_granted is False
    assert h.node.status().voted_for == 2
    assert h.disk() == RaftState(1, 2)


class UnguardedStore:
    """A store that accepts anything, so only the node's own rules are on trial.

    The real store independently refuses to change a vote within a term; this
    one removes that second line of defence to prove the node never asks.
    """

    def __init__(self) -> None:
        self.saved: list[tuple[int, int | None]] = []

    def load(self) -> RaftState:
        return RaftState()

    def save(self, term: int, voted_for: int | None) -> RaftState:
        self.saved.append((term, voted_for))
        return RaftState(term, voted_for)


def test_the_node_itself_refuses_a_second_vote_without_the_stores_help(
    tmp_path: Path,
) -> None:
    store = UnguardedStore()
    h = build(tmp_path, store=store)

    first = h.node.handle_request_vote(vote_req(1, 2))
    second = h.node.handle_request_vote(vote_req(1, 3))

    assert (first.vote_granted, second.vote_granted) == (True, False)
    assert store.saved == [(1, None), (1, 2)]  # it never even tried to record vote 3


def test_the_same_candidate_may_ask_again_in_the_same_term(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.node.handle_request_vote(vote_req(1, 2)).vote_granted
    assert h.node.handle_request_vote(vote_req(1, 2)).vote_granted
    assert h.disk() == RaftState(1, 2)


def test_a_newer_term_frees_the_vote_for_another_candidate(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.node.handle_request_vote(vote_req(1, 2)).vote_granted
    assert h.node.handle_request_vote(vote_req(2, 3)).vote_granted
    assert h.disk() == RaftState(2, 3)


def test_a_node_that_voted_for_itself_refuses_others_in_that_term(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)  # candidate in term 1, voted for itself

    response = h.node.handle_request_vote(vote_req(1, 2))

    assert response.vote_granted is False
    assert h.node.status().voted_for == 1


@pytest.mark.parametrize(
    ("last_term", "last_index", "granted"),
    [
        (1, 9, False),  # older last term loses, however long the log
        (2, 4, False),  # same term, shorter log
        (2, 5, True),  # identical position
        (2, 9, True),  # same term, longer log
        (3, 0, True),  # newer last term wins, however short
    ],
)
def test_the_candidates_log_must_be_at_least_as_up_to_date(
    tmp_path: Path, last_term: int, last_index: int, granted: bool
) -> None:
    h = build(tmp_path, log=FakeLog(last_index=5, last_term=2))

    response = h.node.handle_request_vote(vote_req(1, 2, last_index, last_term))

    assert response.vote_granted is granted
    status = h.node.status()
    assert status.current_term == 1  # the newer term is adopted either way
    assert status.voted_for == (2 if granted else None)


def test_with_empty_logs_the_empty_log_convention_is_enough(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.node.handle_request_vote(vote_req(1, 2, 0, 0)).vote_granted


def test_a_candidate_asks_with_its_own_log_position(tmp_path: Path) -> None:
    h = build(tmp_path, log=FakeLog(last_index=5, last_term=2))
    expire(h)
    for _, request in h.transport.vote_calls:
        assert (request.last_log_index, request.last_log_term) == (5, 2)


def test_granting_a_vote_restarts_the_election_timer(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.clock.advance(0.7)
    assert h.node.handle_request_vote(vote_req(1, 2)).vote_granted

    h.clock.advance(0.7)  # 1.4 s in: past the original deadline
    h.node.tick()

    assert h.node.status().role is Role.FOLLOWER


def test_a_refused_vote_does_not_restart_the_timer(tmp_path: Path) -> None:
    RaftStateStore(tmp_path / "raft_node1.db", 1).save(3, None)
    h = build(tmp_path)
    h.clock.advance(0.7)
    assert not h.node.handle_request_vote(vote_req(2, 2)).vote_granted  # stale term

    h.clock.advance(0.1)
    h.node.tick()  # 0.8 s: the original 0.75 s deadline has passed

    assert h.node.status().role is Role.CANDIDATE


def test_a_refused_vote_for_a_newer_term_does_not_postpone_our_own_election(
    tmp_path: Path,
) -> None:
    """A candidate with an out-of-date log must not be able to keep resetting
    everyone's timers, or the nodes that could actually win would never run."""
    h = build(tmp_path, log=FakeLog(last_index=5, last_term=2))
    h.clock.advance(0.7)

    refused = h.node.handle_request_vote(vote_req(1, 2, last_index=9, last_term=1))
    assert refused.vote_granted is False
    assert h.node.status().current_term == 1  # the newer term is still adopted

    h.clock.advance(0.1)  # 0.8 s from the start: the original deadline has passed
    h.node.tick()
    assert h.node.status().role is Role.CANDIDATE
    assert h.node.status().current_term == 2


@pytest.mark.parametrize("candidate", [1, 7, 0])
def test_requests_from_non_members_and_from_itself_are_ignored(
    tmp_path: Path, candidate: int
) -> None:
    h = build(tmp_path)

    response = h.node.handle_request_vote(vote_req(term=9, candidate=candidate))

    assert response.vote_granted is False
    assert h.node.status().current_term == 0  # their term is not adopted either
    assert not h.path.exists()


def test_a_vote_that_cannot_be_persisted_is_not_granted(tmp_path: Path) -> None:
    flaky = FlakyStore(RaftStateStore(tmp_path / "raft_node1.db", 1), fail_on={2})
    h = build(tmp_path, store=flaky)

    # Save 1 adopts term 1 (ok); save 2 would record the vote (fails).
    response = h.node.handle_request_vote(vote_req(1, 2))

    assert response.vote_granted is False
    assert h.node.status().voted_for is None
    assert h.disk() == RaftState(1, None)

    retry = h.node.handle_request_vote(vote_req(1, 2))  # save 3 succeeds
    assert retry.vote_granted is True
    assert h.disk() == RaftState(1, 2)


def test_a_term_that_cannot_be_persisted_is_not_adopted(tmp_path: Path) -> None:
    flaky = FlakyStore(RaftStateStore(tmp_path / "raft_node1.db", 1), fail_on={1})
    h = build(tmp_path, store=flaky)

    response = h.node.handle_request_vote(vote_req(4, 2))

    assert response.vote_granted is False
    assert response.term == 0
    assert h.node.status().current_term == 0


# ---------------------------------------------------------------------------
# 6. Heartbeats
# ---------------------------------------------------------------------------


def test_a_follower_accepts_a_current_heartbeat(tmp_path: Path) -> None:
    h = build(tmp_path)

    response = h.node.handle_append_entries(beat(term=1, leader=2))

    assert response.success is True
    assert response.term == 1
    assert response.match_index == 0
    status = h.node.status()
    assert (status.current_term, status.leader_id, status.role) == (1, 2, Role.FOLLOWER)
    assert h.disk() == RaftState(1, None)


def test_a_stale_heartbeat_is_refused_and_ignored(tmp_path: Path) -> None:
    RaftStateStore(tmp_path / "raft_node1.db", 1).save(3, None)
    h = build(tmp_path)
    h.clock.advance(0.5)

    response = h.node.handle_append_entries(beat(term=2, leader=2))

    assert response.success is False
    assert response.term == 3
    assert h.node.status().leader_id is None
    h.clock.advance(0.3)  # 0.8 s from the start: the stale beat bought no time
    h.node.tick()
    assert h.node.status().role is Role.CANDIDATE


def test_a_heartbeat_resets_the_election_timeout(tmp_path: Path) -> None:
    h = build(tmp_path)

    for _ in range(6):  # 3.6 s: far past one timeout
        h.clock.advance(0.6)
        h.node.tick()
        assert h.node.handle_append_entries(beat(1, 2)).success

    assert h.node.status().role is Role.FOLLOWER
    assert h.transport.vote_calls == []

    h.clock.advance(0.76)  # silence: now the leader is presumed dead
    h.node.tick()
    assert h.node.status().role is Role.CANDIDATE


def test_a_higher_term_heartbeat_is_adopted_and_clears_the_vote(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.node.handle_request_vote(vote_req(1, 3)).vote_granted

    assert h.node.handle_append_entries(beat(4, 2)).success

    status = h.node.status()
    assert (status.current_term, status.voted_for, status.leader_id) == (4, None, 2)
    assert h.disk() == RaftState(4, None)


def test_a_candidate_steps_down_for_a_leader_of_its_own_term(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    assert h.node.status().role is Role.CANDIDATE

    response = h.node.handle_append_entries(beat(term=1, leader=2))

    assert response.success is True
    status = h.node.status()
    assert (status.role, status.current_term, status.leader_id) == (Role.FOLLOWER, 1, 2)
    assert status.voted_for == 1  # its vote for itself in term 1 still stands
    assert h.disk() == RaftState(1, 1)


def test_a_candidate_steps_down_for_a_leader_of_a_newer_term(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)

    assert h.node.handle_append_entries(beat(term=4, leader=3)).success

    status = h.node.status()
    assert (status.role, status.current_term, status.voted_for) == (Role.FOLLOWER, 4, None)


def test_a_candidate_ignores_a_stale_leaders_heartbeat(tmp_path: Path) -> None:
    h = build(tmp_path)
    expire(h)
    expire(h)  # term 2

    response = h.node.handle_append_entries(beat(term=1, leader=2))

    assert response.success is False
    assert response.term == 2
    assert h.node.status().role is Role.CANDIDATE


@pytest.mark.parametrize("leader", [1, 7, 0])
def test_heartbeats_from_non_members_and_from_itself_are_ignored(
    tmp_path: Path, leader: int
) -> None:
    h = build(tmp_path)

    response = h.node.handle_append_entries(beat(term=9, leader=leader))

    assert response.success is False
    assert h.node.status().current_term == 0
    assert h.node.status().leader_id is None


def test_a_leader_refuses_a_rival_claiming_its_own_term(tmp_path: Path) -> None:
    """Cannot happen if elections are safe; if it does, do not give way silently."""
    h = build(tmp_path)
    make_leader(h)

    response = h.node.handle_append_entries(beat(term=1, leader=3))

    assert response.success is False
    assert h.node.status().role is Role.LEADER


def test_log_entries_are_refused_but_the_leader_is_still_recognised(
    tmp_path: Path,
) -> None:
    h = build(tmp_path)
    request = beat(
        term=1, leader=2, entries=[raft_pb2.LogEntry(index=1, term=1, command=b"x")]
    )

    response = h.node.handle_append_entries(request)

    assert response.success is False  # nothing was appended
    status = h.node.status()
    assert (status.leader_id, status.current_term) == (2, 1)
    assert (status.last_log_index, status.commit_index) == (0, 0)


def test_a_previous_entry_this_node_does_not_have_is_a_mismatch(tmp_path: Path) -> None:
    h = build(tmp_path)

    response = h.node.handle_append_entries(
        beat(term=1, leader=2, prev_log_index=3, prev_log_term=1)
    )

    assert response.success is False
    assert response.conflict_index == 1  # its log is empty: next wanted index is 1
    assert h.node.status().leader_id == 2


def test_a_leader_commit_index_is_not_acted_on(tmp_path: Path) -> None:
    h = build(tmp_path)
    assert h.node.handle_append_entries(beat(term=1, leader=2, leader_commit=5)).success
    assert h.node.status().commit_index == 0


# ---------------------------------------------------------------------------
# 7. The leader
# ---------------------------------------------------------------------------


def test_a_leader_sends_a_heartbeat_every_interval(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)
    h.transport.append_calls.clear()

    h.clock.advance(0.05)
    h.node.tick()
    assert h.transport.append_calls == []  # 100 ms has not passed

    h.clock.advance(0.06)
    h.node.tick()
    assert len(h.transport.append_calls) == 2

    h.clock.advance(0.1)
    h.node.tick()
    assert len(h.transport.append_calls) == 4


def test_a_leader_does_not_start_elections(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)

    for _ in range(30):  # 3 s of heartbeats, far past any election timeout
        h.clock.advance(0.1)
        h.node.tick()

    status = h.node.status()
    assert (status.role, status.current_term) == (Role.LEADER, 1)


def test_a_slow_peer_is_not_sent_a_second_heartbeat_while_one_is_pending(
    tmp_path: Path,
) -> None:
    queue = QueueDispatcher()
    h = build(tmp_path, dispatcher=queue)
    expire(h)
    queue.queue.clear()
    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))
    assert len(queue.queue) == 2  # first heartbeats, held back

    h.clock.advance(0.11)
    h.node.tick()
    assert len(queue.queue) == 2  # no pile-up behind the pending ones

    queue.run_all()
    h.clock.advance(0.11)
    h.node.tick()
    assert len(queue.queue) == 2  # the peers are free again


@pytest.mark.parametrize("how", ["vote_request", "heartbeat", "heartbeat_reply"])
def test_a_leader_steps_down_when_it_sees_a_higher_term(tmp_path: Path, how: str) -> None:
    h = build(tmp_path)
    make_leader(h)

    if how == "vote_request":
        h.node.handle_request_vote(vote_req(term=2, candidate=3))
        expected_vote = 3
    elif how == "heartbeat":
        h.node.handle_append_entries(beat(term=2, leader=3))
        expected_vote = None
    else:
        h.node.handle_append_entries_response(2, 1, append_reply(term=2, success=False))
        expected_vote = None

    status = h.node.status()
    assert status.role is Role.FOLLOWER
    assert status.current_term == 2
    assert status.voted_for == expected_vote
    assert h.disk() == RaftState(2, expected_vote)


def test_a_stepped_down_leader_stops_sending_heartbeats(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)
    h.node.handle_append_entries(beat(term=2, leader=3))
    h.transport.append_calls.clear()

    for _ in range(4):
        h.clock.advance(0.1)
        h.node.tick()

    assert h.transport.append_calls == []
    assert h.node.status().role is Role.FOLLOWER


def test_heartbeat_work_queued_before_a_step_down_is_not_sent(tmp_path: Path) -> None:
    queue = QueueDispatcher()
    h = build(tmp_path, dispatcher=queue)
    expire(h)
    queue.queue.clear()
    h.node.handle_request_vote_response(2, 1, vote_reply(1, True))
    queued = list(queue.queue)

    h.node.handle_append_entries(beat(term=2, leader=3))
    for task in queued:
        task()

    assert h.transport.append_calls == []


def test_a_heartbeat_reply_from_the_current_term_changes_nothing(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)
    before = h.node.status()
    h.node.handle_append_entries_response(2, 1, append_reply(term=1))
    h.node.handle_append_entries_response(3, 1, None)
    assert h.node.status() == before


def test_a_leader_claims_no_commit_progress(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)
    h.node.handle_append_entries_response(2, 1, append_reply(term=1, success=True))
    assert h.node.status().commit_index == 0


# ---------------------------------------------------------------------------
# 8. Restarts and persistence
# ---------------------------------------------------------------------------


def test_a_vote_survives_a_restart(tmp_path: Path) -> None:
    first = build(tmp_path)
    assert first.node.handle_request_vote(vote_req(1, 2)).vote_granted

    second = build(tmp_path)  # a new process, same file

    status = second.node.status()
    assert (status.current_term, status.voted_for) == (1, 2)


def test_a_restart_cannot_cause_a_second_vote_in_the_same_term(tmp_path: Path) -> None:
    first = build(tmp_path)
    assert first.node.handle_request_vote(vote_req(1, 2)).vote_granted

    second = build(tmp_path)
    assert second.node.handle_request_vote(vote_req(1, 3)).vote_granted is False
    assert second.node.handle_request_vote(vote_req(1, 2)).vote_granted is True
    assert second.disk() == RaftState(1, 2)


def test_a_restarted_candidate_remembers_it_voted_for_itself(tmp_path: Path) -> None:
    first = build(tmp_path)
    expire(first)

    second = build(tmp_path)

    assert second.node.status().voted_for == 1
    assert second.node.handle_request_vote(vote_req(1, 2)).vote_granted is False


def test_a_restarted_node_never_goes_back_to_an_earlier_term(tmp_path: Path) -> None:
    first = build(tmp_path)
    expire(first)
    expire(first)
    expire(first)

    second = build(tmp_path)
    assert second.node.status().current_term == 3
    assert second.node.handle_request_vote(vote_req(2, 2)).vote_granted is False
    assert second.node.handle_append_entries(beat(2, 2)).success is False
    expire(second)
    assert second.node.status().current_term == 4


def test_a_restarted_node_comes_back_as_a_follower(tmp_path: Path) -> None:
    first = build(tmp_path)
    make_leader(first)

    second = build(tmp_path)

    status = second.node.status()
    assert status.role is Role.FOLLOWER
    assert status.leader_id is None
    assert status.current_term == 1


# ---------------------------------------------------------------------------
# 9. Timeouts
# ---------------------------------------------------------------------------


def _measure_timeout(tmp_path: Path, seed: int) -> float:
    path = tmp_path / f"seed{seed}"
    path.mkdir()
    h = build(path, rng=random.Random(seed))
    started = h.clock.now()
    for _ in range(200):
        h.clock.advance(0.01)
        h.node.tick()
        if h.node.status().role is Role.CANDIDATE:
            return h.clock.now() - started
    raise AssertionError("no election within 2 s")


def test_election_timeouts_are_randomized_within_the_configured_range(
    tmp_path: Path,
) -> None:
    waits = [_measure_timeout(tmp_path, seed) for seed in range(1, 41)]

    assert all(0.499 <= wait <= 1.02 for wait in waits)  # 10 ms tick granularity
    assert len({round(wait, 2) for wait in waits}) >= 10  # genuinely spread out


def test_the_timeout_comes_from_the_injected_random_source(tmp_path: Path) -> None:
    class Always(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return 0.6  # ignores the configured 500-1000 ms range entirely

    # The node divides the draw by 1000, so the timeout is 0.0006 s: proof the
    # value came from the injected source, not from the configured range.
    h = build(tmp_path, rng=Always())
    h.clock.advance(0.002)
    h.node.tick()
    assert h.node.status().role is Role.CANDIDATE


def test_each_election_round_draws_a_fresh_timeout(tmp_path: Path) -> None:
    draws: list[float] = []

    class Recording(random.Random):
        def uniform(self, a: float, b: float) -> float:
            draws.append(super().uniform(a, b))
            return draws[-1]

    h = build(tmp_path, rng=Recording(7))
    before = len(draws)
    expire(h)
    expire(h)
    assert len(draws) > before + 1


# ---------------------------------------------------------------------------
# 10. Repeated rounds
# ---------------------------------------------------------------------------


def test_repeated_unanswered_elections_stay_safe(tmp_path: Path) -> None:
    h = build(tmp_path)
    terms: list[int] = []

    for _ in range(10):
        expire(h)
        status = h.node.status()
        assert status.role is Role.CANDIDATE
        assert status.voted_for == 1  # only ever its own vote
        assert h.disk() == RaftState(status.current_term, 1)
        terms.append(status.current_term)

    assert terms == list(range(1, 11))  # strictly increasing, no gaps or repeats
    assert all(e.role is not Role.LEADER for e in h.events)


def test_the_listener_sees_each_role_change(tmp_path: Path) -> None:
    h = build(tmp_path)
    make_leader(h)
    h.node.handle_append_entries(beat(2, 3))

    seen = [(e.role, e.current_term) for e in h.events]
    assert seen == [
        (Role.CANDIDATE, 1),
        (Role.LEADER, 1),
        (Role.FOLLOWER, 2),
    ]


def test_a_failing_listener_cannot_break_the_node(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    def broken(status: NodeStatus) -> None:
        raise RuntimeError("observer bug")

    clock = FakeClock()
    node = RaftNode(
        config,
        RaftStateStore(config.state_path, 1),
        StubTransport(),
        clock=clock,
        rng=MidRng(),
        dispatcher=inline,
        listener=broken,
    )
    clock.advance(1.0)
    node.tick()
    assert node.status().role is Role.CANDIDATE


# ---------------------------------------------------------------------------
# 11. Concurrency
# ---------------------------------------------------------------------------


def _other_thread_can_take_the_lock(node: RaftNode) -> bool:
    outcome: list[bool] = []

    def probe() -> None:
        got = node._lock.acquire(blocking=False)
        outcome.append(got)
        if got:
            node._lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return outcome[0]


def test_the_state_lock_is_free_while_vote_requests_are_on_the_network(
    tmp_path: Path,
) -> None:
    h = build(tmp_path)
    free: list[bool] = []
    h.transport.on_vote = lambda peer, request: free.append(
        _other_thread_can_take_the_lock(h.node)
    )

    expire(h)

    assert free == [True, True]


def test_the_state_lock_is_free_while_heartbeats_are_on_the_network(
    tmp_path: Path,
) -> None:
    h = build(tmp_path)
    free: list[bool] = []
    h.transport.on_append = lambda peer, request: free.append(
        _other_thread_can_take_the_lock(h.node)
    )

    make_leader(h)

    assert free == [True, True]


def test_concurrent_candidates_cannot_both_win_one_vote(tmp_path: Path) -> None:
    for trial in range(30):
        folder = tmp_path / f"trial{trial}"
        folder.mkdir()
        h = build(folder)
        barrier = threading.Barrier(2)
        results: dict[int, bool] = {}

        def ask(candidate: int, h: Harness = h, barrier: threading.Barrier = barrier,
                results: dict[int, bool] = results) -> None:
            barrier.wait()
            results[candidate] = h.node.handle_request_vote(vote_req(1, candidate)).vote_granted

        threads = [threading.Thread(target=ask, args=(c,)) for c in (2, 3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert sorted(results.values()) == [False, True], results
        winner = next(c for c, granted in results.items() if granted)
        assert h.disk() == RaftState(1, winner)


def test_concurrent_handlers_and_timers_leave_consistent_state(tmp_path: Path) -> None:
    h = build(tmp_path)
    errors: list[BaseException] = []
    stop = threading.Event()

    def hammer(work: Callable[[], object]) -> None:
        try:
            while not stop.is_set():
                work()
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    terms = iter(range(1, 10_000))
    workers = [
        threading.Thread(target=hammer, args=(lambda: h.node.handle_request_vote(vote_req(next(terms), 2)),)),
        threading.Thread(target=hammer, args=(lambda: h.node.handle_append_entries(beat(next(terms), 3)),)),
        threading.Thread(target=hammer, args=(h.node.tick,)),
        threading.Thread(target=hammer, args=(h.node.status,)),
    ]
    for worker in workers:
        worker.start()
    time.sleep(0.3)
    stop.set()
    for worker in workers:
        worker.join(timeout=30)

    assert errors == []
    status = h.node.status()
    assert h.disk().current_term <= status.current_term  # memory never behind disk


# ---------------------------------------------------------------------------
# 12. The background ticker (real threads, real time)
# ---------------------------------------------------------------------------


def _wait_for(condition: Callable[[], bool], seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


FAST = {"heartbeat_ms": 10, "election_timeout_min_ms": 50, "election_timeout_max_ms": 100}


def _real_node(tmp_path: Path, transport: StubTransport) -> tuple[RaftNode, Path]:
    config = make_config(tmp_path, **FAST)
    node = RaftNode(config, RaftStateStore(config.state_path, 1), transport)
    return node, config.state_path


def test_the_background_ticker_runs_elections_and_stops_cleanly(tmp_path: Path) -> None:
    transport = StubTransport()  # every peer is silent
    node, path = _real_node(tmp_path, transport)

    node.start()
    assert _wait_for(lambda: node.status().current_term >= 2)  # keeps re-running elections
    node.stop()

    assert node._thread is not None and not node._thread.is_alive()
    calls = len(transport.vote_calls)
    time.sleep(0.3)  # several election timeouts of silence
    assert len(transport.vote_calls) == calls
    assert node.status().role is Role.FOLLOWER
    assert RaftStateStore(path, 1).load().current_term >= 2


def test_a_running_leader_heartbeats_until_it_is_stopped(tmp_path: Path) -> None:
    transport = StubTransport()
    transport.on_vote = lambda peer, request: vote_reply(request.term, True)
    transport.on_append = lambda peer, request: append_reply(request.term)
    node, _ = _real_node(tmp_path, transport)

    node.start()
    assert _wait_for(lambda: node.status().role is Role.LEADER)
    assert _wait_for(lambda: len(transport.append_calls) >= 8)
    node.stop()

    sent = len(transport.append_calls)
    time.sleep(0.2)
    assert len(transport.append_calls) == sent
    assert node.status().role is Role.FOLLOWER  # a stopped node never claims leadership


def test_stopping_twice_and_restarting_a_stopped_node_are_handled(tmp_path: Path) -> None:
    node, _ = _real_node(tmp_path, StubTransport())
    node.start()
    with pytest.raises(RuntimeError, match="cannot be started"):
        node.start()
    node.stop()
    node.stop()  # harmless
    with pytest.raises(RuntimeError, match="cannot be started"):
        node.start()


def test_a_stopped_node_answers_nothing_and_changes_nothing(tmp_path: Path) -> None:
    node, path = _real_node(tmp_path, StubTransport())
    node.start()
    node.stop()
    before = RaftStateStore(path, 1).load()

    vote = node.handle_request_vote(vote_req(99, 2))
    beat_reply = node.handle_append_entries(beat(99, 2))

    assert vote.vote_granted is False
    assert beat_reply.success is False
    assert RaftStateStore(path, 1).load() == before


def test_a_node_that_was_never_started_can_be_stopped(tmp_path: Path) -> None:
    h = build(tmp_path)
    h.node.stop()
    assert h.node.status().role is Role.FOLLOWER


# ---------------------------------------------------------------------------
# 13. Whole clusters, simulated
#
# Every node shares one fake clock and talks over an in-memory network. Each
# step advances 10 ms and ticks the live nodes in id order, so a run is exactly
# reproducible from its seeds. SAFETY tests assert something is never true;
# LIVENESS tests assert something happens within a generous simulated bound.
# ---------------------------------------------------------------------------


class Cluster:
    def __init__(self, tmp_path: Path, size: int = 3, seed: int = 0) -> None:
        self.tmp_path = tmp_path
        self.size = size
        self.seed = seed
        self.clock = FakeClock()
        self.nodes: dict[int, RaftNode] = {}
        self.down: set[int] = set()
        self.isolated: set[int] = set()
        self.leaders_by_term: dict[int, set[int]] = defaultdict(set)
        for node_id in range(1, size + 1):
            self.boot(node_id)

    # -- membership of the simulation ---------------------------------------
    def boot(self, node_id: int) -> None:
        config = make_config(self.tmp_path, node_id, self.size)
        self.nodes[node_id] = RaftNode(
            config,
            RaftStateStore(config.state_path, node_id),
            _SimTransport(self, node_id),
            clock=self.clock,
            rng=random.Random(self.seed * 100 + node_id),
            dispatcher=inline,
            listener=self._record,
        )

    def _record(self, status: NodeStatus) -> None:
        if status.role is Role.LEADER:
            self.leaders_by_term[status.current_term].add(status.node_id)

    def crash(self, node_id: int) -> None:
        self.down.add(node_id)
        self.nodes[node_id].stop()

    def restart(self, node_id: int) -> None:
        self.down.discard(node_id)
        self.boot(node_id)

    # -- time ---------------------------------------------------------------
    def step(self) -> None:
        self.clock.advance(0.01)
        for node_id in sorted(self.nodes):
            if node_id not in self.down:
                self.nodes[node_id].tick()

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / 0.01)):
            self.step()

    def run_until(self, condition: Callable[[], bool], seconds: float) -> bool:
        for _ in range(round(seconds / 0.01)):
            if condition():
                return True
            self.step()
        return condition()

    # -- observation --------------------------------------------------------
    def live(self) -> list[RaftNode]:
        return [n for i, n in sorted(self.nodes.items()) if i not in self.down]

    def leaders(self) -> list[int]:
        return [n.status().node_id for n in self.live() if n.status().role is Role.LEADER]

    def one_leader_everyone_follows(self) -> bool:
        leaders = self.leaders()
        if len(leaders) != 1:
            return False
        statuses = [n.status() for n in self.live()]
        term = max(s.current_term for s in statuses)
        return all(s.current_term == term and s.leader_id == leaders[0] for s in statuses)

    def assert_election_safety(self) -> None:
        """At most one leader was ever elected in any single term."""
        for term, leaders in self.leaders_by_term.items():
            assert len(leaders) <= 1, f"term {term} had leaders {sorted(leaders)}"


class _SimTransport:
    def __init__(self, cluster: Cluster, me: int) -> None:
        self.cluster, self.me = cluster, me

    def _reachable(self, peer: int) -> bool:
        c = self.cluster
        blocked = c.down | c.isolated
        return self.me not in blocked and peer not in blocked

    def request_vote(self, peer_id, request, timeout):
        if not self._reachable(peer_id):
            return None
        return self.cluster.nodes[peer_id].handle_request_vote(request)

    def append_entries(self, peer_id, request, timeout):
        if not self._reachable(peer_id):
            return None
        return self.cluster.nodes[peer_id].handle_append_entries(request)


def test_liveness_a_three_node_cluster_elects_exactly_one_leader(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)

    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    cluster.assert_election_safety()


def test_liveness_a_stable_leader_is_not_disturbed(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    leader = cluster.leaders()[0]
    term = cluster.nodes[leader].status().current_term

    cluster.run(10.0)  # many election timeouts' worth of heartbeats

    assert cluster.leaders() == [leader]
    assert cluster.nodes[leader].status().current_term == term
    cluster.assert_election_safety()


@pytest.mark.parametrize("seed", range(1, 21))
def test_safety_and_liveness_hold_across_many_random_schedules(
    tmp_path: Path, seed: int
) -> None:
    cluster = Cluster(tmp_path, seed=seed)

    assert cluster.run_until(cluster.one_leader_everyone_follows, 10.0), seed
    cluster.assert_election_safety()


def test_liveness_a_new_leader_is_elected_after_the_leader_crashes(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    old_leader = cluster.leaders()[0]
    old_term = cluster.nodes[old_leader].status().current_term

    cluster.crash(old_leader)

    assert cluster.run_until(cluster.one_leader_everyone_follows, 10.0)
    new_leader = cluster.leaders()[0]
    assert new_leader != old_leader
    assert cluster.nodes[new_leader].status().current_term > old_term
    cluster.assert_election_safety()


def test_a_restarted_old_leader_rejoins_as_a_follower(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    old_leader = cluster.leaders()[0]
    cluster.crash(old_leader)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 10.0)
    new_leader = cluster.leaders()[0]
    new_term = cluster.nodes[new_leader].status().current_term

    cluster.restart(old_leader)

    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    assert cluster.leaders() == [new_leader]  # it did not seize leadership back
    rejoined = cluster.nodes[old_leader].status()
    assert rejoined.role is Role.FOLLOWER
    assert rejoined.current_term == new_term
    cluster.assert_election_safety()


def test_safety_no_leader_can_be_elected_without_a_majority(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    leader = cluster.leaders()[0]
    others = [i for i in cluster.nodes if i != leader]
    cluster.crash(leader)
    cluster.crash(others[0])
    survivor = others[1]
    elected_before = {t: set(v) for t, v in cluster.leaders_by_term.items()}

    cluster.run(15.0)

    status = cluster.nodes[survivor].status()
    assert status.role is not Role.LEADER
    assert status.current_term > max(elected_before)  # it kept trying
    assert {t: set(v) for t, v in cluster.leaders_by_term.items()} == elected_before


def test_liveness_leadership_returns_when_the_majority_does(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    leader = cluster.leaders()[0]
    others = [i for i in cluster.nodes if i != leader]
    cluster.crash(leader)
    cluster.crash(others[0])
    cluster.run(8.0)
    assert cluster.leaders() == []

    cluster.restart(others[0])

    assert cluster.run_until(cluster.one_leader_everyone_follows, 15.0)
    cluster.assert_election_safety()


def test_safety_terms_and_votes_survive_a_full_cluster_restart(tmp_path: Path) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    before = {i: n.status().current_term for i, n in cluster.nodes.items()}

    for node_id in list(cluster.nodes):
        cluster.crash(node_id)
    for node_id in list(cluster.nodes):
        cluster.restart(node_id)

    after = {i: n.status().current_term for i, n in cluster.nodes.items()}
    assert after == before  # nobody's term moved backwards
    assert cluster.run_until(cluster.one_leader_everyone_follows, 10.0)
    assert cluster.nodes[cluster.leaders()[0]].status().current_term > max(before.values())
    cluster.assert_election_safety()


def test_a_partitioned_old_leader_steps_down_when_it_hears_the_new_term(
    tmp_path: Path,
) -> None:
    cluster = Cluster(tmp_path)
    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    old_leader = cluster.leaders()[0]
    old_term = cluster.nodes[old_leader].status().current_term

    cluster.isolated.add(old_leader)  # alive, but cut off from everyone
    assert cluster.run_until(
        lambda: any(
            n.status().role is Role.LEADER and n.status().node_id != old_leader
            for n in cluster.live()
        ),
        10.0,
    )
    # Without CheckQuorum the cut-off leader still believes it leads its old
    # term. That is allowed: it cannot commit anything, and it leads a
    # *different*, older term than the new leader.
    assert cluster.nodes[old_leader].status().role is Role.LEADER
    assert cluster.nodes[old_leader].status().current_term == old_term
    cluster.assert_election_safety()

    cluster.isolated.clear()  # heal the partition

    assert cluster.run_until(cluster.one_leader_everyone_follows, 5.0)
    assert cluster.nodes[old_leader].status().role is Role.FOLLOWER
    assert cluster.nodes[old_leader].status().current_term > old_term
    cluster.assert_election_safety()
