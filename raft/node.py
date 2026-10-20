"""The Raft election state machine for one node (Phase 6.2).

Scope. Leader election and empty-heartbeat handling, and nothing else: there is
no replicated log, no commit logic and no inventory. The RPC messages already
carry the Phase 7 fields (log position, entries, commit index) so the contract
will not change, but this module answers them with the empty-log convention
(index 0, term 0) and refuses to append entries.

The three roles, in one paragraph. A FOLLOWER waits for a leader. If it hears
nothing for a randomized *election timeout* it becomes a CANDIDATE: it bumps its
term, votes for itself and asks every peer for a vote. A candidate that gathers a
majority becomes the LEADER and sends heartbeats so nobody else starts an
election. Any node that sees a higher term, from any message, drops back to
FOLLOWER at that term. The randomized timeout is what makes split votes rare and
self-correcting: two candidates rarely time out together twice in a row.

Safety rules this file is responsible for (Raft paper, Figure 2):

* A node grants at most one vote per term, and repeats it only to the same
  candidate. ``voted_for`` and ``current_term`` reach disk *before* any reply or
  request that depends on them (:meth:`RaftNode._persist`); if the write fails
  the node does not grant the vote, start the election or adopt the term.
* A vote is refused to a candidate whose log is less up to date than ours.
* A node never lowers its term.

Concurrency design, in the order it matters:

1. **One lock, never held across the network.** ``self._lock`` (an ``RLock``)
   guards every field. Handlers and timers take it, decide, update state, build
   the messages to send, and *release it* before anything touches a transport.
   The only I/O done while holding it is the single-row state save, which is the
   point: "persist, then answer" must not interleave with another decision.
2. **Lock order is node lock, then the store's own lock.** The store never calls
   back into the node, so the order cannot invert.
3. **Replies carry the term they were sent in.** A late reply is judged against
   that term, so a vote from an old election round can never help a newer one,
   and a peer's duplicate grant is counted once (votes are a set of ids).
4. **Time is injected.** :meth:`RaftNode.tick` is the only place the clock is
   read for timers; the background thread merely calls it. Tests drive ``tick``
   with a fake clock and need no sleeping.
5. **No busy-waiting.** The thread sleeps on an ``Event`` until the next
   deadline and is woken early only when the schedule changes (a new leader) or
   the node stops.
"""

from __future__ import annotations

import enum
import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Protocol

from proto import raft_pb2
from raft.config import RaftConfig
from raft.storage import RaftState, RaftStateError

__all__ = [
    "Clock",
    "Dispatcher",
    "EmptyLog",
    "MonotonicClock",
    "NodeStatus",
    "PeerTransport",
    "RaftLog",
    "RaftNode",
    "Role",
    "StateStore",
]

_logger = logging.getLogger("raft.node")

#: The ticker never sleeps longer than this, so a clock that jumps (or a missed
#: wake-up) delays a timer by at most a quarter of a second.
_MAX_IDLE_SECONDS: Final[float] = 0.25
_MIN_IDLE_SECONDS: Final[float] = 0.001
_STOP_JOIN_SECONDS: Final[float] = 10.0


class Role(enum.Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class _Lifecycle(enum.Enum):
    NEW = "new"  # constructed; handlers and tick() work, no thread yet
    RUNNING = "running"  # start() was called
    STOPPED = "stopped"  # stop() was called; the node answers nothing


# --------------------------------------------------------------------------
# Injectable dependencies
# --------------------------------------------------------------------------
class Clock(Protocol):
    """A source of monotonic time in seconds."""

    def now(self) -> float: ...


class MonotonicClock:
    """The real clock. Monotonic, so wall-clock adjustments cannot trigger or
    suppress an election."""

    def now(self) -> float:
        return time.monotonic()


class StateStore(Protocol):
    """What the node needs from :class:`raft.storage.RaftStateStore`."""

    def load(self) -> RaftState: ...

    def save(self, term: int, voted_for: int | None) -> RaftState: ...


class RaftLog(Protocol):
    """The node's view of its log.

    Phase 7 supplies a real one; Phase 6 uses :class:`EmptyLog`. Index 0 and
    term 0 mean "before the first entry", so an empty log is ``(0, 0)``.
    """

    def last_index(self) -> int: ...

    def last_term(self) -> int: ...

    def term_at(self, index: int) -> int | None:
        """Term of the entry at ``index``, or ``None`` if there is none."""
        ...


class EmptyLog:
    """The Phase 6 log: no entries, ever."""

    def last_index(self) -> int:
        return 0

    def last_term(self) -> int:
        return 0

    def term_at(self, index: int) -> int | None:
        return 0 if index == 0 else None


class PeerTransport(Protocol):
    """How the node reaches its peers. Phase 6.3 provides the gRPC one.

    Each call returns the peer's response, or ``None`` if the peer could not be
    reached in time. Implementations must not raise for ordinary network
    failure; a lost message is simply a missing reply, which Raft tolerates.
    """

    def request_vote(
        self, peer_id: int, request: raft_pb2.RequestVoteRequest, timeout: float
    ) -> raft_pb2.RequestVoteResponse | None: ...

    def append_entries(
        self, peer_id: int, request: raft_pb2.AppendEntriesRequest, timeout: float
    ) -> raft_pb2.AppendEntriesResponse | None: ...


#: Runs a unit of outbound work. The default hands it to a thread pool so one
#: slow peer cannot delay the others; tests pass a synchronous or a queueing one.
Dispatcher = Callable[[Callable[[], None]], None]


@dataclass(frozen=True)
class NodeStatus:
    """A consistent snapshot of what a node currently believes."""

    node_id: int
    role: Role
    current_term: int
    voted_for: int | None
    leader_id: int | None
    commit_index: int
    last_log_index: int


# --------------------------------------------------------------------------
# The node
# --------------------------------------------------------------------------
class RaftNode:
    """One Raft member's election logic.

    ``listener``, if given, is called with a :class:`NodeStatus` each time the
    node's role or term changes. It runs while the node lock is held, so it must
    be quick and must not block; recording the status in a list is the intended
    use (it is how tests prove "at most one leader per term").
    """

    def __init__(
        self,
        config: RaftConfig,
        store: StateStore,
        transport: PeerTransport,
        *,
        clock: Clock | None = None,
        rng: random.Random | None = None,
        log: RaftLog | None = None,
        dispatcher: Dispatcher | None = None,
        listener: Callable[[NodeStatus], None] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._transport = transport
        self._clock: Clock = clock or MonotonicClock()
        self._rng = rng or random.Random()  # timing jitter, not security
        self._log: RaftLog = log or EmptyLog()
        self._dispatcher = dispatcher
        self._listener = listener

        self._member_ids = frozenset(m.node_id for m in config.members)
        self._peer_ids = tuple(m.node_id for m in config.peers)
        self._rpc_timeout = config.election_timeout_min_ms / 2000.0
        self._heartbeat_seconds = config.heartbeat_ms / 1000.0

        self._lock = threading.RLock()
        self._wakeup = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._lifecycle = _Lifecycle.NEW

        # Persistent state, loaded from disk: a restart must not forget a vote.
        # A corrupt file raises here, so a node never starts from guessed state.
        persisted = store.load()
        self._term: int = persisted.current_term
        self._voted_for: int | None = persisted.voted_for

        # Volatile state.
        self._role = Role.FOLLOWER
        self._leader_id: int | None = None
        self._votes: set[int] = set()
        self._heartbeat_in_flight: set[int] = set()
        self._deadline = self._clock.now() + self._random_timeout()
        self._next_heartbeat = 0.0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        """Start the background ticker. The node begins as a Follower."""
        with self._lock:
            if self._lifecycle is not _Lifecycle.NEW:
                raise RuntimeError(
                    f"Raft node {self._config.node_id} cannot be started from "
                    f"state {self._lifecycle.value}; create a new node instead"
                )
            self._lifecycle = _Lifecycle.RUNNING
            # A full, fresh timeout from the moment the node actually starts.
            self._deadline = self._clock.now() + self._random_timeout()
            if self._dispatcher is None:
                workers = max(2, 2 * len(self._peer_ids))
                self._executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix=f"raft{self._config.node_id}-rpc",
                )
                self._dispatcher = self._submit_to_pool
            self._thread = threading.Thread(
                target=self._run,
                name=f"raft{self._config.node_id}-ticker",
                daemon=True,
            )
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Stop for good: end the ticker, drain outbound work, answer nothing.

        Safe to call more than once and from any thread. The persisted term and
        vote are untouched, so a new node on the same store resumes correctly.
        """
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return
            self._lifecycle = _Lifecycle.STOPPED
            # A stopped node must never report itself leader.
            self._role = Role.FOLLOWER
            self._leader_id = None
            self._votes.clear()
            thread, executor = self._thread, self._executor
        self._wakeup.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_STOP_JOIN_SECONDS)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def status(self) -> NodeStatus:
        with self._lock:
            return self._snapshot()

    # -- RequestVote --------------------------------------------------------
    def handle_request_vote(
        self, request: raft_pb2.RequestVoteRequest
    ) -> raft_pb2.RequestVoteResponse:
        """Decide a vote request (Raft paper, Figure 2).

        The reply's ``term`` is this node's term after the call, so a stale
        candidate learns it is behind.
        """
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return self._vote_response(False)
            candidate = request.candidate_id
            if candidate == self._config.node_id or candidate not in self._member_ids:
                # Not a cluster member: ignore it entirely, including its term.
                _logger.warning(
                    "node %s: ignoring vote request from unknown candidate %s",
                    self._config.node_id,
                    candidate,
                )
                return self._vote_response(False)

            if request.term < self._term:
                return self._vote_response(False)
            if request.term > self._term and not self._adopt_term(request.term):
                return self._vote_response(False)

            # request.term == self._term from here on.
            may_vote = self._voted_for in (None, candidate)
            if not (may_vote and self._log_is_up_to_date(request)):
                return self._vote_response(False)

            if self._voted_for != candidate:
                try:
                    self._persist(self._term, candidate)
                except RaftStateError:
                    _logger.exception(
                        "node %s: could not persist vote for %s in term %s; "
                        "refusing it",
                        self._config.node_id,
                        candidate,
                        self._term,
                    )
                    return self._vote_response(False)
                self._voted_for = candidate
            # Granting a vote counts as hearing from a viable leader-to-be.
            self._reset_deadline()
            return self._vote_response(True)

    def handle_request_vote_response(
        self,
        peer_id: int,
        sent_term: int,
        response: raft_pb2.RequestVoteResponse | None,
    ) -> None:
        """Account for a peer's answer to the vote request sent in ``sent_term``.

        ``None`` means no reply. A reply only counts if this node is still a
        candidate *in that same term*; anything else is a stale round.
        """
        if response is None:
            return
        tasks: list[Callable[[], None]] = []
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return
            if response.term > self._term:
                self._adopt_term(response.term)
                return
            if (
                self._role is not Role.CANDIDATE
                or self._term != sent_term
                or response.term != sent_term
                or peer_id not in self._member_ids
                or peer_id == self._config.node_id
            ):
                return  # a stale or foreign reply: ignore it
            if not response.vote_granted:
                return
            self._votes.add(peer_id)  # a set: the same peer counts once
            if len(self._votes) >= self._config.quorum:
                tasks = self._become_leader()
        self._send(tasks)

    # -- AppendEntries ------------------------------------------------------
    def handle_append_entries(
        self, request: raft_pb2.AppendEntriesRequest
    ) -> raft_pb2.AppendEntriesResponse:
        """Handle a leader's heartbeat (and, from Phase 7, its entries).

        A request from the current or a newer term is *authority*: this node
        adopts the term, remembers the leader and resets its election timer,
        whether or not the log part of the request can be accepted.
        """
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return self._append_response(False)
            leader = request.leader_id
            if leader == self._config.node_id or leader not in self._member_ids:
                _logger.warning(
                    "node %s: ignoring AppendEntries from unknown leader %s",
                    self._config.node_id,
                    leader,
                )
                return self._append_response(False)

            if request.term < self._term:
                return self._append_response(False)
            if request.term > self._term and not self._adopt_term(request.term):
                return self._append_response(False)

            # request.term == self._term from here on.
            if self._role is Role.LEADER:
                # Impossible if elections are safe: two leaders in one term.
                _logger.error(
                    "node %s: node %s claims leadership of term %s, which this "
                    "node already leads",
                    self._config.node_id,
                    leader,
                    self._term,
                )
                return self._append_response(False)
            if self._role is Role.CANDIDATE:
                self._role = Role.FOLLOWER  # someone else won this term
                self._votes.clear()
                self._notify()
            self._leader_id = leader
            self._reset_deadline()
            return self._check_log_part(request)

    def handle_append_entries_response(
        self,
        peer_id: int,
        sent_term: int,
        response: raft_pb2.AppendEntriesResponse | None,
    ) -> None:
        """Account for a follower's answer to a heartbeat sent in ``sent_term``.

        Phase 6 only cares about one thing: a higher term means this leader has
        been superseded.
        """
        if response is None:
            return
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return
            if response.term > self._term:
                self._adopt_term(response.term)

    # -- timers -------------------------------------------------------------
    def tick(self) -> float:
        """Advance timers once and return the seconds until the next is due.

        Starts an election if a Follower or Candidate has outlived its timeout,
        and sends heartbeats if a Leader is due. Called by the ticker thread;
        tests call it directly with a fake clock.
        """
        tasks: list[Callable[[], None]] = []
        with self._lock:
            if self._lifecycle is _Lifecycle.STOPPED:
                return _MAX_IDLE_SECONDS
            now = self._clock.now()
            if self._role is Role.LEADER:
                if now >= self._next_heartbeat:
                    self._next_heartbeat = now + self._heartbeat_seconds
                    tasks = self._heartbeat_tasks()
            elif now >= self._deadline:
                tasks = self._begin_election()

            due = self._next_heartbeat if self._role is Role.LEADER else self._deadline
            delay = max(_MIN_IDLE_SECONDS, min(_MAX_IDLE_SECONDS, due - now))
        self._send(tasks)
        return delay

    # ======================================================================
    # Everything below assumes ``self._lock`` is held unless stated otherwise.
    # ======================================================================
    def _random_timeout(self) -> float:
        low = self._config.election_timeout_min_ms
        high = self._config.election_timeout_max_ms
        return self._rng.uniform(low, high) / 1000.0

    def _reset_deadline(self) -> None:
        self._deadline = self._clock.now() + self._random_timeout()

    def _snapshot(self) -> NodeStatus:
        return NodeStatus(
            node_id=self._config.node_id,
            role=self._role,
            current_term=self._term,
            voted_for=self._voted_for,
            leader_id=self._leader_id,
            commit_index=0,  # no commit logic until Phase 7
            last_log_index=self._log.last_index(),
        )

    def _notify(self) -> None:
        if self._listener is None:
            return
        try:
            self._listener(self._snapshot())
        except Exception:  # an observer must never break the node
            _logger.exception("node %s: listener failed", self._config.node_id)

    def _persist(self, term: int, voted_for: int | None) -> None:
        """Make ``(term, voted_for)`` durable. Raises RaftStateError on failure,
        in which case the caller must leave its in-memory state unchanged."""
        self._store.save(term, voted_for)

    def _vote_response(self, granted: bool) -> raft_pb2.RequestVoteResponse:
        return raft_pb2.RequestVoteResponse(term=self._term, vote_granted=granted)

    def _append_response(
        self,
        success: bool,
        *,
        match_index: int = 0,
        conflict_index: int = 0,
        conflict_term: int = 0,
    ) -> raft_pb2.AppendEntriesResponse:
        return raft_pb2.AppendEntriesResponse(
            term=self._term,
            success=success,
            match_index=match_index,
            conflict_index=conflict_index,
            conflict_term=conflict_term,
        )

    def _log_is_up_to_date(self, request: raft_pb2.RequestVoteRequest) -> bool:
        """The candidate's log is at least as up to date as ours (Raft, 5.4.1):
        a later last term wins; equal last terms compare by length."""
        mine_term, mine_index = self._log.last_term(), self._log.last_index()
        if request.last_log_term != mine_term:
            return request.last_log_term > mine_term
        return request.last_log_index >= mine_index

    def _adopt_term(self, new_term: int) -> bool:
        """Move to a higher term as a Follower with no vote and no leader.

        Persists first. Returns ``False`` (and changes nothing) if the write
        fails, in which case the caller must not act on the new term.
        """
        if new_term <= self._term:
            return True
        try:
            self._persist(new_term, None)
        except RaftStateError:
            _logger.exception(
                "node %s: could not persist term %s; staying in term %s",
                self._config.node_id,
                new_term,
                self._term,
            )
            return False
        was_leader = self._role is Role.LEADER
        self._term = new_term
        self._voted_for = None
        self._role = Role.FOLLOWER
        self._leader_id = None
        self._votes.clear()
        if was_leader:
            # A leader has no election deadline; it needs one now. Everyone
            # else keeps the timer they had: merely hearing of a newer term is
            # not contact with a leader, and resetting here would let a
            # candidate with an out-of-date log keep postponing everyone's
            # elections. (Granting a vote or hearing a leader resets it, in
            # the handlers.)
            self._reset_deadline()
        self._wakeup.set()  # a leader's heartbeat schedule is now void
        self._notify()
        return True

    def _begin_election(self) -> list[Callable[[], None]]:
        """Become a Candidate in the next term and build the vote requests."""
        new_term = self._term + 1
        me = self._config.node_id
        try:
            self._persist(new_term, me)  # the self-vote is durable before any ask
        except RaftStateError:
            _logger.exception(
                "node %s: could not persist the start of term %s; will retry "
                "after the next timeout",
                me,
                new_term,
            )
            self._reset_deadline()
            return []

        self._term = new_term
        self._voted_for = me
        self._role = Role.CANDIDATE
        self._leader_id = None
        self._votes = {me}
        self._reset_deadline()  # a fresh timeout bounds this election round
        _logger.info("node %s: candidate for term %s", me, new_term)
        self._notify()

        if len(self._votes) >= self._config.quorum:
            return self._become_leader()  # a one-member cluster elects itself

        request = raft_pb2.RequestVoteRequest(
            term=new_term,
            candidate_id=me,
            last_log_index=self._log.last_index(),
            last_log_term=self._log.last_term(),
        )
        return [self._vote_task(peer, new_term, request) for peer in self._peer_ids]

    def _become_leader(self) -> list[Callable[[], None]]:
        self._role = Role.LEADER
        self._leader_id = self._config.node_id
        self._next_heartbeat = self._clock.now() + self._heartbeat_seconds
        _logger.info(
            "node %s: leader of term %s with votes %s",
            self._config.node_id,
            self._term,
            sorted(self._votes),
        )
        self._wakeup.set()  # the ticker must now follow the heartbeat schedule
        self._notify()
        return self._heartbeat_tasks()

    def _check_log_part(
        self, request: raft_pb2.AppendEntriesRequest
    ) -> raft_pb2.AppendEntriesResponse:
        """The log half of AppendEntries, after the leader has been accepted."""
        last = self._log.last_index()
        if request.prev_log_index > last:
            return self._append_response(False, conflict_index=last + 1)
        local_term = self._log.term_at(request.prev_log_index)
        if local_term != request.prev_log_term:
            return self._append_response(
                False,
                conflict_index=request.prev_log_index,
                conflict_term=local_term or 0,
            )
        if request.entries:
            # Phase 7 will append these. Until then refuse rather than pretend.
            _logger.warning(
                "node %s: refusing %s log entries; replication is not implemented",
                self._config.node_id,
                len(request.entries),
            )
            return self._append_response(False, match_index=last)
        return self._append_response(True, match_index=request.prev_log_index)

    # -- outbound work (built under the lock, executed outside it) ----------
    def _vote_task(
        self, peer_id: int, term: int, request: raft_pb2.RequestVoteRequest
    ) -> Callable[[], None]:
        def run() -> None:
            with self._lock:
                if self._role is not Role.CANDIDATE or self._term != term:
                    return  # the round ended before this ask went out
            reply = self._transport.request_vote(peer_id, request, self._rpc_timeout)
            self.handle_request_vote_response(peer_id, term, reply)

        return run

    def _heartbeat_tasks(self) -> list[Callable[[], None]]:
        term = self._term
        request = raft_pb2.AppendEntriesRequest(
            term=term,
            leader_id=self._config.node_id,
            prev_log_index=self._log.last_index(),
            prev_log_term=self._log.last_term(),
            leader_commit=0,  # no commit logic until Phase 7
        )
        tasks = []
        for peer in self._peer_ids:
            if peer in self._heartbeat_in_flight:
                continue  # that peer is slow; don't pile requests onto it
            self._heartbeat_in_flight.add(peer)
            tasks.append(self._heartbeat_task(peer, term, request))
        return tasks

    def _heartbeat_task(
        self, peer_id: int, term: int, request: raft_pb2.AppendEntriesRequest
    ) -> Callable[[], None]:
        def run() -> None:
            try:
                with self._lock:
                    if self._role is not Role.LEADER or self._term != term:
                        return  # stepped down: leader work stops here
                reply = self._transport.append_entries(
                    peer_id, request, self._rpc_timeout
                )
                self.handle_append_entries_response(peer_id, term, reply)
            finally:
                with self._lock:
                    self._heartbeat_in_flight.discard(peer_id)

        return run

    # -- dispatch (always called with the lock released) ---------------------
    def _send(self, tasks: list[Callable[[], None]]) -> None:
        for task in tasks:
            self._dispatch(task)

    def _dispatch(self, task: Callable[[], None]) -> None:
        dispatcher = self._dispatcher
        if dispatcher is None:  # tick() before start(): run in the caller
            self._run_safely(task)
            return
        try:
            dispatcher(task)
        except RuntimeError:
            # The pool was shut down by stop(); the work is moot.
            _logger.debug("node %s: dropped work after stop", self._config.node_id)

    def _submit_to_pool(self, task: Callable[[], None]) -> None:
        executor = self._executor
        if executor is not None:
            executor.submit(self._run_safely, task)

    def _run_safely(self, task: Callable[[], None]) -> None:
        try:
            task()
        except Exception:  # one bad RPC must not kill a worker
            _logger.exception("node %s: outbound task failed", self._config.node_id)

    def _run(self) -> None:
        """Ticker thread body: tick, then sleep until something is due."""
        while True:
            self._wakeup.clear()
            with self._lock:
                if self._lifecycle is _Lifecycle.STOPPED:
                    return
            try:
                delay = self.tick()
            except Exception:  # keep the node alive and retry shortly
                _logger.exception("node %s: tick failed", self._config.node_id)
                delay = 0.05
            self._wakeup.wait(delay)
