"""Raft election across real operating-system processes.

Every test launches three actual ``python -m raft.server`` processes, each with
its own port and its own temporary state file, and watches them only from the
outside: through the ``GetState`` RPC and through their log files. The test
process holds the only handles, tracks the PIDs of exactly the children it
started, and terminates just those in a ``finally`` block. No other Python
process is ever touched.

Two ways of stopping a node are kept distinct on purpose:

* **graceful**: Ctrl+Break on Windows, SIGTERM elsewhere. The node logs that it
  stopped and exits with status 0.
* **abrupt**: ``Popen.kill()`` (TerminateProcess / SIGKILL). Nothing runs; the
  node simply vanishes, which is what a crash looks like. It is safe because a
  term and vote reach disk before they are ever used.

Nothing here reads or writes ``data/``: the state files live in ``tmp_path``,
and one test checks that no production Raft state file appeared.

These tests do not assume that a *partitioned* old leader instantly reports
Follower: without CheckQuorum it legitimately keeps believing it leads its old
term until it hears a newer one. No test here isolates a live leader; the
leaders removed below are dead processes.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from proto import raft_pb2
from raft.config import Member
from raft.peers import fetch_state
from raft.storage import RaftStateStore
from tests.integration.raft_support import free_ports, wait_until

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_STATE_FILES = [PROJECT_ROOT / "data" / f"raft_node{i}.db" for i in (1, 2, 3)]

HEARTBEAT_MS, ELECTION_MIN_MS, ELECTION_MAX_MS = 50, 300, 600
LEADER_LINE = re.compile(r"node (\d+): leader of term (\d+)")
IS_WINDOWS = sys.platform == "win32"


class NodeProcess:
    """One ``raft.server`` child process and the files it writes."""

    def __init__(self, node_id: int, env: dict[str, str], log_dir: Path) -> None:
        self.node_id = node_id
        self._env = env
        self._log_dir = log_dir
        self.proc: subprocess.Popen[bytes] | None = None
        self._handle = None
        self.log_paths: list[Path] = []

    def start(self) -> None:
        assert not self.alive(), f"node {self.node_id} is already running"
        self._close_handle()
        path = self._log_dir / f"node{self.node_id}.run{len(self.log_paths) + 1}.log"
        self.log_paths.append(path)
        self._handle = path.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-B", "-m", "raft.server"],
            cwd=PROJECT_ROOT,
            env=self._env,
            stdout=self._handle,
            stderr=subprocess.STDOUT,
            # Own process group, so Ctrl+Break reaches only this child.
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop_gracefully(self, timeout: float = 20.0) -> int:
        """Ask politely and wait for a clean exit; returns the exit status."""
        assert self.proc is not None
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if IS_WINDOWS else signal.SIGTERM)
        return self.proc.wait(timeout=timeout)

    def kill(self) -> int:
        """Abrupt death, no cleanup of any kind; returns the exit status."""
        assert self.proc is not None
        self.proc.kill()
        return self.proc.wait(timeout=10)

    def cleanup(self) -> None:
        if self.alive():
            self.kill()
        self._close_handle()

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def log(self, run: int = -1) -> str:
        return self.log_paths[run].read_text(encoding="utf-8", errors="replace")

    def all_logs(self) -> str:
        return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in self.log_paths)


class ProcessCluster:
    def __init__(self, tmp_path: Path, size: int = 3) -> None:
        self.tmp_path = tmp_path
        ports = free_ports(size)
        self.members = tuple(Member(i + 1, "127.0.0.1", ports[i]) for i in range(size))
        peers = ",".join(f"{m.node_id}={m.address}" for m in self.members)
        base = {k: v for k, v in os.environ.items() if not k.startswith("RAFT_")}
        base.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            RAFT_PEERS=peers,
            RAFT_HEARTBEAT_MS=str(HEARTBEAT_MS),
            RAFT_ELECTION_TIMEOUT_MIN_MS=str(ELECTION_MIN_MS),
            RAFT_ELECTION_TIMEOUT_MAX_MS=str(ELECTION_MAX_MS),
            INVENTORY_LOG_LEVEL="INFO",
        )
        self.nodes: dict[int, NodeProcess] = {}
        for m in self.members:
            env = dict(base, RAFT_NODE_ID=str(m.node_id), RAFT_STATE_PATH=str(self.state_path(m.node_id)))
            self.nodes[m.node_id] = NodeProcess(m.node_id, env, tmp_path)

    # -- paths and control --------------------------------------------------
    def state_path(self, node_id: int) -> Path:
        return self.tmp_path / f"raft_node{node_id}.db"

    def address(self, node_id: int) -> str:
        return next(m.address for m in self.members if m.node_id == node_id)

    def start_all(self) -> None:
        for node in self.nodes.values():
            node.start()

    def alive_ids(self) -> list[int]:
        return [i for i, n in self.nodes.items() if n.alive()]

    def cleanup(self) -> None:
        for node in self.nodes.values():
            node.cleanup()

    # -- observation --------------------------------------------------------
    def states(self) -> dict[int, raft_pb2.GetStateResponse]:
        found = {}
        for node_id in self.alive_ids():
            state = fetch_state(self.address(node_id), timeout=1.0)
            if state is not None:
                found[node_id] = state
        return found

    def agreed(self) -> tuple[int, int] | None:
        """``(leader_id, term)`` when every live node answers, exactly one leads,
        and all of them agree on who and in which term; otherwise ``None``."""
        states = self.states()
        alive = self.alive_ids()
        if not alive or len(states) != len(alive):
            return None
        leaders = [s for s in states.values() if s.role == raft_pb2.NODE_ROLE_LEADER]
        if len(leaders) != 1:
            return None
        lead = leaders[0]
        agree = all(
            s.current_term == lead.current_term and s.leader_id == lead.node_id
            for s in states.values()
        )
        return (lead.node_id, lead.current_term) if agree else None

    def wait_for_agreement(self, what: str, timeout: float = 30.0) -> tuple[int, int]:
        try:
            return wait_until(self.agreed, timeout=timeout, interval=0.05, message=what)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.diagnostics()}") from None

    def diagnostics(self) -> str:
        parts = []
        for node_id, node in self.nodes.items():
            tail = "\n".join(node.all_logs().splitlines()[-12:]) if node.log_paths else "(never started)"
            parts.append(f"--- node {node_id} (alive={node.alive()}) ---\n{tail}")
        return "\n".join(parts)

    def leaders_by_term(self) -> dict[int, set[int]]:
        """Every 'leader of term N' line any node ever logged, across restarts."""
        found: dict[int, set[int]] = {}
        for node in self.nodes.values():
            if node.log_paths:
                for node_id, term in LEADER_LINE.findall(node.all_logs()):
                    found.setdefault(int(term), set()).add(int(node_id))
        return found

    def assert_election_safety(self) -> None:
        for term, ids in self.leaders_by_term().items():
            assert len(ids) <= 1, f"term {term} was led by several nodes: {sorted(ids)}"


@pytest.fixture
def cluster(tmp_path: Path):
    c = ProcessCluster(tmp_path)
    try:
        yield c
    finally:
        c.cleanup()  # only the children this fixture started


def _survivor_setup(cluster: ProcessCluster) -> tuple[int, int, int, int]:
    """Elect a leader, then remove a majority: the leader dies abruptly and one
    follower stops gracefully. Returns (survivor, old_leader, stopped, old_term)."""
    cluster.start_all()
    leader, term = cluster.wait_for_agreement("the first leader")
    followers = [i for i in cluster.nodes if i != leader]
    stopped, survivor = followers
    assert cluster.nodes[leader].kill() != 0
    assert cluster.nodes[stopped].stop_gracefully() == 0
    return survivor, leader, stopped, term


# ---------------------------------------------------------------------------
# 1. Election
# ---------------------------------------------------------------------------


def test_three_processes_elect_one_leader(cluster: ProcessCluster) -> None:
    before = [p.exists() for p in PRODUCTION_STATE_FILES]

    cluster.start_all()
    leader, term = cluster.wait_for_agreement("one agreed leader")

    states = cluster.states()
    assert sorted(states) == [1, 2, 3]
    assert sum(s.role == raft_pb2.NODE_ROLE_LEADER for s in states.values()) == 1
    assert term >= 1
    cluster.assert_election_safety()
    assert cluster.leaders_by_term()[term] == {leader}

    # Independent state: three distinct files, each tagged with its own node id.
    for node_id in (1, 2, 3):
        path = cluster.state_path(node_id)
        assert path.is_file()
        assert RaftStateStore(path, node_id).load().current_term == term
    # ...and no production Raft state file was created anywhere.
    assert [p.exists() for p in PRODUCTION_STATE_FILES] == before
    assert "RaftService listening" in cluster.nodes[leader].log()


# ---------------------------------------------------------------------------
# 2-3. Losing the leader: graceful versus abrupt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["graceful", "abrupt"])
def test_stopping_the_leader_triggers_a_re_election_in_a_higher_term(
    cluster: ProcessCluster, how: str
) -> None:
    cluster.start_all()
    old_leader, old_term = cluster.wait_for_agreement("the first leader")
    node = cluster.nodes[old_leader]

    if how == "graceful":
        assert node.stop_gracefully() == 0
        assert "Raft node stopped" in node.log()  # it shut down on purpose
    else:
        assert node.kill() != 0
        assert "Raft node stopped" not in node.log()  # nothing ran: it was a crash

    new_leader, new_term = cluster.wait_for_agreement("a new leader among the survivors")

    assert new_leader != old_leader
    assert new_term > old_term
    cluster.assert_election_safety()


# ---------------------------------------------------------------------------
# 4-5. Restarting the old leader
# ---------------------------------------------------------------------------


def test_a_restarted_old_leader_keeps_its_state_and_follows_the_new_leader(
    cluster: ProcessCluster,
) -> None:
    cluster.start_all()
    old_leader, old_term = cluster.wait_for_agreement("the first leader")
    cluster.nodes[old_leader].kill()
    new_leader, new_term = cluster.wait_for_agreement("a new leader")

    on_disk = RaftStateStore(cluster.state_path(old_leader), old_leader).load()
    assert on_disk.current_term == old_term  # the crash lost nothing it had saved
    assert on_disk.voted_for == old_leader  # a leader voted for itself

    cluster.nodes[old_leader].start()

    # The restarted process announces the state it recovered from its own file.
    wait_until(
        lambda: "persisted term=" in cluster.nodes[old_leader].log(),
        timeout=20,
        message="the restarted node to log its recovered state",
    )
    assert (
        f"persisted term={old_term} voted_for={old_leader}"
        in cluster.nodes[old_leader].log()
    )

    leader, term = cluster.wait_for_agreement("all three nodes to agree again")
    assert (leader, term) == (new_leader, new_term)  # it did not seize leadership back
    state = cluster.states()[old_leader]
    assert state.role == raft_pb2.NODE_ROLE_FOLLOWER
    assert state.leader_id == new_leader  # it recognises the current leader
    assert state.current_term == new_term
    assert RaftStateStore(cluster.state_path(old_leader), old_leader).load().current_term >= old_term
    cluster.assert_election_safety()


# ---------------------------------------------------------------------------
# 6-7. No majority, then a majority again
# ---------------------------------------------------------------------------


def test_a_lone_survivor_cannot_elect_itself(cluster: ProcessCluster) -> None:
    survivor, _, _, old_term = _survivor_setup(cluster)
    terms_seen = []

    deadline = time.monotonic() + 4.0  # many election timeouts
    while time.monotonic() < deadline:
        state = fetch_state(cluster.address(survivor), timeout=1.0)
        assert state is not None, "the survivor itself must stay up"
        assert state.role != raft_pb2.NODE_ROLE_LEADER, "a minority elected a leader"
        terms_seen.append(state.current_term)
        time.sleep(0.05)

    assert max(terms_seen) >= old_term + 2  # it kept trying, round after round
    assert "leader of term" not in cluster.nodes[survivor].log()
    cluster.assert_election_safety()


def test_leadership_returns_once_a_majority_is_available_again(
    cluster: ProcessCluster,
) -> None:
    survivor, old_leader, stopped, _ = _survivor_setup(cluster)
    time.sleep(1.5)  # the survivor spins through elections it cannot win
    inflated = fetch_state(cluster.address(survivor), timeout=1.0)
    assert inflated is not None and inflated.role != raft_pb2.NODE_ROLE_LEADER

    cluster.nodes[stopped].start()  # a second node: two of three is a majority

    leader, term = cluster.wait_for_agreement("a leader once two nodes are up")
    assert leader in (survivor, stopped)
    assert term >= inflated.current_term

    cluster.nodes[old_leader].start()  # and the crashed one rejoins too
    _, final_term = cluster.wait_for_agreement("all three nodes to agree")
    assert final_term >= term
    assert sorted(cluster.states()) == [1, 2, 3]
    cluster.assert_election_safety()


# ---------------------------------------------------------------------------
# Whole-cluster shutdown
# ---------------------------------------------------------------------------


def test_every_node_shuts_down_gracefully_and_releases_its_port(
    cluster: ProcessCluster,
) -> None:
    cluster.start_all()
    cluster.wait_for_agreement("one agreed leader")

    codes = {i: n.stop_gracefully() for i, n in cluster.nodes.items()}

    assert codes == {1: 0, 2: 0, 3: 0}
    for node in cluster.nodes.values():
        assert "Raft node stopped" in node.log()
    for m in cluster.members:  # nothing is listening any more
        assert fetch_state(m.address, timeout=0.5) is None
    cluster.assert_election_safety()
