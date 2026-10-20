"""Durable election state for one Raft node: ``current_term`` and ``voted_for``.

Why this file exists, in Raft terms. A node may vote for at most one candidate
per term. If it forgot its vote across a restart it could vote twice in the
same term, two candidates could each win a majority, and the cluster would have
two leaders. So these two values must reach disk *before* the node tells anyone
about them, and they must never silently revert. This module is that promise.

Design:

* **Its own SQLite file per node** (``data/raft_node<id>.db``), standard-library
  ``sqlite3`` only. It never touches the inventory database, its SQLAlchemy
  engine or its schema, and it refuses a path that is the inventory database.
* **One row, one transaction.** ``save`` writes term and vote together under
  ``BEGIN IMMEDIATE`` with ``PRAGMA synchronous=FULL``: SQLite syncs the
  journal and the database before ``COMMIT`` returns, so a crash leaves the old
  pair or the new pair, never a mixture. The rollback journal (not WAL) is used
  deliberately: one tiny row gains nothing from WAL, and it leaves no ``-wal``
  and ``-shm`` sidecar files behind.
* **A connection per call.** Nothing stays open between operations, so there is
  no handle to leak and the file can be removed by a test cleanup on Windows.
  A lock serialises calls inside one process; SQLite's own lock serialises
  processes.
* **Monotonic.** The term never decreases, and within one term a cast vote can
  be repeated but never changed or withdrawn. These are Raft's own rules,
  enforced a second time at the last line of defence.
* **Never resets quietly.** A missing file is a fresh node (term 0, no vote). A
  file that exists but cannot be understood raises
  :class:`RaftStateCorruptError`; resetting it would be exactly the amnesia
  described above. Recovery is a decision for a human, not for this module.

Nothing here writes to disk until :meth:`RaftStateStore.save` is called:
constructing a store and calling :meth:`RaftStateStore.load` are read-only, and
a fresh file is created only at the path the store was given.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from raft.config import ConfigError, validate_state_path

__all__ = [
    "InvalidStateTransitionError",
    "RaftState",
    "RaftStateCorruptError",
    "RaftStateError",
    "RaftStateIOError",
    "RaftStateMismatchError",
    "RaftStateStore",
    "TermRegressionError",
    "VoteConflictError",
]

#: Stored in ``PRAGMA user_version``. Bump when the table layout changes.
SCHEMA_VERSION: Final[int] = 1

_TABLE: Final[str] = "raft_state"
_BUSY_TIMEOUT_SECONDS: Final[float] = 5.0

_CREATE_TABLE: Final[str] = f"""
CREATE TABLE {_TABLE} (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    node_id      INTEGER NOT NULL CHECK (node_id >= 1),
    current_term INTEGER NOT NULL CHECK (current_term >= 0),
    voted_for    INTEGER          CHECK (voted_for IS NULL OR voted_for >= 1)
)
"""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class RaftStateError(Exception):
    """Base class for every failure of the Raft state store."""


class RaftStateIOError(RaftStateError):
    """The file could not be read or written (locked, read-only, disk error).

    The stored state is unchanged and may be fine: the operation can be
    retried or surfaced, but the node must not act as if it had been saved.
    """


class RaftStateCorruptError(RaftStateError):
    """The file exists but is not a valid Raft state file.

    Never repaired automatically. Starting a node with a reset term and vote
    could let it vote twice in one term.
    """


class RaftStateMismatchError(RaftStateError):
    """The file belongs to a different node id than the one opening it."""


class InvalidStateTransitionError(RaftStateError):
    """The requested change would break a Raft rule; nothing was written."""


class TermRegressionError(InvalidStateTransitionError):
    """A save tried to lower ``current_term``."""


class VoteConflictError(InvalidStateTransitionError):
    """A save tried to change or withdraw a vote already cast in this term."""


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RaftState:
    """The persistent election state of one node.

    ``voted_for`` is the candidate id voted for in ``current_term``, or ``None``
    if no vote has been cast in it.
    """

    current_term: int = 0
    voted_for: int | None = None


def _check_state(term: object, voted_for: object) -> None:
    # bool is an int subclass; refuse it so True cannot be stored as term 1.
    if isinstance(term, bool) or not isinstance(term, int) or term < 0:
        raise ValueError(f"current_term must be a non-negative integer, got {term!r}")
    if voted_for is not None and (
        isinstance(voted_for, bool) or not isinstance(voted_for, int) or voted_for < 1
    ):
        raise ValueError(f"voted_for must be None or a node id >= 1, got {voted_for!r}")


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------
class RaftStateStore:
    """Load and save one node's term and vote.

    ``node_id`` is recorded in the file on first save and checked on every
    access, so pointing node 2 at node 1's file fails loudly instead of letting
    two members share one vote record.
    """

    def __init__(self, path: str | os.PathLike[str], node_id: int) -> None:
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id < 1:
            raise ValueError(f"node_id must be an integer >= 1, got {node_id!r}")
        try:
            # Same rule as the config: never the inventory database, never a
            # directory. Reads only; creates nothing.
            self._path = validate_state_path(Path(path))
        except ConfigError as exc:
            raise RaftStateError(str(exc)) from exc
        self._node_id = node_id
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def node_id(self) -> int:
        return self._node_id

    # -- public operations --------------------------------------------------
    def load(self) -> RaftState:
        """Return the persisted state, or term 0 with no vote for a new node.

        Read-only: a missing file (or a database that holds no tables yet) is
        reported as the initial state without creating anything. Raises
        :class:`RaftStateCorruptError` for a file that exists but is not valid,
        :class:`RaftStateMismatchError` for another node's file, and
        :class:`RaftStateIOError` when it cannot be read.
        """
        with self._lock:
            if not self._path.exists():
                return RaftState()
            try:
                with closing(self._connect(write=False)) as conn:
                    state = self._read(conn)
            except sqlite3.Error as exc:
                raise self._translate(exc, "read") from exc
            return state if state is not None else RaftState()

    def save(self, term: int, voted_for: int | None) -> RaftState:
        """Durably record ``term`` and ``voted_for`` together; return them.

        The write is atomic and synced before this returns, so callers may
        reply to a vote request or start an election afterwards.

        Raises :class:`TermRegressionError` if ``term`` is lower than the
        stored term, :class:`VoteConflictError` if it would change or withdraw
        a vote already cast in the same term, and ``ValueError`` for values that
        are not a non-negative term and a node id >= 1 (or ``None``). In every
        error case nothing is written. Saving the stored values again is a
        harmless no-op write.
        """
        _check_state(term, voted_for)
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RaftStateIOError(
                    f"Could not create the folder for Raft state at {self._path}: {exc}"
                ) from exc
            try:
                with closing(self._connect(write=True)) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        current = self._read(conn)
                        if current is None:
                            self._initialise(conn, term, voted_for)
                        else:
                            self._check_transition(current, term, voted_for)
                            conn.execute(
                                f"UPDATE {_TABLE} SET current_term = ?, voted_for = ? "
                                "WHERE id = 1",
                                (term, voted_for),
                            )
                        conn.execute("COMMIT")
                    except BaseException:
                        # Includes the rule violations raised above: leave the
                        # file exactly as it was.
                        if conn.in_transaction:
                            conn.execute("ROLLBACK")
                        raise
            except sqlite3.Error as exc:
                raise self._translate(exc, "write") from exc
            return RaftState(current_term=term, voted_for=voted_for)

    # -- internals ----------------------------------------------------------
    def _connect(self, *, write: bool) -> sqlite3.Connection:
        # isolation_level=None: transactions are explicit (BEGIN IMMEDIATE /
        # COMMIT) rather than implied by the driver's guessing.
        #
        # Reads use an ordinary connection too, not mode=ro: if a crash left a
        # hot journal behind, SQLite must be allowed to roll it back before the
        # state can be read, and a read-only connection cannot.
        conn = sqlite3.connect(
            str(self._path), timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None
        )
        try:
            if write:
                conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
        except BaseException:
            conn.close()
            raise
        return conn

    def _read(self, conn: sqlite3.Connection) -> RaftState | None:
        """The stored state, or ``None`` if this is an empty database.

        Anything present but unexpected is corruption, not "empty".
        """
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if not tables:
            # A database with no tables never held a vote: nothing can be lost
            # by treating it as new (e.g. a crash right after file creation).
            return None
        if _TABLE not in tables:
            raise RaftStateCorruptError(
                f"{self._path} is a SQLite database but not a Raft state file "
                f"(tables: {', '.join(sorted(tables))}). Refusing to use it."
            )

        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise RaftStateCorruptError(
                f"{self._path} has schema version {version}, expected "
                f"{SCHEMA_VERSION}. Refusing to use it."
            )

        rows = conn.execute(
            f"SELECT id, node_id, current_term, voted_for FROM {_TABLE}"
        ).fetchall()
        if len(rows) != 1 or rows[0][0] != 1:
            raise RaftStateCorruptError(
                f"{self._path} should hold exactly one Raft state row, found "
                f"{len(rows)}. Refusing to reset it."
            )

        _, stored_node, term, voted_for = rows[0]
        if isinstance(stored_node, bool) or not isinstance(stored_node, int):
            raise RaftStateCorruptError(
                f"{self._path} holds an invalid node id ({stored_node!r}). "
                "Refusing to reset it."
            )
        try:
            _check_state(term, voted_for)
        except ValueError as exc:
            raise RaftStateCorruptError(
                f"{self._path} holds invalid Raft state ({exc}). Refusing to reset it."
            ) from exc

        if stored_node != self._node_id:
            raise RaftStateMismatchError(
                f"{self._path} belongs to Raft node {stored_node}, not node "
                f"{self._node_id}. Each node needs its own state file."
            )
        return RaftState(current_term=term, voted_for=voted_for)

    def _initialise(
        self, conn: sqlite3.Connection, term: int, voted_for: int | None
    ) -> None:
        conn.execute(_CREATE_TABLE)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute(
            f"INSERT INTO {_TABLE} (id, node_id, current_term, voted_for) "
            "VALUES (1, ?, ?, ?)",
            (self._node_id, term, voted_for),
        )

    @staticmethod
    def _check_transition(current: RaftState, term: int, voted_for: int | None) -> None:
        if term < current.current_term:
            raise TermRegressionError(
                f"Refusing to lower current_term from {current.current_term} to {term}"
            )
        if (
            term == current.current_term
            and current.voted_for is not None
            and voted_for != current.voted_for
        ):
            raise VoteConflictError(
                f"Already voted for node {current.voted_for} in term {term}; "
                f"cannot change it to {voted_for!r}"
            )

    def _translate(self, exc: sqlite3.Error, action: str) -> RaftStateError:
        """Classify a sqlite3 failure as I/O trouble or as a bad file."""
        # OperationalError covers lock timeouts, a read-only file, disk I/O
        # errors: the file may be fine. Other DatabaseErrors ("file is not a
        # database", "malformed") mean the content itself is unusable.
        if isinstance(exc, sqlite3.OperationalError):
            return RaftStateIOError(
                f"Could not {action} Raft state at {self._path}: {exc}"
            )
        if isinstance(exc, sqlite3.DatabaseError):
            return RaftStateCorruptError(
                f"{self._path} is not a usable Raft state file ({exc}). "
                "Refusing to reset it."
            )
        return RaftStateIOError(f"Could not {action} Raft state at {self._path}: {exc}")
