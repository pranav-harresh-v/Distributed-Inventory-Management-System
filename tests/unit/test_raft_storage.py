"""Tests for ``raft.storage``: the durable term and vote.

Every store here lives under pytest's ``tmp_path``. Nothing opens
``data/node1.db``; one test proves it by comparing that file's size and mtime
(and its sidecars') around a full save/load cycle, and others prove the store
refuses the inventory database outright.

The safety rules under test, in Raft's own words: a node's term never goes
backwards, and once it has voted in a term it can neither change nor withdraw
that vote. A restart must not forget either.
"""

from __future__ import annotations

import ast
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from raft import storage as storage_module
from raft.config import DATA_DIR
from raft.storage import (
    SCHEMA_VERSION,
    InvalidStateTransitionError,
    RaftState,
    RaftStateCorruptError,
    RaftStateError,
    RaftStateIOError,
    RaftStateMismatchError,
    RaftStateStore,
    TermRegressionError,
    VoteConflictError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "raft_node1.db"


@pytest.fixture
def store(state_path: Path) -> RaftStateStore:
    return RaftStateStore(state_path, node_id=1)


# --- a new node ------------------------------------------------------------
def test_a_missing_file_loads_as_term_zero_with_no_vote(store: RaftStateStore) -> None:
    assert store.load() == RaftState(current_term=0, voted_for=None)


def test_loading_never_creates_the_file(store: RaftStateStore, state_path: Path) -> None:
    store.load()
    store.load()
    assert not state_path.exists()


def test_constructing_a_store_creates_nothing(tmp_path: Path) -> None:
    RaftStateStore(tmp_path / "sub" / "dir" / "raft_node1.db", node_id=1)
    assert list(tmp_path.iterdir()) == []


def test_the_first_save_creates_the_file_and_its_folder_at_the_given_path(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state" / "raft_node2.db"
    RaftStateStore(target, node_id=2).save(1, None)

    assert target.is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["state"]


# --- saving and loading ----------------------------------------------------
def test_state_saves_and_loads(store: RaftStateStore) -> None:
    saved = store.save(4, 2)

    assert saved == RaftState(current_term=4, voted_for=2)
    assert store.load() == RaftState(current_term=4, voted_for=2)


def test_a_term_with_no_vote_is_stored_as_none(store: RaftStateStore) -> None:
    store.save(3, None)
    assert store.load() == RaftState(current_term=3, voted_for=None)


def test_term_zero_is_a_valid_state(store: RaftStateStore) -> None:
    store.save(0, None)
    assert store.load() == RaftState(0, None)


def test_a_higher_term_clears_the_previous_vote(store: RaftStateStore) -> None:
    store.save(2, 3)
    store.save(3, None)
    assert store.load() == RaftState(3, None)


def test_a_higher_term_may_carry_a_new_vote(store: RaftStateStore) -> None:
    store.save(2, 3)
    store.save(5, 1)
    assert store.load() == RaftState(5, 1)


def test_a_vote_may_be_cast_later_in_the_same_term(store: RaftStateStore) -> None:
    """Term first (no vote yet), then the vote: both in term 4."""
    store.save(4, None)
    store.save(4, 2)
    assert store.load() == RaftState(4, 2)


# --- the term never decreases ---------------------------------------------
def test_the_term_cannot_decrease(store: RaftStateStore, state_path: Path) -> None:
    store.save(5, 2)
    before = state_path.read_bytes()

    with pytest.raises(TermRegressionError, match="lower current_term from 5 to 4"):
        store.save(4, None)

    assert store.load() == RaftState(5, 2)
    assert state_path.read_bytes() == before


def test_a_regression_is_an_invalid_transition_and_a_state_error() -> None:
    assert issubclass(TermRegressionError, InvalidStateTransitionError)
    assert issubclass(InvalidStateTransitionError, RaftStateError)


def test_saving_the_same_state_again_is_harmless(store: RaftStateStore) -> None:
    store.save(6, 3)
    store.save(6, 3)
    assert store.load() == RaftState(6, 3)


# --- a cast vote is final for its term ------------------------------------
def test_a_vote_cannot_be_changed_within_a_term(
    store: RaftStateStore, state_path: Path
) -> None:
    store.save(4, 2)
    before = state_path.read_bytes()

    with pytest.raises(VoteConflictError, match="Already voted for node 2 in term 4"):
        store.save(4, 3)

    assert store.load() == RaftState(4, 2)
    assert state_path.read_bytes() == before


def test_a_vote_cannot_be_withdrawn_within_a_term(store: RaftStateStore) -> None:
    store.save(4, 2)
    with pytest.raises(VoteConflictError):
        store.save(4, None)
    assert store.load() == RaftState(4, 2)


# --- input validation ------------------------------------------------------
@pytest.mark.parametrize(
    ("term", "voted_for"),
    [
        (-1, None),
        (True, None),
        (1.5, None),
        ("2", None),
        (None, None),
        (1, 0),
        (1, -3),
        (1, True),
        (1, "2"),
        (1, 2.0),
    ],
)
def test_invalid_values_are_rejected_and_nothing_is_written(
    store: RaftStateStore, state_path: Path, term: object, voted_for: object
) -> None:
    with pytest.raises(ValueError):
        store.save(term, voted_for)  # type: ignore[arg-type]
    assert not state_path.exists()


@pytest.mark.parametrize("node_id", [0, -1, True, 1.0, "1", None])
def test_an_invalid_node_id_is_rejected(tmp_path: Path, node_id: object) -> None:
    with pytest.raises(ValueError):
        RaftStateStore(tmp_path / "x.db", node_id=node_id)  # type: ignore[arg-type]


# --- surviving a restart ---------------------------------------------------
def test_a_new_store_on_the_same_file_sees_the_saved_state(state_path: Path) -> None:
    RaftStateStore(state_path, 1).save(9, 3)

    reopened = RaftStateStore(state_path, 1)
    assert reopened.load() == RaftState(9, 3)


def test_a_vote_survives_a_restart_and_cannot_then_be_changed(state_path: Path) -> None:
    """The scenario the whole module exists for."""
    RaftStateStore(state_path, 1).save(7, 2)

    after_restart = RaftStateStore(state_path, 1)
    assert after_restart.load().voted_for == 2
    with pytest.raises(VoteConflictError):
        after_restart.save(7, 3)


def test_state_survives_a_real_process_restart(state_path: Path) -> None:
    """Written by one Python process, read back by another."""
    bootstrap = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from raft.storage import RaftStateStore; "
        "s = RaftStateStore(sys.argv[2], 2); "
    )
    write = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",  # -I ignores PYTHONDONTWRITEBYTECODE; -B keeps the repo free of .pyc
            "-c",
            bootstrap + "s.save(11, 3)",
            str(PROJECT_ROOT),
            str(state_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert write.returncode == 0, write.stderr

    read = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            bootstrap + "st = s.load(); print(st.current_term, st.voted_for)",
            str(PROJECT_ROOT),
            str(state_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert read.returncode == 0, read.stderr
    assert read.stdout.split() == ["11", "3"]


def test_two_stores_on_one_file_see_each_others_writes(state_path: Path) -> None:
    first = RaftStateStore(state_path, 1)
    second = RaftStateStore(state_path, 1)

    first.save(2, 1)
    assert second.load() == RaftState(2, 1)
    second.save(3, None)
    assert first.load() == RaftState(3, None)


def test_concurrent_saves_in_one_process_do_not_corrupt_the_file(
    store: RaftStateStore,
) -> None:
    """Different terms from many threads: each save is whole, the highest wins."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(term: int) -> None:
        barrier.wait()
        try:
            store.save(term, None)
        except TermRegressionError:
            pass  # a lower term arrived after a higher one: correctly refused
        except RaftStateError as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(term,)) for term in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert store.load().current_term == 8


# --- corruption is reported, never reset -----------------------------------
def _make_valid_file(state_path: Path) -> bytes:
    RaftStateStore(state_path, 1).save(5, 2)
    return state_path.read_bytes()


def test_a_file_that_is_not_a_database_is_reported_and_left_alone(
    store: RaftStateStore, state_path: Path
) -> None:
    garbage = b"this is definitely not a sqlite database\n" * 100
    state_path.write_bytes(garbage)

    with pytest.raises(RaftStateCorruptError, match="Refusing to reset"):
        store.load()
    with pytest.raises(RaftStateCorruptError):
        store.save(9, 1)

    assert state_path.read_bytes() == garbage


def test_a_truncated_database_is_reported_not_reset(
    store: RaftStateStore, state_path: Path
) -> None:
    valid = _make_valid_file(state_path)
    truncated = valid[:100]
    state_path.write_bytes(truncated)

    with pytest.raises(RaftStateError):
        store.load()
    with pytest.raises(RaftStateError):
        store.save(9, 1)

    assert state_path.read_bytes() == truncated


def test_a_database_that_is_not_a_raft_state_file_is_refused(
    store: RaftStateStore, state_path: Path
) -> None:
    """Pointing the store at some other SQLite database (say, an inventory
    one) must not write into it."""
    with sqlite3.connect(state_path) as conn:
        conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT)")
    before = state_path.read_bytes()

    with pytest.raises(RaftStateCorruptError, match="not a Raft state file"):
        store.load()
    with pytest.raises(RaftStateCorruptError, match="products"):
        store.save(1, None)

    assert state_path.read_bytes() == before


def test_a_missing_state_row_is_corruption_not_a_fresh_start(
    store: RaftStateStore, state_path: Path
) -> None:
    _make_valid_file(state_path)
    with sqlite3.connect(state_path) as conn:
        conn.execute("DELETE FROM raft_state")

    with pytest.raises(RaftStateCorruptError, match="exactly one Raft state row"):
        store.load()
    with pytest.raises(RaftStateCorruptError):
        store.save(99, None)


def test_an_unknown_schema_version_is_refused(
    store: RaftStateStore, state_path: Path
) -> None:
    _make_valid_file(state_path)
    with sqlite3.connect(state_path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(RaftStateCorruptError, match="schema version"):
        store.load()


def test_a_zero_byte_file_is_a_new_node_not_corruption(
    store: RaftStateStore, state_path: Path
) -> None:
    """e.g. a crash right after the file was created. It never held a vote."""
    state_path.write_bytes(b"")

    assert store.load() == RaftState(0, None)
    store.save(1, 2)
    assert store.load() == RaftState(1, 2)


def test_a_file_belonging_to_another_node_is_refused(state_path: Path) -> None:
    RaftStateStore(state_path, 1).save(3, 1)
    other = RaftStateStore(state_path, 2)

    with pytest.raises(RaftStateMismatchError, match="belongs to Raft node 1, not node 2"):
        other.load()
    with pytest.raises(RaftStateMismatchError):
        other.save(4, 2)
    assert RaftStateStore(state_path, 1).load() == RaftState(3, 1)


# --- I/O errors ------------------------------------------------------------
def test_a_locked_database_is_an_io_error_and_changes_nothing(
    store: RaftStateStore, state_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save(2, 1)
    monkeypatch.setattr(storage_module, "_BUSY_TIMEOUT_SECONDS", 0.1)

    blocker = sqlite3.connect(state_path, isolation_level=None)
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(RaftStateIOError, match="Could not write Raft state"):
            store.save(3, None)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert store.load() == RaftState(2, 1)


def test_a_folder_that_cannot_be_created_is_an_io_error(tmp_path: Path) -> None:
    store = RaftStateStore(tmp_path / "a" / "b" / "raft_node1.db", node_id=1)
    # Make the first folder a regular file after validation has passed.
    (tmp_path / "a").write_text("in the way")

    with pytest.raises(RaftStateIOError, match="Could not create the folder"):
        store.save(1, None)


def test_a_path_under_a_regular_file_is_refused_at_construction(tmp_path: Path) -> None:
    blocker = tmp_path / "plainfile"
    blocker.write_text("x")
    with pytest.raises(RaftStateError, match="not a folder"):
        RaftStateStore(blocker / "raft_node1.db", node_id=1)


# --- durability settings and layout ---------------------------------------
def test_the_file_uses_the_documented_layout(store: RaftStateStore, state_path: Path) -> None:
    store.save(3, 2)

    with sqlite3.connect(state_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        # Rollback journal, not WAL: one tiny row, and no sidecar files.
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        assert tables == ["raft_state"]
        assert conn.execute("SELECT id, node_id, current_term, voted_for FROM raft_state").fetchall() == [
            (1, 1, 3, 2)
        ]


def test_the_table_itself_refuses_invalid_values(store: RaftStateStore, state_path: Path) -> None:
    store.save(3, 2)
    with sqlite3.connect(state_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE raft_state SET current_term = -1")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE raft_state SET voted_for = 0")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO raft_state VALUES (2, 1, 0, NULL)")


def test_saving_leaves_no_sidecar_files_behind(store: RaftStateStore, state_path: Path) -> None:
    for term in range(1, 6):
        store.save(term, term)
    store.load()

    assert sorted(p.name for p in state_path.parent.iterdir()) == ["raft_node1.db"]


def test_the_connection_is_released_so_the_file_can_be_deleted(
    store: RaftStateStore, state_path: Path
) -> None:
    """On Windows an open handle would make this fail."""
    store.save(1, None)
    store.load()
    state_path.unlink()
    assert not state_path.exists()


# --- isolation from the inventory database --------------------------------
@pytest.mark.parametrize("name", ["node1.db", "node1.db-wal", "node1.db-shm"])
def test_the_store_refuses_the_inventory_database(name: str) -> None:
    with pytest.raises(RaftStateError, match="inventory database"):
        RaftStateStore(DATA_DIR / name, node_id=1)


def test_the_store_refuses_the_path_named_by_inventory_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = tmp_path / "inventory_copy.db"
    monkeypatch.setenv("INVENTORY_DB_PATH", str(inventory))
    with pytest.raises(RaftStateError, match="inventory database"):
        RaftStateStore(inventory, node_id=1)


def test_every_test_store_lives_under_tmp_path(
    store: RaftStateStore, tmp_path: Path
) -> None:
    assert tmp_path in store.path.parents
    assert store.path.name != "node1.db"


def test_a_save_and_load_cycle_leaves_the_inventory_database_alone(
    store: RaftStateStore,
) -> None:
    """Size and mtime of data/node1.db and its sidecars, before and after."""
    names = ("node1.db", "node1.db-wal", "node1.db-shm")

    def snapshot() -> dict[str, tuple[int, int] | None]:
        result: dict[str, tuple[int, int] | None] = {}
        for name in names:
            path = DATA_DIR / name
            if path.exists():
                info = path.stat()
                result[name] = (info.st_size, info.st_mtime_ns)
            else:
                result[name] = None
        return result

    before = snapshot()
    store.save(3, 1)
    store.load()
    assert snapshot() == before


def test_the_raft_package_imports_nothing_from_the_application() -> None:
    """Raft must start without the inventory app, its database or the LLM."""
    forbidden = {"app", "client", "llm", "sqlalchemy", "streamlit", "ollama", "httpx"}
    for source in sorted((PROJECT_ROOT / "raft").glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert roots.isdisjoint(forbidden), f"{source.name} imports {roots & forbidden}"


def test_the_storage_module_uses_the_standard_library_sqlite_driver() -> None:
    source = (PROJECT_ROOT / "raft" / "storage.py").read_text(encoding="utf-8")
    assert "import sqlite3" in source
    assert "app.storage" not in source
