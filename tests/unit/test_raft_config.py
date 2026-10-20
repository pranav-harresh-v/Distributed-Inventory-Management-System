"""Tests for ``raft.config``.

Every test passes an explicit ``env`` mapping (or builds a ``RaftConfig``
directly), so none of them reads or changes ``os.environ``. State paths live
under ``tmp_path``; loading a config creates nothing on disk, and one test
proves it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raft.config import (
    DATA_DIR,
    DEFAULT_ELECTION_TIMEOUT_MAX_MS,
    DEFAULT_ELECTION_TIMEOUT_MIN_MS,
    DEFAULT_HEARTBEAT_MS,
    ConfigError,
    Member,
    RaftConfig,
    default_state_path,
    load_config,
    parse_peers,
    reserved_ports,
)

LOCAL_PEERS = "1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103"


def make_env(tmp_path: Path, **overrides: str | None) -> dict[str, str]:
    """A valid environment for node 1; ``None`` removes a key."""
    env: dict[str, str] = {
        "RAFT_NODE_ID": "1",
        "RAFT_STATE_PATH": str(tmp_path / "raft_node1.db"),
    }
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def make_config(tmp_path: Path, **overrides: object) -> RaftConfig:
    """A valid direct-built config; ``reserved={}`` switches the port check off."""
    values: dict[str, object] = {
        "node_id": 1,
        "members": parse_peers(LOCAL_PEERS),
        "host": "127.0.0.1",
        "port": 50101,
        "state_path": tmp_path / "raft_node1.db",
        "reserved": {},
    }
    values.update(overrides)
    return RaftConfig(**values)  # type: ignore[arg-type]


# --- a valid configuration loads -------------------------------------------
def test_a_minimal_environment_loads_with_local_defaults(tmp_path: Path) -> None:
    config = load_config(make_env(tmp_path))

    assert config.node_id == 1
    assert [m.node_id for m in config.members] == [1, 2, 3]
    assert (config.host, config.port) == ("127.0.0.1", 50101)
    assert config.heartbeat_ms == DEFAULT_HEARTBEAT_MS == 100
    assert config.election_timeout_min_ms == DEFAULT_ELECTION_TIMEOUT_MIN_MS == 500
    assert config.election_timeout_max_ms == DEFAULT_ELECTION_TIMEOUT_MAX_MS == 1000
    assert config.state_path == (tmp_path / "raft_node1.db").resolve()


def test_every_variable_can_be_set(tmp_path: Path) -> None:
    env = make_env(
        tmp_path,
        RAFT_NODE_ID="2",
        RAFT_PEERS="1=127.0.0.1:6001,2=127.0.0.1:6002,3=127.0.0.1:6003",
        RAFT_HOST="127.0.0.1",
        RAFT_PORT="6002",
        RAFT_STATE_PATH=str(tmp_path / "custom.db"),
        RAFT_HEARTBEAT_MS="50",
        RAFT_ELECTION_TIMEOUT_MIN_MS="300",
        RAFT_ELECTION_TIMEOUT_MAX_MS="600",
    )
    config = load_config(env)

    assert config.node_id == 2
    assert config.listen_address == "127.0.0.1:6002"
    assert (config.heartbeat_ms, config.election_timeout_min_ms) == (50, 300)
    assert config.election_timeout_max_ms == 600
    assert config.state_path == (tmp_path / "custom.db").resolve()


def test_listen_address_defaults_to_this_nodes_membership_entry(tmp_path: Path) -> None:
    config = load_config(make_env(tmp_path, RAFT_NODE_ID="3"))
    assert config.listen_address == "127.0.0.1:50103"


def test_the_default_state_file_is_data_raft_node_id_db() -> None:
    assert default_state_path(2) == DATA_DIR / "raft_node2.db"
    assert default_state_path(2).name == "raft_node2.db"
    assert default_state_path(2).name != "node1.db"


def test_the_default_state_path_is_used_when_none_is_given() -> None:
    config = load_config({"RAFT_NODE_ID": "3"})
    assert config.state_path == DATA_DIR / "raft_node3.db"


def test_blank_variables_count_as_unset(tmp_path: Path) -> None:
    config = load_config(make_env(tmp_path, RAFT_PEERS="  ", RAFT_HEARTBEAT_MS=""))
    assert config.cluster_size == 3
    assert config.heartbeat_ms == 100


def test_membership_views(tmp_path: Path) -> None:
    config = load_config(make_env(tmp_path, RAFT_NODE_ID="2"))

    assert config.cluster_size == 3
    assert config.quorum == 2
    assert config.self_member == Member(2, "127.0.0.1", 50102)
    assert [m.node_id for m in config.peers] == [1, 3]
    assert config.self_member.address == "127.0.0.1:50102"


@pytest.mark.parametrize(("size", "quorum"), [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)])
def test_quorum_is_a_strict_majority(tmp_path: Path, size: int, quorum: int) -> None:
    members = tuple(Member(i, "127.0.0.1", 6000 + i) for i in range(1, size + 1))
    config = make_config(tmp_path, members=members, port=6001)
    assert config.quorum == quorum


def test_loading_reads_only_the_mapping_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAFT_NODE_ID", "9")
    monkeypatch.setenv("RAFT_PEERS", "9=127.0.0.1:7000")
    assert load_config(make_env(tmp_path)).node_id == 1


def test_loading_creates_nothing_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    config = load_config(make_env(tmp_path, RAFT_STATE_PATH=str(tmp_path / "a" / "b" / "r.db")))

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not config.state_path.exists()
    assert not config.state_path.parent.exists()


# --- required values -------------------------------------------------------
def test_the_node_id_is_required(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="RAFT_NODE_ID is required"):
        load_config(make_env(tmp_path, RAFT_NODE_ID=None))


@pytest.mark.parametrize("value", ["abc", "1.5", "0", "-1"])
def test_a_bad_node_id_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError, match="RAFT_NODE_ID"):
        load_config(make_env(tmp_path, RAFT_NODE_ID=value))


def test_this_node_must_be_a_member(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not in RAFT_PEERS"):
        load_config(make_env(tmp_path, RAFT_NODE_ID="4"))


def test_a_direct_config_must_also_contain_its_own_node(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not in the Raft membership"):
        make_config(tmp_path, node_id=7)


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_a_non_integer_node_id_is_rejected_when_built_directly(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, node_id=value)


# --- membership ------------------------------------------------------------
def test_peers_are_parsed_and_sorted_by_id() -> None:
    members = parse_peers("3=127.0.0.1:3, 1 = localhost:1 ,2=10.0.0.2:2")
    assert [(m.node_id, m.host, m.port) for m in members] == [
        (1, "localhost", 1),
        (2, "10.0.0.2", 2),
        (3, "127.0.0.1", 3),
    ]


def test_duplicate_member_ids_are_rejected() -> None:
    with pytest.raises(ConfigError, match="Duplicate Raft member id 2"):
        parse_peers("1=127.0.0.1:6001,2=127.0.0.1:6002,2=127.0.0.1:6003")


def test_duplicate_member_addresses_are_rejected() -> None:
    with pytest.raises(ConfigError, match="Duplicate Raft member address"):
        parse_peers("1=127.0.0.1:6001,2=127.0.0.1:6001")


def test_duplicate_addresses_are_compared_case_insensitively() -> None:
    with pytest.raises(ConfigError, match="Duplicate Raft member address"):
        parse_peers("1=LocalHost:6001,2=localhost:6001")


def test_the_same_host_on_different_ports_is_fine() -> None:
    assert len(parse_peers("1=127.0.0.1:6001,2=127.0.0.1:6002")) == 2


def test_the_same_port_on_different_hosts_is_fine() -> None:
    assert len(parse_peers("1=10.0.0.1:6001,2=10.0.0.2:6001")) == 2


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "1",
        "1=",
        "1=127.0.0.1",
        "1=127.0.0.1:abc",
        "a=127.0.0.1:6001",
        "-1=127.0.0.1:6001",
        "0=127.0.0.1:6001",
        "1=127.0.0.1:0",
        "1=127.0.0.1:65536",
        "1=:6001",
        "1=127.0.0.1:6001,,2=127.0.0.1:6002",
        "1=http://127.0.0.1:6001",
        "1=host name:6001",
    ],
)
def test_malformed_peer_lists_are_rejected(raw: str) -> None:
    with pytest.raises(ConfigError):
        parse_peers(raw)


# --- the listen address ----------------------------------------------------
def test_the_listen_port_must_match_the_membership_entry(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not match"):
        load_config(make_env(tmp_path, RAFT_PORT="50999"))


def test_the_listen_host_must_match_the_membership_entry(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not match"):
        load_config(make_env(tmp_path, RAFT_HOST="10.9.9.9"))


def test_a_wildcard_bind_may_differ_in_host_but_not_port(tmp_path: Path) -> None:
    config = load_config(make_env(tmp_path, RAFT_HOST="0.0.0.0"))
    assert config.host == "0.0.0.0"
    assert config.port == 50101
    with pytest.raises(ConfigError, match="does not match"):
        load_config(make_env(tmp_path, RAFT_HOST="0.0.0.0", RAFT_PORT="50102"))


def test_host_comparison_ignores_case(tmp_path: Path) -> None:
    members = parse_peers("1=localhost:6001,2=localhost:6002")
    config = make_config(tmp_path, members=members, host="LOCALHOST", port=6001)
    assert config.host == "LOCALHOST"


@pytest.mark.parametrize("value", ["0", "65536", "-5", "abc", "50101.5"])
def test_an_invalid_listen_port_is_rejected(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError):
        load_config(make_env(tmp_path, RAFT_PORT=value))


# --- timeouts --------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAFT_HEARTBEAT_MS", "0"),
        ("RAFT_HEARTBEAT_MS", "-10"),
        ("RAFT_HEARTBEAT_MS", "fast"),
        ("RAFT_ELECTION_TIMEOUT_MIN_MS", "0"),
        ("RAFT_ELECTION_TIMEOUT_MIN_MS", "-1"),
        ("RAFT_ELECTION_TIMEOUT_MAX_MS", "0"),
        ("RAFT_ELECTION_TIMEOUT_MAX_MS", "soon"),
        ("RAFT_ELECTION_TIMEOUT_MAX_MS", "60001"),
    ],
)
def test_non_positive_or_non_numeric_timeouts_are_rejected(
    tmp_path: Path, name: str, value: str
) -> None:
    with pytest.raises(ConfigError):
        load_config(make_env(tmp_path, **{name: value}))


def test_the_maximum_timeout_must_exceed_the_minimum(tmp_path: Path) -> None:
    equal = make_env(
        tmp_path, RAFT_ELECTION_TIMEOUT_MIN_MS="600", RAFT_ELECTION_TIMEOUT_MAX_MS="600"
    )
    with pytest.raises(ConfigError, match="must be greater"):
        load_config(equal)

    smaller = make_env(
        tmp_path, RAFT_ELECTION_TIMEOUT_MIN_MS="900", RAFT_ELECTION_TIMEOUT_MAX_MS="600"
    )
    with pytest.raises(ConfigError, match="must be greater"):
        load_config(smaller)


def test_the_election_timeout_must_be_well_above_the_heartbeat(tmp_path: Path) -> None:
    # 5 x 100 ms = 500 ms is the floor.
    ok = make_env(tmp_path, RAFT_ELECTION_TIMEOUT_MIN_MS="500")
    assert load_config(ok).election_timeout_min_ms == 500

    too_close = make_env(tmp_path, RAFT_ELECTION_TIMEOUT_MIN_MS="499")
    with pytest.raises(ConfigError, match="at least 5x heartbeat_ms"):
        load_config(too_close)

    slow_heartbeat = make_env(tmp_path, RAFT_HEARTBEAT_MS="200")
    with pytest.raises(ConfigError, match="at least 5x heartbeat_ms"):
        load_config(slow_heartbeat)


@pytest.mark.parametrize("value", [True, 1.5, "100", None])
def test_non_integer_timeouts_are_rejected_when_built_directly(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(ConfigError):
        make_config(tmp_path, heartbeat_ms=value)


# --- port conflicts --------------------------------------------------------
@pytest.mark.parametrize(
    ("port", "owner"),
    [
        (50051, "application gRPC"),
        (50052, "LLM gRPC"),
        (11434, "Ollama"),
        (8501, "Streamlit"),
        (8502, "Streamlit"),
    ],
)
def test_ports_used_by_the_rest_of_the_system_are_rejected(
    tmp_path: Path, port: int, owner: str
) -> None:
    peers = f"1=127.0.0.1:{port},2=127.0.0.1:50102,3=127.0.0.1:50103"
    with pytest.raises(ConfigError, match=owner):
        load_config(make_env(tmp_path, RAFT_PEERS=peers))


def test_a_conflict_with_another_members_port_is_also_rejected(tmp_path: Path) -> None:
    peers = "1=127.0.0.1:50101,2=127.0.0.1:50052,3=127.0.0.1:50103"
    with pytest.raises(ConfigError, match="LLM gRPC"):
        load_config(make_env(tmp_path, RAFT_PEERS=peers))


def test_ports_moved_by_the_environment_are_reserved_too(tmp_path: Path) -> None:
    env = make_env(
        tmp_path,
        INVENTORY_GRPC_PORT="50999",
        RAFT_PEERS="1=127.0.0.1:50999,2=127.0.0.1:50102,3=127.0.0.1:50103",
    )
    assert 50999 in reserved_ports(env)
    with pytest.raises(ConfigError, match="INVENTORY_GRPC_PORT"):
        load_config(env)


def test_the_default_raft_ports_clash_with_nothing() -> None:
    assert {50101, 50102, 50103}.isdisjoint(reserved_ports({}))


# --- state path ------------------------------------------------------------
@pytest.mark.parametrize("name", ["node1.db", "node1.db-wal", "node1.db-shm"])
def test_the_inventory_database_and_its_sidecars_are_refused(
    tmp_path: Path, name: str
) -> None:
    env = make_env(tmp_path, RAFT_STATE_PATH=str(DATA_DIR / name))
    with pytest.raises(ConfigError, match="inventory database"):
        load_config(env)


def test_the_path_named_by_inventory_db_path_is_refused(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory_copy.db"
    env = make_env(
        tmp_path, INVENTORY_DB_PATH=str(inventory), RAFT_STATE_PATH=str(inventory)
    )
    with pytest.raises(ConfigError, match="inventory database"):
        load_config(env)


def test_a_directory_is_not_a_state_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is a directory"):
        load_config(make_env(tmp_path, RAFT_STATE_PATH=str(tmp_path)))


def test_a_state_path_beneath_a_regular_file_is_rejected(tmp_path: Path) -> None:
    blocker = tmp_path / "plainfile"
    blocker.write_text("x")
    env = make_env(tmp_path, RAFT_STATE_PATH=str(blocker / "raft.db"))
    with pytest.raises(ConfigError, match="not a folder"):
        load_config(env)


def test_a_state_path_in_a_folder_that_does_not_exist_yet_is_allowed(
    tmp_path: Path,
) -> None:
    target = tmp_path / "deep" / "er" / "raft_node1.db"
    config = load_config(make_env(tmp_path, RAFT_STATE_PATH=str(target)))
    assert config.state_path == target.resolve()
    assert not target.parent.exists()
