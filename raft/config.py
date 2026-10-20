"""Validated configuration for one Raft node.

Everything is read from environment variables, following ``app/server.py`` and
``llm/server.py``, but through an injectable mapping: :func:`load_config` takes
``env`` so tests never touch ``os.environ``, and :class:`RaftConfig` validates
itself on construction so a bad hand-built config fails exactly like a bad
environment.

Variables::

    RAFT_NODE_ID                  this node's member id (required, >= 1)
    RAFT_PEERS                    the whole membership, this node included:
                                  1=127.0.0.1:50101,2=127.0.0.1:50102,...
    RAFT_HOST / RAFT_PORT         where this node listens (default: its own
                                  entry in RAFT_PEERS)
    RAFT_STATE_PATH               this node's election-state SQLite file
                                  (default: data/raft_node<id>.db)
    RAFT_HEARTBEAT_MS             default 100
    RAFT_ELECTION_TIMEOUT_MIN_MS  default 500
    RAFT_ELECTION_TIMEOUT_MAX_MS  default 1000

Loading has no side effects: it creates no file and no directory, opens no
socket and starts no thread.

The ``RAFT_`` prefix is new. Existing settings use ``INVENTORY_`` (application
server) and ``LLM_`` (LLM node); Raft is a third, separate concern.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_ELECTION_TIMEOUT_MAX_MS",
    "DEFAULT_ELECTION_TIMEOUT_MIN_MS",
    "DEFAULT_HEARTBEAT_MS",
    "DEFAULT_PEERS",
    "ConfigError",
    "Member",
    "RaftConfig",
    "default_state_path",
    "inventory_database_paths",
    "load_config",
    "parse_peers",
    "reserved_ports",
]

# raft/config.py -> raft -> <project root>. Computed here rather than
# imported from app.storage.database, which would drag the application's
# SQLAlchemy engine into a package that must not depend on it.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"

NODE_ID_ENV_VAR: Final[str] = "RAFT_NODE_ID"
PEERS_ENV_VAR: Final[str] = "RAFT_PEERS"
HOST_ENV_VAR: Final[str] = "RAFT_HOST"
PORT_ENV_VAR: Final[str] = "RAFT_PORT"
STATE_PATH_ENV_VAR: Final[str] = "RAFT_STATE_PATH"
HEARTBEAT_ENV_VAR: Final[str] = "RAFT_HEARTBEAT_MS"
ELECTION_MIN_ENV_VAR: Final[str] = "RAFT_ELECTION_TIMEOUT_MIN_MS"
ELECTION_MAX_ENV_VAR: Final[str] = "RAFT_ELECTION_TIMEOUT_MAX_MS"

DEFAULT_HEARTBEAT_MS: Final[int] = 100
DEFAULT_ELECTION_TIMEOUT_MIN_MS: Final[int] = 500
DEFAULT_ELECTION_TIMEOUT_MAX_MS: Final[int] = 1000

#: The local development cluster. A default rather than a requirement so a
#: developer only has to say which member this process is.
DEFAULT_PEERS: Final[str] = (
    "1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103"
)

#: Smallest election timeout, as a multiple of the heartbeat interval. Raft
#: needs the timeout to be well above the heartbeat, or a single delayed
#: heartbeat (a GC pause, a Windows timer tick of ~15 ms) starts an election.
MIN_ELECTION_TO_HEARTBEAT_RATIO: Final[int] = 5

#: Sanity ceiling for any timeout, one minute, so a typo such as 10000000
#: fails loudly instead of producing a cluster that never elects.
MAX_TIMEOUT_MS: Final[int] = 60_000

#: Binding to these listens on every interface, so the bind host legitimately
#: differs from the address other members dial.
WILDCARD_HOSTS: Final[frozenset[str]] = frozenset({"0.0.0.0"})

#: Ports the rest of the system listens on by default. Raft must never use one:
#: a collision would only surface as a confusing bind failure at start-up.
_FIXED_RESERVED_PORTS: Final[dict[int, str]] = {
    50051: "application gRPC server",
    50052: "LLM gRPC server",
    11434: "Ollama",
    8501: "admin Streamlit client",
    8502: "user Streamlit client",
}
_RESERVED_PORT_ENV_VARS: Final[dict[str, str]] = {
    "INVENTORY_GRPC_PORT": "application gRPC server (INVENTORY_GRPC_PORT)",
    "LLM_GRPC_PORT": "LLM gRPC server (LLM_GRPC_PORT)",
}

#: The inventory database's env override, defined by app/storage/database.py.
_INVENTORY_DB_ENV_VAR: Final[str] = "INVENTORY_DB_PATH"
_INVENTORY_DB_DEFAULT: Final[Path] = DATA_DIR / "node1.db"
#: SQLite creates these beside a database in WAL mode.
_SQLITE_SIDECAR_SUFFIXES: Final[tuple[str, ...]] = ("-wal", "-shm", "-journal")


class ConfigError(ValueError):
    """The Raft configuration is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Member:
    """One Raft cluster member and the address others dial to reach it."""

    node_id: int
    host: str
    port: int

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


# --------------------------------------------------------------------------
# Small validators
# --------------------------------------------------------------------------
def _require_int(value: object, name: str) -> int:
    # bool is an int subclass; True would otherwise pass as a node id of 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    return value


def _validate_node_id(value: object, name: str = "node id") -> int:
    number = _require_int(value, name)
    if number < 1:
        # 0 is the protobuf default and means "no node" on the wire.
        raise ConfigError(f"{name} must be at least 1, got {number}")
    return number


def _validate_host(value: object, name: str = "host") -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string, got {value!r}")
    host = value.strip()
    if not host:
        raise ConfigError(f"{name} must not be empty")
    if any(ch.isspace() for ch in host) or any(ch in host for ch in ":/@"):
        raise ConfigError(
            f"{name} {host!r} is not a plain hostname or IPv4 address "
            "(IPv6 and URLs are not supported)"
        )
    return host


def _validate_port(value: object, name: str = "port") -> int:
    number = _require_int(value, name)
    if not 1 <= number <= 65535:
        raise ConfigError(f"{name} must be between 1 and 65535, got {number}")
    return number


def _validate_timeout(value: object, name: str) -> int:
    number = _require_int(value, name)
    if number <= 0:
        raise ConfigError(f"{name} must be positive, got {number}")
    if number > MAX_TIMEOUT_MS:
        raise ConfigError(f"{name} must not exceed {MAX_TIMEOUT_MS} ms, got {number}")
    return number


def _address_key(host: str, port: int) -> tuple[str, int]:
    """Hostnames are case-insensitive, so compare them lower-cased."""
    return (host.lower(), port)


def reserved_ports(env: Mapping[str, str] | None = None) -> dict[int, str]:
    """Ports Raft may not use, mapped to the service that owns each.

    Includes the defaults of the application server, LLM node, Ollama and the
    two Streamlit clients, plus any port the environment moves the first two to.
    """
    source = os.environ if env is None else env
    reserved = dict(_FIXED_RESERVED_PORTS)
    for variable, owner in _RESERVED_PORT_ENV_VARS.items():
        raw = (source.get(variable) or "").strip()
        if raw.isdigit():
            reserved.setdefault(int(raw), owner)
    return reserved


def inventory_database_paths(env: Mapping[str, str] | None = None) -> set[str]:
    """Normalised paths that belong to the inventory database.

    The default ``data/node1.db`` and whatever ``INVENTORY_DB_PATH`` points at,
    each with its SQLite sidecar files. A Raft state file must never be one of
    them: Raft would write its tables into the inventory database.
    """
    source = os.environ if env is None else env
    bases = [_INVENTORY_DB_DEFAULT]
    override = (source.get(_INVENTORY_DB_ENV_VAR) or "").strip()
    if override:
        bases.append(Path(override).expanduser().resolve())

    forbidden: set[str] = set()
    for base in bases:
        forbidden.add(os.path.normcase(str(base)))
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            forbidden.add(os.path.normcase(f"{base}{suffix}"))
    return forbidden


def default_state_path(node_id: int) -> Path:
    """``data/raft_node<id>.db``: flat under ``data/`` so the existing
    ``data/*.db`` ignore rules cover it (a subfolder would not match them)."""
    return DATA_DIR / f"raft_node{_validate_node_id(node_id)}.db"


def validate_state_path(
    path: Path, env: Mapping[str, str] | None = None
) -> Path:
    """Check a state-file location and return it resolved.

    Reads only; never creates the file or its folder (the store does that, at
    this exact path, on first save).
    """
    resolved = Path(path).expanduser().resolve()
    if os.path.normcase(str(resolved)) in inventory_database_paths(env):
        raise ConfigError(
            f"Raft state path {resolved} is the inventory database. Raft must "
            "use its own file, for example data/raft_node<id>.db."
        )
    if resolved.exists() and resolved.is_dir():
        raise ConfigError(f"Raft state path {resolved} is a directory, not a file")

    # The folder must exist, or be creatable beneath one that does.
    ancestor = resolved.parent
    while not ancestor.exists():
        if ancestor.parent == ancestor:
            raise ConfigError(f"Raft state path {resolved} has no usable parent folder")
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ConfigError(
            f"Raft state path {resolved}: {ancestor} exists but is not a folder"
        )
    return resolved


# --------------------------------------------------------------------------
# Membership
# --------------------------------------------------------------------------
def parse_peers(raw: str) -> tuple[Member, ...]:
    """Parse ``1=host:port,2=host:port,...`` into members, sorted by id.

    Rejects malformed entries, duplicate ids and duplicate addresses.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{PEERS_ENV_VAR} must not be empty")

    members: list[Member] = []
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            raise ConfigError(f"{PEERS_ENV_VAR} contains an empty entry: {raw!r}")
        node_part, separator, address = entry.partition("=")
        if not separator:
            raise ConfigError(
                f"{PEERS_ENV_VAR} entry {entry!r} must look like 1=127.0.0.1:50101"
            )
        node_text = node_part.strip()
        if not node_text.isdigit():
            raise ConfigError(
                f"{PEERS_ENV_VAR} entry {entry!r}: node id must be a whole number"
            )
        host_part, colon, port_text = address.strip().rpartition(":")
        if not colon or not port_text.strip().isdigit():
            raise ConfigError(
                f"{PEERS_ENV_VAR} entry {entry!r}: address must be host:port"
            )
        members.append(
            Member(
                node_id=_validate_node_id(int(node_text), f"{PEERS_ENV_VAR} node id"),
                host=_validate_host(host_part, f"{PEERS_ENV_VAR} host"),
                port=_validate_port(int(port_text.strip()), f"{PEERS_ENV_VAR} port"),
            )
        )
    return _validated_members(members)


def _validated_members(members: list[Member] | tuple[Member, ...]) -> tuple[Member, ...]:
    if not members:
        raise ConfigError("Raft membership must contain at least one member")

    ids: set[int] = set()
    addresses: set[tuple[str, int]] = set()
    for member in members:
        _validate_node_id(member.node_id, "member node id")
        _validate_host(member.host, f"member {member.node_id} host")
        _validate_port(member.port, f"member {member.node_id} port")
        if member.node_id in ids:
            raise ConfigError(f"Duplicate Raft member id {member.node_id}")
        key = _address_key(member.host, member.port)
        if key in addresses:
            raise ConfigError(f"Duplicate Raft member address {member.address}")
        ids.add(member.node_id)
        addresses.add(key)
    return tuple(sorted(members, key=lambda m: m.node_id))


# --------------------------------------------------------------------------
# The configuration object
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RaftConfig:
    """Everything one Raft node needs to know before it starts.

    Validated in ``__post_init__``: if an instance exists, it is consistent.
    ``reserved`` lets a test supply the ports to avoid; by default they are
    derived from the process environment (see :func:`reserved_ports`).
    """

    node_id: int
    members: tuple[Member, ...]
    host: str
    port: int
    state_path: Path
    heartbeat_ms: int = DEFAULT_HEARTBEAT_MS
    election_timeout_min_ms: int = DEFAULT_ELECTION_TIMEOUT_MIN_MS
    election_timeout_max_ms: int = DEFAULT_ELECTION_TIMEOUT_MAX_MS
    reserved: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        node_id = _validate_node_id(self.node_id, "node id")
        members = _validated_members(tuple(self.members))
        object.__setattr__(self, "members", members)

        mine = next((m for m in members if m.node_id == node_id), None)
        if mine is None:
            known = ", ".join(str(m.node_id) for m in members)
            raise ConfigError(
                f"This node (id {node_id}) is not in the Raft membership (ids: {known})"
            )

        host = _validate_host(self.host, "listen host")
        port = _validate_port(self.port, "listen port")
        # The listen address must be the member entry others dial. A wildcard
        # bind may differ in host (0.0.0.0 serves every interface) but never
        # in port.
        if port != mine.port:
            raise ConfigError(
                f"Listen port {port} does not match this node's membership entry "
                f"{mine.address}"
            )
        if host.lower() != mine.host.lower() and host not in WILDCARD_HOSTS:
            raise ConfigError(
                f"Listen host {host!r} does not match this node's membership entry "
                f"{mine.address}"
            )

        heartbeat = _validate_timeout(self.heartbeat_ms, "heartbeat_ms")
        low = _validate_timeout(self.election_timeout_min_ms, "election_timeout_min_ms")
        high = _validate_timeout(self.election_timeout_max_ms, "election_timeout_max_ms")
        if high <= low:
            raise ConfigError(
                f"election_timeout_max_ms ({high}) must be greater than "
                f"election_timeout_min_ms ({low})"
            )
        if low < heartbeat * MIN_ELECTION_TO_HEARTBEAT_RATIO:
            raise ConfigError(
                f"election_timeout_min_ms ({low}) must be at least "
                f"{MIN_ELECTION_TO_HEARTBEAT_RATIO}x heartbeat_ms ({heartbeat}), "
                f"i.e. {heartbeat * MIN_ELECTION_TO_HEARTBEAT_RATIO} ms"
            )

        reserved = dict(self.reserved) if self.reserved is not None else reserved_ports()
        for member in members:
            if member.port in reserved:
                raise ConfigError(
                    f"Raft member {member.node_id} uses port {member.port}, which "
                    f"belongs to the {reserved[member.port]}"
                )

        object.__setattr__(self, "state_path", validate_state_path(Path(self.state_path)))

    # -- derived views ------------------------------------------------------
    @property
    def self_member(self) -> Member:
        return next(m for m in self.members if m.node_id == self.node_id)

    @property
    def peers(self) -> tuple[Member, ...]:
        """Every member except this node."""
        return tuple(m for m in self.members if m.node_id != self.node_id)

    @property
    def cluster_size(self) -> int:
        return len(self.members)

    @property
    def quorum(self) -> int:
        """Votes (counting your own) needed for a majority."""
        return self.cluster_size // 2 + 1

    @property
    def listen_address(self) -> str:
        return f"{self.host}:{self.port}"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _text(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _env_int(env: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = _text(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def load_config(env: Mapping[str, str] | None = None) -> RaftConfig:
    """Build and validate a :class:`RaftConfig` from environment variables.

    ``env`` defaults to ``os.environ``; tests pass a plain dict. Raises
    :class:`ConfigError` with a message naming the offending variable.
    """
    source = os.environ if env is None else env

    node_id = _env_int(source, NODE_ID_ENV_VAR, None)
    if node_id is None:
        raise ConfigError(f"{NODE_ID_ENV_VAR} is required")
    node_id = _validate_node_id(node_id, NODE_ID_ENV_VAR)

    members = parse_peers(_text(source, PEERS_ENV_VAR) or DEFAULT_PEERS)
    mine = next((m for m in members if m.node_id == node_id), None)
    if mine is None:
        known = ", ".join(str(m.node_id) for m in members)
        raise ConfigError(
            f"{NODE_ID_ENV_VAR}={node_id} is not in {PEERS_ENV_VAR} (ids: {known})"
        )

    host = _text(source, HOST_ENV_VAR) or mine.host
    port = _env_int(source, PORT_ENV_VAR, mine.port)
    state_text = _text(source, STATE_PATH_ENV_VAR)

    state_path = Path(state_text) if state_text else default_state_path(node_id)
    # Checked against the injected env as well, so a test (or a caller with a
    # custom environment) is protected from *its* INVENTORY_DB_PATH too.
    validate_state_path(state_path, source)

    return RaftConfig(
        node_id=node_id,
        members=members,
        host=host,
        port=port,
        state_path=state_path,
        heartbeat_ms=_env_int(source, HEARTBEAT_ENV_VAR, DEFAULT_HEARTBEAT_MS),
        election_timeout_min_ms=_env_int(
            source, ELECTION_MIN_ENV_VAR, DEFAULT_ELECTION_TIMEOUT_MIN_MS
        ),
        election_timeout_max_ms=_env_int(
            source, ELECTION_MAX_ENV_VAR, DEFAULT_ELECTION_TIMEOUT_MAX_MS
        ),
        reserved=reserved_ports(source),
    )
