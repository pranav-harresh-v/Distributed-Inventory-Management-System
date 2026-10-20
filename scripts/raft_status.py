"""Show what each Raft node currently believes. Read-only.

Usage (from anywhere)::

    python scripts/raft_status.py --peers "1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103"

With no ``--peers`` it uses ``RAFT_PEERS`` from the environment, then the local
development cluster. Each node is asked through its ``GetState`` RPC with a short
deadline; the tool opens no database, changes nothing on any node, and an
unreachable node is reported, not fatal.

Exit status: 0 if at least one node answered, 2 if none did (or the arguments
were invalid).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the project importable when run as a script (python scripts/raft_status.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proto import raft_pb2
from raft.config import DEFAULT_PEERS, ConfigError, Member, parse_peers
from raft.peers import fetch_state

DEFAULT_TIMEOUT_SECONDS = 1.0

_ROLE_NAMES = {
    raft_pb2.NODE_ROLE_FOLLOWER: "FOLLOWER",
    raft_pb2.NODE_ROLE_CANDIDATE: "CANDIDATE",
    raft_pb2.NODE_ROLE_LEADER: "LEADER",
}


def collect(members: tuple[Member, ...], timeout: float) -> dict[int, object]:
    """Ask every member; the value is its GetStateResponse or None."""
    return {m.node_id: fetch_state(m.address, timeout) for m in members}


def render(members: tuple[Member, ...], states: dict[int, object]) -> str:
    """Format the cluster as a table plus a one-line verdict."""
    lines = [
        f"Raft cluster status ({len(members)} members)",
        f"{'ID':<4}{'ADDRESS':<22}{'STATE':<13}{'ROLE':<11}{'TERM':<7}{'LEADER'}",
    ]
    leaders: list[tuple[int, int]] = []  # (term, node id)
    for member in members:
        state = states.get(member.node_id)
        if state is None:
            lines.append(
                f"{member.node_id:<4}{member.address:<22}{'unreachable':<13}{'-':<11}{'-':<7}-"
            )
            continue
        role = _ROLE_NAMES.get(state.role, "UNKNOWN")
        leader = state.leader_id if state.leader_id else "unknown"
        lines.append(
            f"{member.node_id:<4}{member.address:<22}{'reachable':<13}{role:<11}"
            f"{state.current_term:<7}{leader}"
        )
        if state.role == raft_pb2.NODE_ROLE_LEADER:
            leaders.append((state.current_term, member.node_id))

    reachable = sum(1 for s in states.values() if s is not None)
    lines.append("")
    lines.append(f"{reachable} of {len(members)} nodes reachable.")
    lines.append(_verdict(leaders))
    return "\n".join(lines)


def _verdict(leaders: list[tuple[int, int]]) -> str:
    if not leaders:
        return "No leader reported (an election may be in progress, or no majority is up)."
    if len(leaders) == 1:
        term, node_id = leaders[0]
        return f"Leader: node {node_id} (term {term})."
    terms = {term for term, _ in leaders}
    listing = ", ".join(f"node {n} (term {t})" for t, n in sorted(leaders, reverse=True))
    if len(terms) < len(leaders):
        return f"!! Two leaders in the SAME term: {listing}. This must never happen."
    newest = max(leaders)
    return (
        f"Several nodes claim leadership in different terms: {listing}. "
        f"Node {newest[1]} has the newest term; the others are stale (for example "
        "cut off from the cluster) and will step down when they hear it."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--peers",
        default=os.environ.get("RAFT_PEERS") or DEFAULT_PEERS,
        help="all members, e.g. 1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"seconds to wait for each node (default {DEFAULT_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2
    try:
        members = parse_peers(args.peers)
    except ConfigError as exc:
        print(f"Invalid --peers: {exc}", file=sys.stderr)
        return 2

    states = collect(members, args.timeout)
    print(render(members, states))
    return 0 if any(s is not None for s in states.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
