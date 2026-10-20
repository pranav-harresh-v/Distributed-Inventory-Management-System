# Raft package: Phase 6, leader election

Phase 6 builds the Raft foundation of the distributed inventory system: three
independent Python processes that talk over gRPC, elect one leader, notice when it
dies, and elect another. **That is all it does.** There is no replicated log, no
commit logic and no connection to the inventory application or its database. Those
are Phase 7 and Phase 8.

## 1. Objectives

- Three independent Raft nodes (Follower, Candidate, Leader) as separate processes.
- Persistent `current_term` and `voted_for`, one SQLite file per node.
- Randomized election timeouts, `RequestVote`, and empty `AppendEntries` heartbeats.
- Higher-term handling, leader failure detection and re-election.
- Controlled start-up and shutdown, and automated election tests.

## 2. Modules

| Module | Role |
|---|---|
| `config.py` | Validated `RAFT_` configuration (`load_config`, `RaftConfig`). |
| `storage.py` | `RaftStateStore`: durable `current_term` / `voted_for` in a per-node SQLite file. |
| `node.py` | `RaftNode`: the election state machine. Clock, storage, transport and randomness are injected, so it is tested without sockets. |
| `peers.py` | `GrpcPeerTransport`: outbound `RequestVote` / `AppendEntries` with bounded deadlines; `fetch_state`. |
| `service.py` | `RaftService`: gRPC translation only. All Raft rules live in `RaftNode`. |
| `server.py` | `python -m raft.server`: one cluster member per process; `create_server`, `RaftMember`. |

Related files: `proto/raft.proto` (the contract), `scripts/raft_status.py` (status tool).
The package imports nothing from `app`, `client` or `llm` (a test checks this).

## 3. RPCs (`inventory.raft.v1.RaftService`)

| RPC | Purpose |
|---|---|
| `RequestVote` | A candidate asks for a vote. |
| `AppendEntries` | The leader's heartbeat. In Phase 6 it carries no entries; a request with entries is refused. |
| `GetState` | Read-only: node id, role, term, known leader, vote, commit and log indexes. |

The log-related fields (`prev_log_*`, `entries`, `leader_commit`, `match_index`,
`conflict_*`) already exist in the messages so Phase 7 does not change the wire contract.

## 4. Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `RAFT_NODE_ID` | *(required)* | This node's id, 1 or more. |
| `RAFT_PEERS` | `1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103` | The whole membership, this node included. |
| `RAFT_HOST` / `RAFT_PORT` | this node's entry in `RAFT_PEERS` | Where it listens; must match its entry (`0.0.0.0` is allowed as host). |
| `RAFT_STATE_PATH` | `data/raft_node<id>.db` | This node's election-state file. |
| `RAFT_HEARTBEAT_MS` | 100 | Heartbeat interval. |
| `RAFT_ELECTION_TIMEOUT_MIN_MS` / `_MAX_MS` | 500 / 1000 | Randomized election timeout range; min must be at least 5 times the heartbeat. |
| `INVENTORY_LOG_LEVEL` | `INFO` | Reused from the application for log verbosity. |

Raft refuses ports used by the rest of the system (50051, 50052, 11434, 8501, 8502)
and refuses a state path that is the inventory database (`data/node1.db`).

## 5. Default ports and state location

| Node | Address | State file |
|---|---|---|
| 1 | 127.0.0.1:50101 | `data/raft_node1.db` |
| 2 | 127.0.0.1:50102 | `data/raft_node2.db` |
| 3 | 127.0.0.1:50103 | `data/raft_node3.db` |

The state files are gitignored (`data/*.db`) and are separate from the inventory
database `data/node1.db`, which Raft never opens.

## 6. Start a local three-node cluster

From the repository root, one terminal per node (PowerShell; adjust the interpreter path):

```powershell
$env:RAFT_NODE_ID = "1"; & ..\venv\Scripts\python.exe -m raft.server
```
```powershell
$env:RAFT_NODE_ID = "2"; & ..\venv\Scripts\python.exe -m raft.server
```
```powershell
$env:RAFT_NODE_ID = "3"; & ..\venv\Scripts\python.exe -m raft.server
```

## 7. Inspect the cluster

```powershell
& ..\venv\Scripts\python.exe scripts/raft_status.py --peers "1=127.0.0.1:50101,2=127.0.0.1:50102,3=127.0.0.1:50103"
```

It asks every node for `GetState` (bounded 1 s deadline each) and prints reachability,
role, term and known leader. It is read-only and never opens a database. Exit status is
0 if at least one node answered, 2 otherwise.

## 8. Stop and restart nodes

- **Graceful:** Ctrl+C or Ctrl+Break in the node's terminal (SIGTERM elsewhere). The node
  logs `Raft node stopped` and exits 0.
- **Abrupt:** closing the window or killing the process is safe: term and vote are on disk
  before they are used.
- **Restart:** run the same start command again. The node logs
  `starting as follower with persisted term=... voted_for=...` and rejoins as a follower.
- **Do not delete** a `data/raft_node<id>.db` file while the cluster is in use. A node that
  forgets its vote can vote twice in one term, which breaks Raft's safety guarantee. Reset
  the files only with every node stopped.
- A node left running **alone** keeps starting elections it cannot win, so its term grows
  (about one per election timeout). This is expected without PreVote and harmless.

## 9. Automated tests

Generated protobuf stubs are **gitignored**. After a fresh checkout, regenerate them first
(`scripts/generate_proto.ps1`, or `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/raft.proto`
for the Raft stubs alone). The tests fail to import without them.

```powershell
& ..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/unit/test_raft_proto_contract.py tests/unit/test_raft_config.py tests/unit/test_raft_storage.py tests/unit/test_raft_node.py
```
```powershell
& ..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/integration/test_raft_election.py
```
```powershell
& ..\venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/integration/test_raft_cluster_processes.py
```

All Raft tests use temporary state files and free ports; none touches `data/`.

## 10. Verified results (2026-10-20)

| Run | Result |
|---|---|
| Raft unit tests (4 files) | 286 passed, 0 failed, 0 skipped, 9.25 s |
| Raft gRPC integration (in-process, 3 servers) | 22 passed, 15.70 s |
| Raft process-level (3 real processes) | 7 passed, 18.87 s |
| Full suite in an isolated copy | 1440 passed, 0 failed, 0 skipped: 1125 pre-existing + 315 Raft |
| Ruff (`app llm client raft tests scripts`) | All checks passed |
| compileall (same directories, in the isolated copy) | exit 0 |

See `docs/audits/phase6_final_report.md` for the full review.

## 11. Known limitations

- **No log replication, no commit index** (`commit_index` is always 0): Phase 7.
- **No inventory integration:** the Raft server is a separate process from the application server.
- **No CheckQuorum:** a leader cut off from the majority keeps believing it leads its old term
  until it hears a newer one. It can commit nothing, so it is not a safety problem, but it is a
  liveness and observability one.
- **No PreVote:** a node that was isolated or restarted can bump the term and depose a healthy leader.
- **Insecure channels:** no TLS and no authentication between nodes. `GetState` is read-only and unauthenticated.
- **Windows graceful shutdown** relies on Ctrl+Break, which needs a console.
- **One unexplained intermittent test failure** was seen once in about 58 combined runs of the
  two integration files (see the final report). It has not recurred.

## 12. What remains for Phase 7

Persistent log (`raft_log`, stored with the term in the same node file so they commit together),
`AppendEntries` with entries and the log-matching check, per-follower `nextIndex` / `matchIndex`,
majority commit and commit-index advance, applying committed entries to a state machine, and the
election restriction using real log positions. Phase 8 then routes inventory writes through the log
(deterministic commands, request-id de-duplication, client redirect to the leader, session handling).
