# Phase 6 Final Report: Raft Foundation and Leader Election

Branch `phase6/raft-election` (base `96af753`, tag `milestone-1`), reviewed 2026-10-20.
Nothing is committed or pushed. Every result below was produced by a command run during
this review; nothing is carried over from older logs. Anything not run is marked
NOT VERIFIED.

## 1. Scope and outcome

Phase 6 delivers three independent Raft node processes that communicate over gRPC,
elect one leader dynamically, detect its failure and elect a replacement, and keep
`current_term` and `voted_for` durable across restarts. It deliberately contains no
log replication, no commit logic, and no connection to the inventory application or
`data/node1.db`.

## 2. What was reviewed

`proto/raft.proto`; `raft/config.py`, `storage.py`, `node.py`, `peers.py`, `service.py`,
`server.py`; `scripts/raft_status.py`; seven new test files and one helper module.
The review checked the code against `docs/audits/pre_phase6_audit.md` and the agreed design.

### 2.1 Safety review: no defects found

| Property | Evidence |
|---|---|
| At most one vote per term; a repeat only to the same candidate | `RaftNode.handle_request_vote`: `voted_for in (None, candidate)`; tested with a store that has no guards, so the node's own rule is on trial. |
| Persist before use | Every assignment of `_term` / `_voted_for` in `node.py` immediately follows a successful `_persist`. If the save fails the vote is refused, the term is not adopted and the election is not started. |
| Terms never decrease | Node refuses lower terms; the store independently rejects a lowering `save` and a changed vote within a term. |
| Leader only with a majority | `_become_leader` has exactly two callers, each after a `quorum` check. No leader is assigned anywhere else (grep). |
| Stale replies cannot elect | Replies are judged against the term they were sent in; votes are a set of peer ids. |
| Higher term deposes any role | Every handler adopts a higher term and becomes Follower; tested for vote requests, heartbeats and heartbeat replies. |
| No log replication pretended | Requests with entries are refused (`success=false`); `commit_index` is always 0. |
| No inventory access | `raft/` imports nothing from `app`, `client`, `llm` or `sqlalchemy` (AST test and grep). `RaftStateStore` refuses `data/node1.db`, its sidecars and `INVENTORY_DB_PATH`. |
| Independent node state | One file per node, tagged with its node id; opening another node's file raises. |

### 2.2 Liveness review

No defect against the Phase 6 requirements. One real defect was found and fixed in
Phase 6.2 (a higher term from a refused vote reset the election timer, letting a
candidate with a stale log postpone everyone's elections; now only a granted vote or a
valid leader resets it). Known characteristics, not defects: no PreVote (an isolated or
restarted node can bump the term and depose a healthy leader; a lone node's term keeps
growing, about one per election timeout) and no CheckQuorum (a cut-off leader keeps
believing it leads its old term). Both are documented and tested as such.

### 2.3 Change made during this review

`tests/integration/raft_support.py::free_ports` now draws test ports from 20000-29999 and
proves each bindable, instead of taking OS-assigned ports. Reason: the OS hands out its
ephemeral range (49152 and up on Windows) for outgoing connections too, so a port chosen
for a test server could be in use by an unrelated live connection when the server binds
it. This is a **hypothesis** for the intermittent failure in section 6; it has not been
proven to be the cause. No production Raft code and no Milestone 1 module was changed.

## 3. Test results (all executed in this review)

Commands run from the repository root with `PYTHONDONTWRITEBYTECODE=1`, interpreter
`AOS\venv\Scripts\python.exe`, `-p no:cacheprovider`.

| Group | Collected | Passed | Failed | Skipped | Duration |
|---|---|---|---|---|---|
| Raft unit: `test_raft_proto_contract` | 21 | 21 | 0 | 0 | 0.99 s |
| Raft unit: `test_raft_config` | 85 | 85 | 0 | 0 | 1.35 s |
| Raft unit: `test_raft_storage` | 58 | 58 | 0 | 0 | 2.80 s |
| Raft unit: `test_raft_node` | 122 | 122 | 0 | 0 | 6.17 s |
| **Raft unit, all four files** | **286** | **286** | 0 | 0 | 9.25 s |
| Raft gRPC integration (`test_raft_election.py`, 3 servers in one process) | 22 | 22 | 0 | 0 | 15.70 s |
| Raft process-level (`test_raft_cluster_processes.py`, 3 real processes) | 7 | 7 | 0 | 0 | 18.87 s |

The unit tests that cover the election rules use a fake clock, a stub transport and real
temporary state files; they sleep almost never. The integration tests poll with deadlines.

**Process-level results (real operating-system processes, real gRPC):** three processes
elect exactly one leader with three separate state files; stopping the leader triggers
re-election in a higher term, both **gracefully** (Ctrl+Break, exit 0, `Raft node stopped`
logged) and **abruptly** (`kill()`, non-zero exit, no stop line); a restarted old leader
logs the recovered `persisted term=... voted_for=...`, rejoins as a follower of the new
leader and does not seize leadership back; a lone survivor (leader killed, another node
stopped) never becomes leader over about 4 s while its term keeps rising; leadership
returns when a second node starts; every node shuts down gracefully with exit 0 and frees
its port. No leftover process or listener remained afterwards. The tests terminate only
the child processes they started. No test isolates a live leader.

## 4. Milestone 1 regression

A fresh disposable copy of the repository was created outside the project, excluding
`.git`, `data/`, `.env`, `CLAUDE.local.md`, virtual environments, caches and every
`*.db*` file, and including the generated protobuf stubs (client, llm, raft) and all
Raft sources. The full suite ran there with pytest's **default** temporary directory.

| Result | Value |
|---|---|
| Command | `python -m pytest -q -ra -p no:cacheprovider --junitxml=... --durations=5` |
| Totals | **1440 passed, 0 failed, 0 errors, 0 skipped**, 405.38 s |
| Pre-existing (non-Raft) tests | **1125 passed** (the historical baseline), 0 failed |
| New Raft tests | 315 passed (286 unit + 29 integration) |

This includes `test_the_suite_runs_against_a_temporary_database`, which fails under a
custom `--basetemp` (documented in the pre-Phase-6 audit) and passes here.

## 5. Lint and compilation

- `ruff check app llm client raft tests scripts --no-cache`: **All checks passed.**
- `python -m compileall -q app llm client raft tests scripts`, run inside the isolated
  copy so that no bytecode lands in the project: **exit code 0.**

## 6. Remaining defect: one unexplained intermittent test failure

During Phase 6.3, in the first batch of repeated runs of the two Raft integration files,
run 6 of 6 reported `1 failed, 28 passed`. The failing test and its output were **not
captured**. It overlapped with an unrelated ruff run and edits to the test files.

Investigation since then, all without reproducing the failure:

- 58 runs of both integration files together (1 failure, the original), then 57 clean runs:
  12 + 5 (under full CPU load on every core) + 20 + 15 (after the port-selection change).
- 25 further runs of the in-process file and 15 of the process file, all under full CPU load: 0 failures.
- 60 follower-restart iterations in-process, checking whether a restarted follower disturbs the
  leader (a hypothesis): 0 disturbed.
- Result: the cause is **unknown**. The port-range change is a plausible but **unproven**
  mitigation. At a base rate near 1 in 58 it cannot be shown to have helped.

Treat it as an open item. If it recurs, run with `-ra` and keep the output. The tests that
rely on strict timing are `test_a_stable_leader_keeps_leading_through_heartbeats` and the
election-timing assertions, and a long scheduling stall would legitimately trigger an
election.

## 7. Database and Git safety

- `data/node1.db`, `-wal`, `-shm`: SHA-256 prefixes `295175F3...`, `2C6C9EF4...`, `C488E1E4...` and
  modification times (2026-09-26) unchanged from before Phase 6 began, after every run in this review.
- `git status`: modified tracked files are only `proto/raft.proto` and `raft/README.md`;
  everything else is new and untracked. Nothing is staged or committed.
- `git diff` shows **no change** to `app/`, `client/`, `llm/`, `proto/client.proto`,
  `proto/llm.proto`, `app/storage/models.py`, `requirements.txt`, `.gitignore` or `docker-compose.yml`.
  No dependency was added or upgraded. The inventory schema is unchanged.
- Git warns that LF will become CRLF in `proto/raft.proto` and `raft/README.md` on next touch; harmless.

### 7.1 Raft state files that now exist in `data/`

`data/raft_node1.db`, `raft_node2.db` and `raft_node3.db` (8 KB each, gitignored) exist,
created 9:23-9:24 PM and last modified 9:27-9:29 PM on 2026-10-20, **after** the previous phase's
checks. Read from copies, they hold: node 1 term 63 voted 1; node 2 term 60 voted 2; node 3 term 60 voted 2.

They were **not created by an automated test**: every Raft test supplies a temporary state path
explicitly (checked in the source), and none of this review's in-repository test runs after
9:29 PM changed their modification times. Their pattern (creation 44 and 48 seconds apart; a
term near 60 on a node that ran alone for about 44 s at one election per 0.75 s) matches someone
starting the three nodes by hand, one after another, as the documented manual procedure describes.
This is an inference, not a log-backed fact. They were left untouched. They are harmless but mean
the next manual run resumes at term 63.

## 8. Known limitations

No log replication or commit; no inventory integration; no CheckQuorum; no PreVote; insecure
unauthenticated channels; Windows graceful stop needs a console; the unexplained flake in section 6.

## 9. Phase 7 prerequisites

1. Persist a `raft_log` in the same per-node file as the term, so a log append and a term change can share a transaction.
2. Implement `AppendEntries` with entries: the log-matching check, conflict resolution, per-follower `nextIndex` / `matchIndex`, and the `conflict_index` fast backup (fields already in the proto).
3. Majority commit with the current-term rule, `commit_index` advance, and an apply loop to a state machine.
4. Supply a real `RaftLog` to the node so the election restriction compares real log positions (the interface exists; Phase 6 uses `EmptyLog`).
5. Decide on CheckQuorum and PreVote before relying on leader-only writes.
6. For Phase 8: deterministic commands (the leader must fix `order_number` and timestamps), `request_id` de-duplication, client redirect to the leader, replicated or stateless sessions, identical initial state on every replica, and integrating the Raft server into the application server process.

## 10. Final acceptance checklist

| # | Requirement | Result | Evidence |
|---|---|---|---|
| A | Three independent Raft nodes start | PASS | process-level test; manual rehearsal on default ports |
| B | One leader under stable conditions | PASS | integration + process tests |
| C | Voting obeys standard Raft rules | PASS | 122 node tests incl. log freshness and one-vote-per-term |
| D | Terms never decrease | PASS | storage and node tests; full-cluster restart simulation |
| E | Persistent votes survive restart | PASS | unit tests; process test logs recovered state |
| F | Heartbeats prevent unnecessary elections | PASS | unit tests; stable-leader integration test |
| G | Stale leaders step down on higher terms | PASS | unit tests (3 paths); gRPC deposal test; simulated partition heal |
| H | Leader failure triggers re-election | PASS | integration and process tests (graceful and abrupt) |
| I | Restored nodes rejoin safely | PASS | restarted follower and old-leader process tests |
| J | No new leader without a majority | PASS | minority unit tests; lone-node and lone-survivor tests |
| K | Existing Milestone 1 tests pass | PASS | 1125 of 1125 in the isolated copy |
| L | Original application data protected | PASS | node1.db hashes and timestamps unchanged |
| M | No unintended API changes | PASS | `client.proto` / `llm.proto` / app code unchanged; contract tests pass |
| N | No inventory replication introduced | PASS | no app imports in `raft/`; entries refused; `commit_index` always 0 |

NOT VERIFIED: stability of the Raft integration tests over very long runs (one unexplained
failure in about 58 combined runs); behaviour across separate machines (all testing was on
one Windows host over loopback); any Linux or macOS run.
