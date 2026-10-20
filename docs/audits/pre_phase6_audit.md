# Pre-Phase 6 Audit Report

Conventions used in this report:

- **[V]** verified from source, live system state, or an executed test run in this audit.
- **[I]** inferred from verified facts; not directly observed.
- **[NV]** not verified.
- **[P]** proposal (design recommendation, not existing code).

No secrets, session tokens, usernames or password hashes appear in this report.

---

## 1. Project Information

| Item | Value |
|---|---|
| Course | CS G623 Advanced Operating Systems, BITS Pilani |
| Project | Distributed Inventory Management System (Option B) |
| Repository root | `distributed-inventory-system/` (the parent folder `AOS/` is not itself a Git repository) [V] |
| Remote | `origin` = `pranav-harresh-v/Distributed-Inventory-Management-System` on GitHub [V] |
| Audit scope | Read-only static inspection, isolated regression run, Phase 6 design |
| Next phase | Phase 6: Raft foundation and leader election |

Audit method, in order: static repository inspection; read-only schema inspection of a *copy* of `data/node1.db`; test-safety analysis; execution of the test suite from a disposable copy of the repository; Phase 6 design analysis. No application code, database, configuration, test or dependency was modified. The only file written inside the repository by this audit is this report.

---

## 2. Milestone-1 Completion Status

**Assessment: complete and stable as a base for Phase 6, with two items that were not exercised live (see below).**

| Milestone-1 requirement | Status | Evidence |
|---|---|---|
| Python implementation | Done [V] | `app/`, `llm/`, `client/`, `scripts/` |
| gRPC service definitions | Done [V] | `proto/client.proto`, `proto/llm.proto` |
| Client-server communication | Done [V] | `client/grpc_client.py` to `app/api/client_service.py` |
| Authentication and sessions | Done [V] | `app/domain/auth.py` |
| Order processing, correct decrement, no negative stock | Done [V] | `app/domain/orders.py:place_order`, `app/storage/repositories.py:InventoryRepository.decrement_quantity` |
| Demand prediction and reorder suggestions | Done [V] | `app/domain/forecasting.py` (read in part only; see Section 11 for tests) |
| Inventory analytics | Done [V] | `app/domain/analytics.py` (read in part only) |
| Separate context-aware LLM server | Done [V] | `llm/server.py`, `llm/service.py`, `llm/ollama_client.py`, `llm/prompts.py` |
| Supplier restocking | **Not implemented** [V]: `restock_orders` table and ORM model exist, no operation writes to it, and `app/domain/suppliers.py` is a 0-byte file |

Not exercised in this audit:

- Real Ollama / Phi-3 generation. **NOT EXECUTED.** Ollama was listening on 11434 [V], but LLM tests use injected fakes and the test suite deliberately blocks real model access. The project notes record that the `phi3:mini` tag is not pulled and `phi3:latest` is; this was **not re-checked** [NV].
- Live-server smoke tests (`scripts/grpc_smoke_test.py`, `scripts/demo_llm_queries.py`) and Streamlit start-up. **NOT EXECUTED.**

---

## 3. Repository and Git Baseline

All values below were read with read-only Git commands (`GIT_OPTIONAL_LOCKS=0`) [V].

| Item | Value |
|---|---|
| Branch | `main` |
| HEAD commit | `96af753` "complete milestone 1" (2026-09-26) |
| Tag | `milestone-1` (same commit) |
| Upstream state | `main...origin/main [ahead 10]`: **10 commits exist only locally** |
| Working tree | Clean: nothing staged, modified or untracked, before and after the regression run |
| Tracked database or `.env` files | None |

Ignored files present (from `.gitignore`): `data/` (`data/*.db`, `-wal`, `-shm`), `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `.env`, `.vscode/`, and the generated stubs `proto/*_pb2.py` and `proto/*_pb2_grpc.py`. The generated stubs are **not committed**; they exist locally.

Documentation present at the repository root: `README.md`, `ARCHITECTURE.md`, `PROJECT_SPEC.md`, `PHASE_F_NOTES.md`, `CLAUDE.md`, `CLAUDE.local.md`. (`CLAUDE.local.md` is described as personal and untracked in its own text; it did not appear in `git status`.)

Repository tree (virtualenv, caches and `.git` excluded):

```
distributed-inventory-system/
  app/
    server.py
    api/        business_service.py, client_service.py, llm_client.py
    domain/     analytics.py, auth.py, forecasting.py, inventory.py, orders.py, suppliers.py (0 bytes)
    state_machines/  commands.py (0 bytes; no __init__.py)
    storage/    database.py, models.py, repositories.py
  client/       app.py, user_app.py, grpc_client.py
  llm/          server.py, service.py, ollama_client.py, prompts.py, engine.py (0 bytes)
  proto/        client.proto, llm.proto, raft.proto (0 bytes), inventory.proto (0 bytes)
                + gitignored client_pb2*.py, llm_pb2*.py
  raft/         README.md (0 bytes; no __init__.py)
  scripts/      init_db.py, seed_data.py, grpc_smoke_test.py, demo_llm_queries.py, generate_proto.ps1
  docker/       app.Dockerfile, client.Dockerfile, llm.Dockerfile (all 0 bytes)
  docker-compose.yml (0 bytes)
  tests/        conftest.py, unit/ (21 files), integration/ (11 files, plus conftest.py)
  data/         node1.db (+ -wal, -shm)   [gitignored]
  docs/audits/  pre_phase6_audit.md   [this report]
```

---

## 4. Development Environment

| Item | Value | Status |
|---|---|---|
| Python | 3.12.10 | [V] |
| Virtual environment | `AOS/venv`, one level **above** the repository | [V] |
| Platform | Windows 11; README states "verified on Windows 11 with Git Bash"; stub generation is a PowerShell script (`scripts/generate_proto.ps1`) | [V] |
| `pip check` | "No broken requirements found" | [V] |
| `INVENTORY_DB_PATH` | Unset at audit time | [V] |

Installed versions match the pins in `requirements.txt` [V]:

| Package | Version |
|---|---|
| SQLAlchemy | 2.0.54 |
| bcrypt | 5.0.0 |
| grpcio / grpcio-tools | 1.84.0 |
| protobuf | 7.36.2 |
| httpx | 0.28.1 |
| ollama | 0.6.2 |
| streamlit | 1.64.0 |
| pytest | 9.1.1 |
| ruff | 0.16.8 |

`pytest-cov` 7.1.0 is also installed but is not declared in `requirements.txt` [V]. `.coverage` exists in the repository folder (dated 2026-09-22) and was not changed by this audit [V].

Test, lint and run commands (from the repository README) [V]: `pytest -q`; `python -m ruff check app llm client tests scripts`; `python -m compileall -q app llm client tests scripts`; `python -m app.server`; `python -m llm.server`; `streamlit run client/app.py --server.port 8501`; `streamlit run client/user_app.py --server.port 8502`.

Configuration notes:

- There is no `pytest.ini`, `pyproject.toml`, `setup.cfg` or `ruff.toml` [V]. `tests/conftest.py` inserts the project root into `sys.path`.
- `.env.example` documents the environment variables, but no `python-dotenv` dependency is declared, so a `.env` file does not appear to be loaded by any code [I]. Only real process environment variables take effect.

---

## 5. Current System Architecture

```
client (Streamlit 8501 / 8502) --gRPC--> application server :50051 --gRPC--> LLM server :50052 --HTTP--> Ollama :11434
                                                |
                                          SQLite (data/node1.db)
```

[V] from `ARCHITECTURE.md`, `app/server.py`, `llm/server.py`, `.env.example`.

Verified service ports and configuration variables:

| Service | Default | Variable |
|---|---|---|
| Application gRPC | `127.0.0.1:50051` | `INVENTORY_GRPC_HOST`, `INVENTORY_GRPC_PORT` |
| LLM gRPC | `127.0.0.1:50052` | `LLM_GRPC_HOST`, `LLM_GRPC_PORT` |
| Ollama | `http://127.0.0.1:11434` | `OLLAMA_HOST` |
| Admin Streamlit | 8501 | `--server.port` |
| User Streamlit | 8502 | `--server.port` |

Application server (`app/server.py`) [V]:

- `create_server()` (l.59) builds a `grpc.server` on a `ThreadPoolExecutor` (default 8 workers) and registers only `ClientService` (l.87). It returns `(server, bound_port)` without starting, and supports port 0.
- `serve()` (l.96) calls `init_db()` (l.104), installs `SIGINT`/`SIGTERM` handlers (l.118-119) and blocks.
- Channels and servers are insecure (no TLS).

Storage (`app/storage/database.py`) [V]:

- One process-global engine is built at import time from `INVENTORY_DB_PATH` (l.115); it connects lazily.
- Per-connection pragmas: `foreign_keys=ON`, `journal_mode=WAL`, `busy_timeout=5000`.
- `session_scope()` (l.126) commits once on success and rolls back and re-raises on failure.
- `init_db()` (l.154) runs `Base.metadata.create_all`. There is no migration framework.

Concurrency model [V]: no background threads, no asyncio, no timers. The only `threading` use is `ClientService._llm_lock` (`app/api/client_service.py:330`). The system is purely request-driven.

Single-node assumptions present today [V]: no node id, no peer list, no Raft port, no leader concept; the default bind address is loopback; the client dials one fixed address with no leader redirect or retry (`client/grpc_client.py:177-186`).

---

## 6. gRPC and Protocol Buffer Contracts

| Service | Proto | Server implementation | Client implementation |
|---|---|---|---|
| `inventory.client.v1.ClientService` | `proto/client.proto` | `app/api/client_service.py` | `client/grpc_client.py` |
| `inventory.llm.v1.LLMService` | `proto/llm.proto` | `llm/service.py` | `app/api/llm_client.py` |

Generated stubs (`proto/client_pb2.py`, `client_pb2_grpc.py`, `llm_pb2.py`, `llm_pb2_grpc.py`) are gitignored and present locally [V]. They were not regenerated.

**ClientService RPCs** (exact names and casing as implemented) [V]:

| RPC | Request | Response |
|---|---|---|
| `login` | `LoginRequest` | `LoginResponse` |
| `logout` | `LogoutRequest` | `StatusResponse` |
| `post` | `PostRequest` | `StatusResponse` |
| `get` | `GetRequest` | `GetResponse` |
| `processBusinessRequest` | `BusinessRequest` | `BusinessResponse` |

- Lowercase `login`/`logout`/`post`/`get` and camelCase `processBusinessRequest` follow the assignment's naming deliberately (stated in `proto/client.proto`).
- Domain payloads are `google.protobuf.Struct` (`payload`, `params`, `context`, `data`); only the authentication and envelope messages are strongly typed.
- `request_id` exists **only** on `BusinessRequest` (field 4). `PostRequest` has no `request_id`.
- **`processBusinessRequest` is currently the LLM advisory operation, not the write path** (`app/api/client_service.py:474-528`; operations `DEMAND_FORECAST`, `REORDER_RECOMMENDATION`, `INVENTORY_SUMMARY` in `app/api/business_service.py`). All authoritative writes go through `post` with a `resource` string such as `order.place`.

**LLMService:** one RPC, `getLLMAnswer(GetLLMAnswerRequest) -> GetLLMAnswerResponse`; context is a `Struct`; failures are reported as gRPC status codes (`UNAVAILABLE`, `DEADLINE_EXCEEDED`, `INVALID_ARGUMENT`, `INTERNAL`) [V].

**Raft RPCs:** none. `proto/raft.proto` and `proto/inventory.proto` are 0-byte placeholders [V].

`scripts/generate_proto.ps1` compiles every `.proto` under `proto/` that has content, with the repository root as include path [V from its documentation block].

`tests/unit/test_proto_contract.py` checks that the five ClientService RPCs and their message types exist and that no response message carries a credential field [V]. The project's working rules treat `client.proto` and the database schema as frozen.

---

## 7. SQLite Database Schema

Source: read-only inspection of a **copy** of `data/node1.db`, `-wal` and `-shm` placed in the scratchpad (the live files were never opened). The live files' hashes and timestamps were identical before and after all work [V]. ORM source: `app/storage/models.py`.

The database exists and contains 11 tables, no triggers and no views [V].

| Table | Model (`models.py`) | Key constraints | Rows in copy |
|---|---|---|---|
| `users` | `User` :163 | unique `username`, unique `email` | 2 |
| `sessions` | `Session` :194 | unique `token_hash`; FK `user_id` to `users` (ON DELETE CASCADE); `revoked` flag | 20 |
| `products` | `Product` :220 | unique `sku`; CHECK `unit_price >= 0` | 4 |
| `warehouses` | `Warehouse` :247 | unique `code` | 2 |
| `inventory` | `Inventory` :270 | UNIQUE(`warehouse_id`, `product_id`); CHECK `quantity >= 0`, `reorder_point >= 0`, `safety_stock >= 0`; FKs to warehouses, products | 8 |
| `suppliers` | `Supplier` :324 | **no unique constraint** (name uniqueness is an application-level rule only) | 2 |
| `supplier_products` | `SupplierProduct` :343 | UNIQUE(`supplier_id`, `product_id`); CHECK cost >= 0, lead time >= 0, minimum order qty > 0 | 5 |
| `orders` | `Order` :392 | unique `order_number`; CHECK `total_amount >= 0`; FKs to users, warehouses | 4 |
| `order_items` | `OrderItem` :442 | UNIQUE(`order_id`, `product_id`); CHECK `quantity > 0`, prices >= 0; FK to orders (CASCADE), products | 5 |
| `restock_orders` | `RestockOrder` :477 | unique `restock_number`; CHECK `quantity > 0`, `unit_cost >= 0`; FKs to warehouses, products, suppliers | 0 |
| `sales_history` | `SalesHistory` :524 | CHECK `quantity > 0`; FKs to orders, warehouses, products; indexes on `sold_at` | 5 |

Presence of tables and columns named in the audit brief [V]:

| Name | Status |
|---|---|
| `users`, `products`, `warehouses`, `inventory`, `orders`, `order_items`, `sales_history`, `suppliers`, `supplier_products` | EXISTS |
| `restock_orders` | EXISTS (table and ORM model only; unused by any operation) |
| `sessions` | EXISTS (additional table) |
| `request_id` column on any table | MISSING |
| `processed_requests` | MISSING |
| `reserved_quantity` | MISSING |
| `raft_log` | MISSING |
| `raft_metadata` | MISSING |

Money columns use `Numeric(12, 2)` through a `Money` type that rejects floats; timestamps are stored as UTC [V]. The `sessions` table accumulates rows: expired and revoked sessions are rejected but never purged [V].

---

## 8. Inventory Business Logic

All authoritative writes enter through `ClientService.post` (`app/api/client_service.py:435-453`), which runs the handler inside one `session_scope` transaction: a single commit on success, rollback on any exception. Repositories flush only (`app/storage/repositories.py`) [V].

| `post` resource | Function | Tables written | Validation / errors |
|---|---|---|---|
| `product.create` | `app/domain/inventory.py:create_product` (:356) | products | duplicate SKU, price >= 0, field whitelist |
| `product.update` | `update_product` (:438) | products | field whitelist |
| `warehouse.create` | `create_warehouse` (:502) | warehouses | unique code |
| `inventory.set_initial_stock` | `set_initial_stock` (:658) | inventory (INSERT) | quantity >= 0; refuses if the row exists |
| `inventory.add_stock` | `add_stock` (:712) | inventory (`quantity = quantity + n`) | amount >= 1 |
| `supplier.create` | `create_supplier` (:879) | suppliers | name uniqueness is app-level only (not race-proof) |
| `supplier.associate_product` | `associate_supplier_product` (:957) | supplier_products | unique pair, cost and lead-time checks |
| `order.place` | `app/domain/orders.py:place_order` (:193) | orders, order_items, inventory, sales_history | see below |

Not through `post`: `login` writes a `sessions` row (`app/domain/auth.py:login`, :206) and `logout` marks it revoked (`revoke_session`, :263).

**Order placement** [V] (`app/domain/orders.py`): everything is validated before any stock is touched; duplicate lines are merged and processed in ascending product-id order; each decrement is a single guarded statement, `UPDATE inventory SET quantity = quantity - :amount WHERE ... AND quantity >= :amount` (`InventoryRepository.decrement_quantity`, `repositories.py:373-405`); zero affected rows raises `InsufficientStockError` (mapped to `FAILED_PRECONDITION`); any failure rolls back the whole order. Stock cannot become negative through this path, and the `CHECK (quantity >= 0)` constraint is a second barrier.

**Non-determinism relevant to Raft** [V]: `place_order` builds `order_number` from `uuid4` (`orders.py:166`), takes `sold_at` from `utc_now()`, and stores database-assigned autoincrement ids; `decrement_quantity` and `increment_quantity` also stamp `updated_at=utc_now()`. These values would differ if the same operation were re-executed independently on each replica.

**Not implemented:** restock creation and receipt; order cancellation and status transitions; any write path that uses `request_id` for de-duplication. The only direct database writers outside the repositories are the offline scripts `scripts/init_db.py` and `scripts/seed_data.py`.

---

## 9. Authentication and Sessions

[V] from `app/domain/auth.py`:

- Passwords: bcrypt, rejected above 72 bytes; a fixed decoy hash equalises timing for unknown users.
- Session tokens: `secrets.token_urlsafe(32)`; only an unsalted SHA-256 digest is stored in `sessions.token_hash`; the raw token exists only in the login response.
- Expiry: `INVENTORY_SESSION_TTL_MINUTES`, default 480 minutes. Expired and revoked sessions are rejected with one generic error.
- Logout revokes only the presented session.
- Every authenticated RPC passes through `ClientService._authenticated` (`client_service.py`).

Observations:

- **Server-side role enforcement was not found** [I]. Roles (`admin`, `manager`, `staff`) are stored and returned to the client, but `_authenticated` does not check them; role gating appears to be UI-level. The Streamlit files were not read line by line [NV].
- **State that exists only in memory:** `ClientService._llm_client` (a lazily built channel) and the Streamlit `session_state` that holds the user's token. The server keeps no in-memory session cache [V].
- **Behaviour after a leadership change:** sessions are persisted in each node's *local* SQLite, not replicated. A token issued by one node would be rejected by another node unless sessions are replicated or tokens are made stateless [I]. Because a session contains a random token and a wall-clock expiry, `login` cannot be replicated as a deterministic command without redesign.

---

## 10. LLM Integration

[V] from source:

- Chain: client, application server (`processBusinessRequest`), `LLMClient` (`app/api/llm_client.py`), `LLMService` (`llm/service.py`), `OllamaClient` (`llm/ollama_client.py`), Ollama over HTTP.
- Model: default `phi3:mini` (`OLLAMA_MODEL`); request timeout default 60 s (`LLM_REQUEST_TIMEOUT_SECONDS`); the application-to-LLM RPC adds a margin.
- Failure mapping: timeout to `DEADLINE_EXCEEDED`, unreachable to `UNAVAILABLE`, other errors to `INTERNAL`. The LLM server starts without Ollama and reports `UNAVAILABLE` per request.
- The model call is made outside the database transaction (`client_service.py:502-511`).
- Deterministic figures (daily rate, baseline forecast, baseline reorder quantity) are computed by the application server before the model is called and are returned beside the prose; every reply carries `advisory: true`.

**The LLM cannot directly modify authoritative inventory data.** The LLM service imports no database or storage code, `business_service.py` builders only read, and the model's text is never parsed into a quantity. `tests/unit/test_llm_no_side_effects.py` asserts that no table is written by any LLM operation [V].

Caveat for the replicated design: a follower's LLM answer would be built from that follower's local database and could be stale [I].

---

## 11. Existing Tests

- Runner: pytest 9.1.1. 1125 tests were collected (see Section 12) [V].
- Layout: `tests/unit/` (21 files) and `tests/integration/` (11 files plus `conftest.py`). There are **no Raft tests** [V].
- Categories, with the files in each [V]:
  - gRPC contract: `unit/test_proto_contract.py`
  - LLM: `test_prompts.py`, `test_ollama_client.py`, `test_llm_client.py`, `test_llm_service.py`, `test_llm_no_side_effects.py`, `integration/test_grpc_llm_service.py`, `integration/test_grpc_llm_client.py`
  - Client logic: `test_client_grpc_client.py`, `test_user_client_flow.py` (fake Streamlit)
  - Storage and domain: `test_database.py`, `test_models.py`, `test_products.py`, `test_warehouses.py`, `test_inventory.py`, `test_suppliers.py`, `test_sales_history.py`, `test_forecasting.py`, `test_analytics.py`, `test_seed_data.py`
  - Auth, service and orders: `test_auth.py`, `test_client_service.py`, `test_orders.py`, `test_business_service.py`
  - gRPC integration: `integration/test_grpc_client_service.py`, `test_grpc_products.py`, `test_grpc_warehouses.py`, `test_grpc_inventory.py`, `test_grpc_suppliers.py`, `test_grpc_business_request.py`, `test_phase_f_grpc.py`
  - End-to-end: `integration/test_milestone1_integration.py`

Database isolation, as verified from `tests/conftest.py` and the call sites:

- Every database-using test gets a temporary file-backed SQLite database under pytest's `tmp_path` (fixtures `db_engine` and `session_factory`, `conftest.py:139-153`). No in-memory database, environment-variable URL or hard-coded path is used.
- A search of `tests/` found no `session_scope()` or `init_db()` call without arguments, and no use of `SessionLocal`, `main()` or `serve()`. All `ClientService(...)` and `create_server(...)` calls receive the temporary session factory.
- Integration servers bind port 0 on `127.0.0.1`.
- Two session-scoped autouse guards run on every invocation: one compares the size and mtime of `data/node1.db` before and after the session (`conftest.py:47-70`); the other blocks construction of a real `ollama.Client` and any socket to port 11434 or a non-loopback host (`conftest.py:73-130`).
- No test requires a running server, Ollama or network access. No test calls any script's `main()`.

**Test-safety weaknesses** (recorded, not fixed):

1. The `node1.db` guard compares only the main database file. In WAL mode a stray write would land first in `node1.db-wal`, so it could go unnoticed [V from source; consequence inferred].
2. `test_default_database_path_is_project_relative` (`tests/unit/test_database.py`) fails if `INVENTORY_DB_PATH` is set [V].
3. A full run is slow because of bcrypt at cost 12; about 8 minutes of test time in this audit.

---

## 12. Actual Test Results

**Method.** The suite was *not* run inside the repository. The repository was copied (excluding `data/`, `.git`, `.venv` and caches; generated stubs included) to a scratchpad folder outside the repository, so that no code path could reach the real `data/node1.db`. All commands ran with `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`, from the copy. In the copy, `get_database_path()` resolved to the copy's own `data/node1.db` path, and no `data/` directory existed in the copy afterwards (so no stray default-engine use occurred) [V].

Commands (working directory: the isolated copy; `$py` is `AOS\venv\Scripts\python.exe`):

```powershell
& $py -m pytest --collect-only -q -p no:cacheprovider
& $py -m pytest -q -ra -p no:cacheprovider --basetemp <tmp> tests/unit/test_proto_contract.py tests/unit/test_prompts.py tests/unit/test_ollama_client.py tests/unit/test_llm_client.py tests/unit/test_llm_service.py tests/unit/test_client_grpc_client.py tests/unit/test_user_client_flow.py          # G1
& $py -m pytest -q -ra -p no:cacheprovider --basetemp <tmp> --durations=5 tests/unit/test_database.py tests/unit/test_models.py tests/unit/test_products.py tests/unit/test_warehouses.py tests/unit/test_inventory.py tests/unit/test_suppliers.py tests/unit/test_sales_history.py tests/unit/test_forecasting.py tests/unit/test_analytics.py tests/unit/test_seed_data.py          # G2
& $py -m pytest -q -ra -p no:cacheprovider --basetemp <tmp> --durations=5 tests/unit/test_auth.py tests/unit/test_client_service.py tests/unit/test_orders.py tests/unit/test_business_service.py tests/unit/test_llm_no_side_effects.py          # G3
& $py -m pytest -q -ra -p no:cacheprovider --basetemp <tmp> --durations=5 tests/integration --ignore=tests/integration/test_milestone1_integration.py          # G4
& $py -m pytest -q -ra -p no:cacheprovider tests/integration/test_milestone1_integration.py          # G5 (clean rerun, default basetemp)
& $py -m ruff check app llm client tests scripts --no-cache
& $py -m compileall -q app llm client tests scripts
```

Results [V, from this audit's runs]:

| Group | Tests run | Passed | Failed | Skipped | Errors | Duration |
|---|---|---|---|---|---|---|
| Collect-only | 1125 collected | - | - | - | 0 | 4.5 s |
| G1 contract, prompts, LLM, client logic | 337 | 337 | 0 | 0 | 0 | 3.5 s |
| G2 storage and domain | 416 | 416 | 0 | 0 | 0 | 78.5 s |
| G3 auth and order logic | 180 | 180 | 0 | 0 | 0 | 110.0 s |
| G4 in-process gRPC integration | 170 | 170 | 0 | 0 | 0 | 114.7 s |
| G5 end-to-end, first run (custom `--basetemp`) | 22 | 21 | **1** | 0 | 0 | 20.1 s |
| G5 end-to-end, rerun (default basetemp) | 22 | 22 | 0 | 0 | 0 | 15.7 s |
| `ruff check` | - | all checks passed | | | | |
| `compileall` | - | exit code 0 | | | | |

**Clean total: 1125 passed, 0 failed, 0 skipped, 0 errors** (337 + 416 + 180 + 170 + 22).

**The one failure.** `test_milestone1_integration.py::test_the_suite_runs_against_a_temporary_database` failed on the first G5 run at the assertion that the string "pytest" appears in the temporary database path (`test_milestone1_integration.py:772`). The auditor's custom `--basetemp` directory name did not contain "pytest". It is a harness-path artefact and not an application defect; the test passes with pytest's default base directory. The other 21 tests in that run passed.

**Coverage by area (all executed in this audit):**

- Application RPCs, authentication, inventory: PASSED (G3, G4, G5).
- Business correctness: PASSED, including order success, insufficient-stock rejection (`test_orders.py`, `test_phase_f_grpc.py`), negative/zero quantity rejection (`test_inventory.py`, `test_grpc_inventory.py`, `test_models.py`), rollback and atomicity (`test_orders.py`, `test_sales_history.py`, `test_database.py`, `test_warehouses.py`, `test_suppliers.py`), and concurrency (`test_concurrent_adds_do_not_lose_updates`, `test_concurrent_orders_write_exactly_one_history_row_each`).
- Supplier operations: creation and product association PASSED. **Restocking: NOT TESTED, not implemented.**
- LLM tests: PASSED, but **using fakes only**; real Ollama / Phi-3 behaviour is **NOT EXECUTED**.
- Live servers, smoke scripts, Streamlit start-up: **NOT EXECUTED.**

**Protection of the development database.** SHA-256 and timestamps of `data/node1.db`, `-wal` and `-shm` were identical before and after (prefixes `295175F3…`, `2C6C9EF4…`, `C488E1E4…`; sizes 167,936 / 78,312 / 32,768 bytes). `git status` was clean after the run, and no new `__pycache__` appeared in the repository [V].

**Overall Milestone-1 regression rating: PARTIAL.** Every available automated test passed after the one harness-path failure was correctly rerun; the rating is PARTIAL rather than PASS only because real-model, live-server and Streamlit checks were not run and restocking has nothing to test.

---

## 13. Existing Raft Scaffolding

| Item | Status | Evidence |
|---|---|---|
| `proto/raft.proto` | PLACEHOLDER (0 bytes) | [V] |
| `RequestVote`, `AppendEntries` | NOT IMPLEMENTED | [V] |
| `RaftNode`, Follower / Candidate / Leader | NOT IMPLEMENTED | [V] |
| `term`, `voted_for`, `commit_index`, election timeout, heartbeat | NOT IMPLEMENTED | [V] |
| `raft_log`, `raft_metadata` tables | NOT IMPLEMENTED | [V] |
| `raft/` package | PLACEHOLDER (empty `README.md`, no `__init__.py`) | [V] |
| `app/state_machines/commands.py` | PLACEHOLDER (0 bytes, no `__init__.py`) | [V] |
| `BusinessRequest.request_id` | IMPLEMENTED as a field only; not used for de-duplication | [V] `proto/client.proto` |

Raft is mentioned only in comments, docstrings and documentation (for example `ARCHITECTURE.md` section 10, `proto/client.proto` comments).

---

## 14. Phase 6 Integration Requirements

Phase 6 scope: three independent Raft node processes; Follower/Candidate/Leader roles; persistent `current_term` and `voted_for`; randomized election timeouts; `RequestVote`; `AppendEntries` heartbeats; higher-term handling; leader failure detection and re-election; controlled start-up and shutdown; automated election tests. Phase 6 does not replicate inventory state.

Existing code suitable for reuse [V locations, [P] use]:

| Reuse | Location | Intended use |
|---|---|---|
| Server skeleton | `llm/server.py`: `create_server` (l.81), `serve` (l.109), signal and grace handling (l.125-131) | Copy the pattern (returns `(server, bound_port)` unstarted; supports port 0). Preferred over `app/server.py` because it has no database dependency. |
| Env helper | `llm/server.py:_env_int` (l.61) | Duplicate rather than import, following the project's precedent for keeping nodes independent. |
| Outbound client pattern | `app/api/llm_client.py:LLMClient` (l.187) | Model for a per-peer client: lazy channel, per-call deadline, `close()`. |
| Stub generation | `scripts/generate_proto.ps1` | Unchanged; picks up `raft.proto` once it has content. |
| Test patterns | `tests/conftest.py` guards and `tmp_path`; `tests/integration/conftest.py:grpc_server`; `test_milestone1_integration.py:_nodes` | Multi-server in-process tests on ephemeral ports. |
| Logging | `logging.getLogger(__name__)`, `basicConfig` format, `INVENTORY_LOG_LEVEL` | Reuse. |

Not to be reused for Raft state: `session_factory` / `init_db` (would place Raft tables in the inventory database) and `app.server` (imports SQLAlchemy, the client service and the LLM client).

Standalone entry point `python -m raft.server` means `app/` is not modified in Phase 6. Integration into the application server is a later change.

**Proposed ports** [P] (current verified ports are in Section 5):

| Raft node | Address | State file |
|---|---|---|
| 1 | `127.0.0.1:50101` | `data/raft_node1.db` |
| 2 | `127.0.0.1:50102` | `data/raft_node2.db` |
| 3 | `127.0.0.1:50103` | `data/raft_node3.db` |

For a later multi-machine deployment the same Raft port can repeat per host and the bind address must change from loopback. Future application client ports should avoid consecutive numbering from 50051 because 50052 is the LLM (suggest 50051, 50061, 50071).

**Proposed environment variables** [P] (new `RAFT_` prefix; this deviates from the existing `INVENTORY_` and `LLM_` prefixes and should be confirmed): `RAFT_NODE_ID`, `RAFT_PEERS`, `RAFT_HOST`, `RAFT_PORT`, `RAFT_STATE_PATH`, `RAFT_HEARTBEAT_MS`, `RAFT_ELECTION_TIMEOUT_MIN_MS`, `RAFT_ELECTION_TIMEOUT_MAX_MS`. Suggested timings: demo 100 ms heartbeat with 500-1000 ms election timeout; tests 50 ms with 300-600 ms. Windows timer granularity (about 15 ms) and Python scheduling jitter argue against anything tighter.

**Proposed persistence** [P]: a separate SQLite file per Raft node (`data/raft_node<id>.db`) using the standard-library `sqlite3`, with one row `(current_term, voted_for)` and `synchronous=FULL`. This leaves the frozen inventory schema and `data/node1.db` untouched, gives atomic durable writes, and lets Phase 7 add `raft_log` to the same file. The name pattern matches the existing `.gitignore` rules (`data/*.db`, `data/*.db-wal`, `data/*.db-shm`); a nested folder would not match. Alternatives: a new `raft_metadata` table in the inventory database (a schema change requiring sign-off), or an atomically replaced JSON file. **This choice requires the project owner's decision before the storage step.**

Persistence rules: write term and vote in one transaction before replying to a vote request and before soliciting votes as a candidate; reset `voted_for` whenever the term increases; never hold the state lock across network calls; never point a Raft node at `data/node1.db`.

---

## 15. Compatibility and Correctness Risks

**Contract and tooling compatibility**

1. `client.proto` is unchanged; adding `raft.proto` as a new service and package (`inventory.raft.v1`) does not affect `test_proto_contract.py`.
2. Generated stubs are gitignored: every checkout must regenerate them with `scripts/generate_proto.ps1` (PowerShell and Windows-oriented). Its `-Clean` switch deletes existing stubs and should not be used casually.
3. A top-level package named `raft` could collide with a third-party distribution of the same name. Whether one is installed was **not verified** [NV]; `raft/` has no `__init__.py` today.
4. Existing tests that scan source for ports or imports (`test_llm_client.py` around l.514, `test_client_grpc_client.py` around l.591, `test_llm_no_side_effects.py`) could react to new modules. Their exact scan scope was **not verified** [NV]; the full suite should be re-run in the isolated copy after each Phase 6 step. Raft ports must not coincide with 50051, 50052 or 11434, and Raft code must not be placed where those scans look.

**Runtime and concurrency**

5. The application gRPC pool has 8 workers, and an LLM call can occupy a worker for up to about 60 s. Raft heartbeats must not share that pool; use a separate gRPC server and executor.
6. The database engine is process-global (`database.py:115`); three nodes on one machine require three processes, each with its own `INVENTORY_DB_PATH` when the application server is integrated later.
7. `serve()` installs signal handlers, which only work on the main thread; in-process test clusters must use `create_server`, not `serve`.
8. Channels are insecure and there is no authentication between nodes (acceptable for the coursework; to be stated in the viva).

**Raft protocol decisions to make explicitly** (affect liveness, not election safety)

9. No PreVote: a node cut off from the majority keeps incrementing its term and can disrupt a stable leader when it rejoins.
10. No CheckQuorum: a leader isolated from the majority does not step down until it hears a higher term.

**Safety hazard**

11. Loss or deletion of a node's Raft state file makes it forget its vote and can allow two votes in one term. Tests must keep state in temporary directories, and operations documentation must warn against deleting it.

**Later-phase design risks** (not Phase 6 blockers)

12. Order placement is non-deterministic (`uuid4` order numbers, `utc_now()` timestamps, autoincrement ids); the leader must fix these values before a command enters the log.
13. Sessions are node-local (Section 9).
14. Writes have no `request_id` de-duplication; `post` has no request id field.
15. All replicas must start from identical inventory state (empty or copies of one snapshot) or they will diverge. The Milestone-1 `node1.db` is a development database and should not be reused for any Raft node.
16. The `Warehouse` model docstring (`models.py:248`) says "One node owns one warehouse in Milestone 2", contradicting the chosen architecture in which every node replicates all warehouses.
17. No server-side role enforcement was found (Section 9).

---

## 16. Blockers and Warnings

**BLOCKER (must be resolved before Phase 6): none.** The absence of Raft code is the purpose of Phase 6 and is not a blocker.

**WARNING (address, but does not prevent isolated Raft development)**

- Persistence location needs the project owner's sign-off (Section 14) before the storage module is written.
- The 10 local-only commits exist on this machine only; push or otherwise back them up before branching.
- Use a separate gRPC server and thread pool for Raft (risk 5).
- Decide whether to implement PreVote and CheckQuorum or document their absence (risks 9 and 10).
- Check for a third-party `raft` package and add `raft/__init__.py` (risk 3).
- The `node1.db` guard only stats the main file; Raft tests must stay in `tmp_path` (Section 11).
- Windows timer resolution constrains how short test timeouts can safely be.
- Real Ollama / Phi-3, live-server smoke tests and Streamlit start-up have not been exercised in this audit (Section 2).

**LATER PHASE (Phase 7 or 8)**

- Replicated log and its persistence; commit index and state-machine application.
- Deterministic command construction on the leader; routing `post` writes through the log; `request_id` de-duplication.
- Session replication or stateless tokens; client leader redirect and retry; server-side role checks.
- Restock operations and their replication.
- Identical initial state for all replicas; correcting the stale `Warehouse` docstring.
- Docker files and Compose (currently empty).

**OPTIONAL**

- A `GetState` RPC for observing role, term and leader.
- PreVote; TLS between nodes; a `pytest.ini` with a slow-test marker; a cluster start/stop PowerShell script; a dotenv loader; declaring `pytest-cov` if coverage is wanted.

---

## 17. Recommended Phase 6 File Plan

All entries are proposals [P]; none exist yet.

| File | Responsibility and public interface | Existing dependencies | Tests required |
|---|---|---|---|
| `proto/raft.proto` (exists, empty) | Service `inventory.raft.v1.RaftService`: `RequestVote` and `AppendEntries` with the log-related fields present but unused in Phase 6; optional `GetState` | `scripts/generate_proto.ps1` | stub-import and contract test |
| `raft/__init__.py` | Package marker | - | - |
| `raft/config.py` | `RaftConfig` dataclass; `load_config(env)`; validates unique ids, self in peer list, election timeout at least about 5 times the heartbeat | stdlib | `tests/unit/test_raft_config.py` |
| `raft/storage.py` | `RaftStateStore.load()` returns `(term, voted_for)`; `.save(term, voted_for)`; rejects a decreasing term | stdlib `sqlite3` | `tests/unit/test_raft_storage.py` (round trip, restart, corrupt file) |
| `raft/node.py` | `RaftNode(config, store, peers, clock, rng)`; `start()`, `stop()`, `handle_request_vote()`, `handle_append_entries()`, `status()`; roles; one lock; ticker thread; injectable time and randomness | storage, peers, generated stubs | `tests/unit/test_raft_node.py` (deterministic, fake clock and peers) |
| `raft/peers.py` | `PeerClient(address)`: lazy channel, per-call deadline, no-reply on failure, `close()` | grpc, generated stubs; pattern of `app/api/llm_client.py` | unit tests with a fake stub |
| `raft/service.py` | `RaftService` servicer: translation only | node, generated stubs; pattern of `llm/service.py` | via integration tests |
| `raft/server.py` | `create_server(...)` returning `(server, bound_port)`; `serve()`; `main()`; run as `python -m raft.server` | pattern of `llm/server.py` | `tests/integration/test_raft_election.py` |
| `tests/integration/test_raft_election.py` | In-process three-node cluster on ephemeral ports | `tests/conftest.py` guards, `tmp_path` | see Section 18 |
| `tests/integration/test_raft_cluster_processes.py` | Real subprocesses, kill by PID, temporary state directories; marked slow | `subprocess` | see Section 18 |
| `scripts/raft_cluster.ps1` (optional) | `start` / `stop` / `status` for the three processes; touches only its own PIDs | PowerShell | manual |
| `raft/README.md` (exists, empty) | Documentation of configuration and run commands | - | - |

Recommended order: (1) decisions on persistence, timeouts and PreVote/CheckQuorum; (2) `raft.proto`, regenerate stubs, add a contract test; (3) `config.py`; (4) `storage.py`; (5) `node.py` rules using fake time and peers; (6) election loop and heartbeats with real threads; (7) `peers.py`, `service.py`, `server.py`; (8) in-process three-node tests; (9) process-level crash tests and optional script; (10) full regression in the isolated copy and documentation. After each step run `pytest`, `compileall` and `ruff` and report exact counts.

---

## 18. Phase 6 Acceptance Criteria

None of these tests exists yet. They are proposed acceptance tests [P].

**Safety properties** (must hold at all times):

1. **Single leader per term:** record every `(term, node)` leadership transition; no two different nodes ever lead the same term.
2. **Voting restrictions:** at most one vote per term; reject a request with a lower term; reject a candidate whose log is not at least as up-to-date (checked against a stubbed log, since Phase 6 logs are empty).
3. **Higher-term handling:** any message with a higher term makes the receiver adopt the term, become Follower and clear `voted_for`; a stale leader receiving a higher term steps down; a stale-term `AppendEntries` is rejected with the current term.
4. **Persistence:** after restart, `current_term` and `voted_for` are unchanged; a node that voted in term T cannot vote for a different candidate in T after restarting; terms never decrease.
5. **Loss of majority, safety half:** with two of three nodes down, the survivor never becomes Leader.

**Liveness properties** (timing-dependent; assert eventual conditions with generous deadlines, not fixed sleeps):

6. **Initial election:** exactly one leader is elected within a bounded time; the others become Followers that report that leader.
7. **Heartbeats:** with a stable leader, no new election and no term change over many election timeouts.
8. **Leader crash and re-election:** after the leader is stopped (in-process) and then killed (separate process), a new leader with a higher term appears within the bound; the former leader, restarted, becomes Follower.
9. **Node restart:** a restarted follower rejoins without disturbing the leader.
10. **Loss of majority, liveness half:** after the majority is restored, a leader is elected again.

Unit tests with a fake clock and fake peers prove the rules deterministically; integration tests with short real timeouts prove behaviour end to end; process-level tests prove crash behaviour. All Raft tests must use temporary directories, never `data/node1.db`, and the Milestone-1 suite must continue to pass 1125 of 1125 when re-run in the isolated copy.

---

## 19. Final Readiness Decision

**READY FOR PHASE 6: YES.**

Reasons:

- The Milestone-1 baseline is committed and the working tree is clean (`main`, `96af753`, tag `milestone-1`).
- 1125 of 1125 collected tests passed in an isolated run, with ruff and compileall clean and 0 skipped; the one first-run failure was a harness-path artefact and passed on rerun.
- `data/node1.db` and its sidecars were proven unchanged across all test execution.
- Phase 6 can be built as a self-contained `raft/` package with its own proto, ports and state file, without modifying the existing `client.proto`, the inventory schema, or `app/`.
- No blocking issue was found. The listed warnings are manageable inside Phase 6.

Condition: the project owner should decide the Raft state persistence location (Section 14) before the storage module is written, and should back up or push the 10 local-only commits before branching.

Unresolved items recorded honestly: real Ollama / Phi-3 behaviour, live-server smoke tests and Streamlit start-up were not exercised; the Streamlit files and parts of `forecasting.py`, `analytics.py` and `prompts.py` were not read in full; the existence of a third-party `raft` package and the exact scope of the source-scanning tests were not verified.
