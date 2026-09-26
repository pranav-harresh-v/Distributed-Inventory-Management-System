# Distributed Inventory Management System

A multi-warehouse inventory system built for CS G623 (Advanced Operating
Systems). Services communicate exclusively over gRPC, each application node
owns a local SQLite database, and an LLM runs as its own service that answers
questions about inventory without ever touching the data itself.

This document describes **the system as it exists today**, at the end of
Milestone 1. Where something is planned but not built, it says so.

---

## 1. Project overview

Two Streamlit clients talk to one application server over gRPC. That server
owns the business rules and the database. When a question needs a language
model, the application server — never the client — calls a separate LLM
service over gRPC, which in turn calls a locally running Ollama.

The shape of that dependency is the point of the design. A client has no
database connection and no address for the LLM. The LLM has no database
connection and no way to change anything. Every write goes through one
authenticated, transactional path on the application server.

---

## 2. Milestone 1 features

Implemented and working:

- **Authentication and sessions** — bcrypt password hashing, SHA-256 session
  token hashes, 8-hour expiry.
- **gRPC service definitions** — two frozen protobuf contracts.
- **Products, warehouses, inventory, suppliers** — create, update and read.
- **Atomic order placement** — multi-line orders that succeed or fail whole.
- **Sales history** — written by every order, and read by forecasting.
- **LLM integration** — demand forecast, reorder recommendation, inventory
  summary, each answered by a separate service.
- **Admin Streamlit client** — full catalogue and stock management.
- **User Streamlit client** — browse, check availability, place orders.
- **Test suite** — 1125 tests, unit and integration.
- **Seed data** — deterministic, idempotent demo dataset.

Not built in Milestone 1, and not claimed anywhere in this document: Raft
consensus, replication, leader election, failure recovery, multi-node
clustering, restocking workflows, order history, role enforcement, and Docker
deployment. See §14 and §15.

---

## 3. Architecture

```
  Admin Streamlit :8501 ──┐
                          │
                          ├── gRPC ──> Application Server :50051
                          │                    │
  User Streamlit :8502 ───┘                    ├──> SQLite (local to the node)
                                               │
                                               └── gRPC ──> LLM Server :50052
                                                                  │
                                                                  └── HTTP ──> Ollama :11434
                                                                                    │
                                                                                    └──> phi3:mini
```

Three rules hold this together, and the test suite enforces all three:

1. **A client never reaches the database.** No module under `client/` imports
   SQLAlchemy, `sqlite3`, or anything from `app.storage`.
2. **A client never reaches the LLM.** No module under `client/` names port
   50052 or 11434, or imports `ollama`. AI answers arrive through the
   application server, which authenticates the caller first.
3. **The LLM never reaches the database.** No module under `llm/` imports
   SQLAlchemy. It receives facts in the request and returns prose.

Layering inside the application server:

```
gRPC API  ->  domain  ->  repositories  ->  SQLAlchemy 2.x  ->  SQLite
```

Both protobuf contracts are generic: `post` and `get` dispatch on a
`resource` string, and `processBusinessRequest` on an `operation` string.
Adding an operation costs a domain function and a handler registration — no
`.proto` change and no regenerated stubs.

---

## 4. Technology stack

| Area | Choice |
|---|---|
| Language | Python 3.12 |
| RPC | gRPC (`grpcio`, `grpcio-tools`), protobuf 3 syntax |
| Storage | SQLite via SQLAlchemy 2.x (declarative `Mapped[]` style) |
| Passwords | `bcrypt` |
| LLM runtime | Ollama running `phi3:mini` |
| LLM transport | `ollama` Python client over local HTTP |
| UI | Streamlit |
| Tests | pytest |
| Linting | ruff |

No REST framework is used anywhere. All inter-service communication is gRPC.

---

## 5. Repository structure

```
app/                        application server
  server.py                 entry point, binds :50051
  api/
    client_service.py       ClientService: login, logout, post, get,
                            processBusinessRequest; action routing tables
    business_service.py     the three AI operations
    llm_client.py           outbound gRPC client to the LLM node
  domain/
    auth.py                 hashing, sessions, expiry
    inventory.py            products, warehouses, stock, suppliers
    orders.py               atomic order placement
    forecasting.py          demand statistics, forecast + reorder contexts
    analytics.py            inventory summary context
  storage/
    models.py               11 tables
    database.py             engine, session_scope, PRAGMAs
    repositories.py         data access; flush only, never commit

llm/                        LLM service (its own process)
  server.py                 entry point, binds :50052
  service.py                LLMService servicer
  prompts.py                system prompt and the three analysis prompts
  ollama_client.py          the only file that imports `ollama`

client/                     Streamlit clients
  grpc_client.py            shared transport; no Streamlit import
  app.py                    admin client
  user_app.py               user client

proto/
  client.proto              inventory.client.v1.ClientService
  llm.proto                 inventory.llm.v1.LLMService

scripts/
  init_db.py                create the schema
  seed_data.py              deterministic demo data
  grpc_smoke_test.py        end-to-end smoke test
  demo_llm_queries.py       drives the LLM node directly
  generate_proto.ps1        regenerate the stubs

tests/
  unit/                     domain, storage, prompts, client logic
  integration/              real servers over real channels
```

Generated stubs (`proto/*_pb2.py`, `proto/*_pb2_grpc.py`) are gitignored;
regenerate them with `scripts/generate_proto.ps1`.

**Empty placeholders**, reserved for later work and containing no code today:
`app/domain/suppliers.py`, `app/state_machines/commands.py`, `llm/engine.py`,
`proto/inventory.proto`, `proto/raft.proto`, `docker-compose.yml`, and
`docker/*.Dockerfile`.

---

## 6. Windows prerequisites

Developed and verified on Windows 11 with Git Bash.

- **Python 3.12** (verified on 3.12.10) — <https://www.python.org/downloads/>
- **Ollama for Windows** — <https://ollama.com/download>
- **Git** (Git Bash is used for the shell examples below)
- **PowerShell** — only needed to regenerate protobuf stubs

Note for Git Bash users: `pkill` is unavailable. Stop servers with `Ctrl+C`
in their own terminal.

---

## 7. Virtual environment

In this project the virtual environment lives **outside** the repository, one
level up, so it is never touched by repository operations:

```
AOS/
  venv/                              <- the environment
  distributed-inventory-system/      <- this repository
```

Create it:

```bash
python -m venv ../venv
```

Activate it (Git Bash):

```bash
source ../venv/Scripts/activate
```

Every command in this document can also be run without activating, by calling
the interpreter directly from the repository root:

```bash
../venv/Scripts/python.exe -m pytest -q
```

---

## 8. Installing dependencies

```bash
pip install -r requirements.txt
```

Ten direct dependencies, each pinned: SQLAlchemy, bcrypt, grpcio,
grpcio-tools, protobuf, httpx, ollama, streamlit, pytest, ruff. Transitive
packages are resolved by pip.

Verify the install:

```bash
python -m pip check
```

---

## 9. Ollama setup

Ollama must be running locally and listening on its default port, 11434. On
Windows the desktop app usually starts it for you. To start it manually:

```bash
ollama serve
```

Confirm it is up:

```bash
ollama list
```

---

## 10. phi3:mini setup

```bash
ollama pull phi3:mini
```

The pull is idempotent, so it is safe to run when the model is already
present. To use a tag you already have instead, set `OLLAMA_MODEL` — for
example `OLLAMA_MODEL=phi3:latest`, which is the same 3.8B Mini model under
its default tag.

Generation on CPU takes a few seconds per query, and the first request after
starting Ollama also pays for loading the model.

---

## 11. Database initialization

```bash
python scripts/init_db.py
```

Creates all 11 tables at `data/node1.db` and prints the tables present.
Override the location with `INVENTORY_DB_PATH`.

To start completely clean, delete the database first. **This destroys any
existing demo data**, including orders placed during a previous demo:

```bash
rm -f data/node1.db data/node1.db-wal data/node1.db-shm
```

---

## 12. Seed data

```bash
python scripts/seed_data.py
```

Idempotent: every row is looked up by its natural key and inserted only when
absent, so running it twice produces exactly the same rows as running it
once. A second run prints *"Nothing to do; demo data was already present."*

A fresh seed creates:

| Data | Count |
|---|---|
| Accounts | 2 |
| Products | 5 |
| Warehouses | 2 |
| Inventory rows | 10 (2 low-stock, 1 near threshold, 7 normal) |
| Suppliers | 2 |
| Supplier/product links | 7 |
| Historical orders | 15 (`SEED-ORD-` prefixed) |
| Sales history rows | 25, across 15 distinct days |

The historical orders exist because `sales_history.order_id` is a NOT NULL
foreign key — history cannot be written without orders to attach it to. They
carry a `SEED-ORD-` prefix so they are never confused with the `ORD-` numbers
generated during a live demo. Their dates are relative to the moment you run
the seed, because forecasting looks back a window of days from now; a fixed
calendar date would slide out of that window and leave the forecast with
nothing to read.

---

## 13. Demo accounts

Both accounts are created by `scripts/seed_data.py`. Passwords are bcrypt
hashed before they reach the database; no plaintext password is ever stored.

| Account | Password | Role | Use |
|---|---|---|---|
| `admin` | `ChangeMe-Dev-Only-123` | `admin` | Admin client (§16) |
| `user1` | `User1-Dev-Only-123` | `staff` | User client (§17) |

**These are local demonstration credentials only.** They are written in plain
text in this repository, are intended for a single-machine demo, and must
never be reused anywhere real. Override them at seed time with
`SEED_ADMIN_PASSWORD` and `SEED_USER_PASSWORD`.

There is no signup flow. Accounts are created by the seed script only.

**Roles are recorded but not enforced.** `users.role` is stored and displayed,
but the server applies no authorization checks, so `user1` is not actually
restricted from admin operations at the API level. The user client simply
does not offer them. Role enforcement is not part of Milestone 1.

---

## 14. Starting the system

Five processes, each in its own terminal. Start them in this order.

**1. Ollama** (only needed for the AI features)

```bash
ollama serve
```

**2. LLM server**

```bash
python -m llm.server
```

Prints `LLMService listening on 127.0.0.1:50052`. It starts whether or not
Ollama is reachable; a runtime that is down surfaces as `UNAVAILABLE` on a
request rather than as a failure to boot.

**3. Application server**

```bash
python -m app.server
```

Prints `ClientService listening on 127.0.0.1:50051` and the database path.

**4. Admin Streamlit client**

```bash
streamlit run client/app.py --server.port 8501
```

**5. User Streamlit client**

```bash
streamlit run client/user_app.py --server.port 8502
```

Only the application server is required for everything except the AI
Insights page. Without the LLM server, the three analyses report
`UNAVAILABLE` and every other page keeps working.

To point a client at a different node:

```bash
INVENTORY_GRPC_HOST=10.0.0.4 INVENTORY_GRPC_PORT=50051 streamlit run client/user_app.py
```

---

## 15. Ports

| Port | Service | Protocol |
|---|---|---|
| 11434 | Ollama | HTTP |
| 50052 | LLM server | gRPC |
| 50051 | Application server | gRPC |
| 8501 | Admin Streamlit client | HTTP |
| 8502 | User Streamlit client | HTTP |

Only the LLM server may reach 11434. Only the application server may reach
50052. Both are checked by tests against the source.

---

## 16. Admin functionality

`streamlit run client/app.py --server.port 8501` — sign in as `admin`.

| Page | What it does | RPCs |
|---|---|---|
| Dashboard | Product, warehouse and low-stock counts; low-stock table across every warehouse | `get products`, `get warehouses`, `get inventory.low_stock` |
| Products | List, create, update | `get products`, `post product.create`, `post product.update` |
| Warehouses | List, create | `get warehouses`, `post warehouse.create` |
| Inventory | Per-warehouse stock, low-stock highlighting, set initial stock, add stock | `get inventory.warehouse`, `post inventory.set_initial_stock`, `post inventory.add_stock` |
| Orders | Place a multi-line order | `get warehouses`, `get products`, `post order.place` |
| Suppliers | List suppliers, create, link a product with cost and lead time | `get suppliers`, `post supplier.create`, `post supplier.associate_product` |
| AI Insights | The three analyses in §19 | `processBusinessRequest` |

No admin page computes a total, decides whether stock is sufficient, or
subtracts a quantity.

---

## 17. User functionality

`streamlit run client/user_app.py --server.port 8502` — sign in as `user1`.

- **Login** — username and password against the same `login` RPC the admin
  client uses. The session token is kept in Streamlit session state; the
  password is not stored after login.
- **Browse Products** — the catalogue with SKU, name, description and unit
  price, with a search box.
- **Availability** — pick a warehouse and see stock, reorder point and the
  server's low-stock flag. Nothing is calculated in the browser.
- **Place Order** — pick a warehouse, add one or more products with positive
  quantities to a basket, review it, and submit. Obvious input errors
  (missing quantity, zero or negative, non-integer, empty basket) are caught
  in the UI, but the server validates everything again and is the authority.
- **Insufficient stock** — a shortfall comes back as `FAILED_PRECONDITION` and
  is shown as a plain message naming the product and the numbers, for example
  *"Insufficient stock for 'SKU-1001': requested 99999, available 148."* The
  order is not claimed to have succeeded, the basket is kept so a quantity can
  be adjusted, and stock is re-read from the server.
- **After a successful order** — the server's confirmation (order number, line
  count, total, status), the lines that were submitted, and stock re-read from
  the server.
- **Logout** — revokes the session on the server first, then clears local
  state and returns to the login screen. Clearing only local state would leave
  the session row valid on the node until it expired.

**My Orders is not implemented.** The backend exposes no read operation that
can identify orders belonging to the authenticated user: `get` handlers do not
receive the caller's identity, and `OrderRepository` has no per-user query.
Building the page would require a backend change, which is out of scope for
Milestone 1. No order-history feature is claimed anywhere in this client.

---

## 18. Order atomicity and non-negative stock

Two invariants, enforced by the server and covered by tests.

**Order placement is atomic.** `place_order` runs inside a single
`session_scope`, which commits once on clean exit and rolls back on any
exception. Domain code and repositories flush but never commit. So an order
either lands whole — header, lines, stock decrements and sales rows — or
leaves no trace. A multi-line order whose second line is short does not leave
the first line's decrement behind.

**Stock can never go negative.** Every decrement is a guarded conditional
UPDATE:

```sql
UPDATE inventory SET quantity = quantity - :n
 WHERE warehouse_id = :w AND product_id = :p AND quantity >= :n
```

The availability test and the write are one statement, so two concurrent
orders cannot both pass a check and then both write. A row count of zero means
the units were not there, and the order is rejected with
`FAILED_PRECONDITION`. A CHECK constraint on the column backs this up, and
order lines are processed in a fixed order (ascending product id) so the same
request always touches rows in the same sequence.

---

## 19. LLM features

Three analyses, selected by an `operation` string:

- **`DEMAND_FORECAST`** — expected demand for one product in one warehouse,
  from sales history: daily rate, trend direction, a baseline forecast over a
  horizon, and a data-quality flag when history is sparse.
- **`REORDER_RECOMMENDATION`** — whether to restock one product and roughly
  how much, combining stock on hand, reorder point, recent demand and supplier
  lead times and costs.
- **`INVENTORY_SUMMARY`** — a management summary across the node: totals,
  stock value, low-stock and out-of-stock lines, sales metrics and supplier
  coverage.

The request path is always client → application server → LLM server → Ollama.
A client has no address for the LLM and cannot reach it directly.

**Every number is computed before the model is asked anything.** The
application server reads SQLite, calculates the daily rate, trend, baseline
forecast and baseline reorder quantity itself, and sends those figures to the
model as context. The reply carries the prose *and* the figures it was based
on, so a reader can tell a measured number from a written one.

---

## 20. The LLM cannot change anything

By construction, not by convention:

- The LLM service opens no database. No module under `llm/` imports
  SQLAlchemy — asserted by a test.
- It has no write API. Its only RPC, `getLLMAnswer`, returns a string.
- **Nothing parses the model's answer into a quantity.** Every figure the
  system acts on was computed deterministically before the model saw it, so a
  hallucinated number cannot reach stock.
- Every AI reply carries `advisory: true`, stated in the payload and shown in
  the UI.
- A dedicated test module asserts that running all three analyses writes to no
  table, creates no order, and leaves every stock level untouched.

Product and supplier names are user-supplied text and reach the prompt, so the
system prompt pins everything in the context block as data rather than
instructions.

---

## 21. Testing

```bash
pytest -q
```

**1125 tests**, all passing, in roughly 8–15 minutes (bcrypt at cost 12 is
deliberately slow).

Integration tests only:

```bash
pytest tests/integration -q
```

The six graded Milestone-1 scenarios:

```bash
pytest tests/integration/test_milestone1_integration.py -q
```

Linting and byte-compilation:

```bash
python -m ruff check app llm client tests scripts
```

```bash
python -m compileall -q app llm client tests scripts
```

The suite is hermetic by enforcement. Session-scoped fixtures fail the run if
anything constructs a real `ollama.Client`, opens a socket to port 11434 or
off-loopback, or modifies `data/node1.db`. Every test uses a temporary SQLite
file and an ephemeral port, so **the suite needs no Ollama, no model and no
running server**, and never touches your demo database.

---

## 22. End-to-end smoke test

With the application server running (and the LLM server, for the AI check):

```bash
python scripts/grpc_smoke_test.py
```

Checks LLM reachability (a diagnostic only), application server reachability,
login, an authenticated read, one AI analysis **through the application
server**, logout, that the revoked token is refused, and that a wrong password
is rejected. Prints `PASS`/`FAIL` per check and exits non-zero on any failure.

Without the LLM server running:

```bash
python scripts/grpc_smoke_test.py --skip-llm
```

Other flags: `--address`, `--llm-address`, `--llm-timeout`, `--username`,
`--password`.

To exercise the LLM service directly, bypassing the application server:

```bash
python scripts/demo_llm_queries.py
```

That script carries mock context, holds no database session and cannot change
inventory. Useful flags: `--only`, `--show-context`, `--timeout`, `--address`.

---

## 23. Demo workflow

A run-through that exercises everything built in Milestone 1.

**Setup**

```bash
python scripts/init_db.py && python scripts/seed_data.py
```

Start all five processes as in §14, then:

**1. Admin — the starting position.** Open <http://localhost:8501>, sign in as
`admin`. The Dashboard shows 5 products, 2 warehouses and 2 low-stock lines.
Inventory → `WH-001` shows SKU-1005 below its reorder point.

**2. Admin — AI Insights.** Run a **Demand forecast** for SKU-1001 in WH-001:
the seeded history gives 15 selling days and a rising trend, so the answer is
grounded rather than sparse. Run a **Reorder recommendation** for SKU-1005,
the low-stock item, and note the supplier options and lead times. Run an
**Inventory summary**. Expand *"The data the answer was based on"* on any of
them to show the deterministic figures the model was given.

**3. User — browse and order.** Open <http://localhost:8502>, sign in as
`user1`. Browse Products, check Availability for WH-001, then Place Order: add
two different products, review the basket, submit. The confirmation shows the
order number and the server-calculated total, and stock is re-read below it.

**4. User — the failure path.** Add a product with a quantity far beyond stock
and submit. The order is refused with the product and the real numbers, the
basket is kept, and stock is unchanged.

**5. Admin — confirm the effect.** Back on 8501, Inventory → the same
warehouse shows the decremented quantities from step 3. This is the same
database, reached through the same gRPC API, from a different client.

**6. Resilience.** Stop the LLM server with `Ctrl+C`. On 8501, AI Insights
reports `UNAVAILABLE`; every other page still works. Restart it and the
analyses work again.

**7. Verify from the command line.**

```bash
python scripts/grpc_smoke_test.py
```

---

## 24. Troubleshooting

**`Failed to bind to address 127.0.0.1:50051`** — something is already on that
port, usually an application server you forgot to stop. Stop it with `Ctrl+C`
in its terminal, or run on another port with `INVENTORY_GRPC_PORT=50061`.

**`Port 8501 is not available`** — a Streamlit client is already running.
Use `--server.port 8503`, or stop the existing one.

**Login fails with `UNAUTHENTICATED`** — the database has not been seeded, or
the admin password was overridden at seed time. Run
`python scripts/seed_data.py`, or pass `--password` to the smoke test.

**AI Insights reports `UNAVAILABLE`** — either the LLM server is not running
(`python -m llm.server`) or Ollama is not running (`ollama serve`). The smoke
test's step 0 tells you which.

**AI Insights reports `DEADLINE_EXCEEDED`** — generation outran the timeout.
Raise it with `LLM_REQUEST_TIMEOUT_SECONDS=180`, or use a smaller model.

**`model "phi3:mini" not found`** — run `ollama pull phi3:mini`, or set
`OLLAMA_MODEL` to a tag from `ollama list`.

**`ModuleNotFoundError: proto.client_pb2`** — the generated stubs are missing.
Regenerate them:

```bash
powershell -File scripts/generate_proto.ps1
```

**Forecast says the data is sparse** — the seeded history is dated relative to
when you ran the seed. If the database was seeded long ago, the history has
aged out of the 30-day window; re-seed on a clean database.

**A test run reports the node database was modified** — a test opened
`data/node1.db` instead of a temporary file. That guard is doing its job;
treat it as a bug in the test.

---

## 25. Milestone 1 scope

**In scope and complete:** project foundation, gRPC architecture,
authentication and sessions, products, warehouses, inventory, suppliers,
atomic orders, sales history, demand forecasting, reorder recommendations,
inventory analytics, the independent LLM server, both Streamlit clients, seed
data, and the test suite.

**Known limitations**, stated rather than hidden:

- Roles are stored and displayed but **not enforced**.
- **No order history** — there is no actor-aware read operation (§17).
- **No restocking workflow** — the `restock_orders` table exists in the schema
  and is unused.
- Supplier name uniqueness is an application rule, not a unique index, so it
  is not race-proof.
- Order numbers are random and server-generated; for Milestone 2 the leader
  will need to put the id inside the replicated command.
- **Docker is not set up.** `docker-compose.yml` and the files under `docker/`
  are empty placeholders. The system runs natively, as described in §14.

---

## 26. Milestone 2

None of the following exists in this repository today. There is no consensus
code, no replication, no clustering, and no fault tolerance. `proto/raft.proto`
and `app/state_machines/commands.py` are empty placeholders.

Milestone 2 will add:

- **Raft consensus** across multiple application nodes
- **Leader election**
- **Log replication** and majority commit
- **A replicated inventory state machine** applying committed commands
  deterministically
- **Failure recovery** — replicas rejoining and catching up
- **Multi-node consensus** for order placement and restocking

Milestone 1 was built with this in mind: business operations are
deterministic, order lines are processed in a fixed order, and every state
change already flows through a single transactional path that can become a
replicated log command.
