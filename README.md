# Distributed Inventory Management System

A Distributed Inventory Management System developed as an Advanced
Operating Systems (AOS) project.

The project combines:

-   gRPC-based service communication
-   multi-warehouse inventory management
-   Raft consensus
-   replicated state machines
-   concurrency handling
-   fault tolerance and recovery
-   LLM-assisted demand analysis and reporting

------------------------------------------------------------------------

## Project Status

**Current Phase:** Phase 0 --- Project Understanding & Architecture

No Raft implementation or complicated distributed logic is being
developed during Phase 0.

------------------------------------------------------------------------

## Project Milestones

### Milestone 1 --- Project Foundation, gRPC & LLM Integration

The system must provide:

1.  Correct order processing with stock decrement.
2.  Prevention of negative stock.
3.  Demand prediction using mock historical data.
4.  Automated reorder suggestions.
5.  Summarized inventory analytics and reports using an LLM.

### Milestone 2 --- Raft Consensus & Fault Tolerance

The system must provide:

1.  Multi-warehouse stock tracking with synchronized updates.
2.  Supplier coordination for restocking, replicated via Raft.
3.  Strong consistency under concurrent orders.
4.  Fault tolerance and recovery from warehouse node crashes.

------------------------------------------------------------------------

## High-Level Architecture

``` text
                         +----------------+
                         |     Client     |
                         +-------+--------+
                                 |
                                gRPC
                                 |
                                 v
                    +-------------------------+
                    |    3-Node Raft Cluster  |
                    |                         |
                    | Node A   Node B   Node C|
                    | WH-001   WH-002   WH-003|
                    +------------+------------+
                                 |
                         Committed Commands
                                 |
                                 v
                    +-------------------------+
                    | Inventory State Machine |
                    +------------+------------+
                                 |
                                 v
                         +---------------+
                         | SQLite / Node |
                         +---------------+


 Historical Sales
        |
        v
 Demand Prediction
        |
        v
 Reorder Analysis
        |
        v
 LLM Reports
```

------------------------------------------------------------------------

## Technology Stack

-   **Python 3**
-   **gRPC**
-   **Protocol Buffers**
-   **Raft**
-   **SQLite**
-   **Ollama + local LLM**
-   **Pandas**
-   **pytest**
-   **Streamlit**
-   **Plotly**
-   **Git/GitHub**
-   **Docker** where useful

------------------------------------------------------------------------

## Repository Structure

The repository will evolve incrementally. The target structure is:

``` text
distributed-inventory-system/
|
+-- README.md
+-- PROJECT_SPEC.md
+-- ARCHITECTURE.md
|
+-- proto/
+-- client/
+-- server/
+-- inventory/
+-- raft/
+-- storage/
+-- llm/
+-- analytics/
+-- dashboard/
+-- tests/
+-- scripts/
+-- data/
+-- docs/
```

------------------------------------------------------------------------

## Development Roadmap

### Phase 0 --- Architecture

-   Requirements
-   Architecture
-   Technology decisions
-   Data model
-   gRPC design
-   LLM scope
-   Raft scope
-   Failure scenarios
-   Testing strategy
-   Team responsibilities

### Milestone 1

**Phase 1:** Basic inventory system\
**Phase 2:** gRPC communication\
**Phase 3:** Demand prediction\
**Phase 4:** Automated reorder suggestions\
**Phase 5:** Analytics and LLM reports

### Milestone 2

**Phase 6:** Multi-warehouse architecture\
**Phase 7:** Raft consensus\
**Phase 8:** Replicated inventory state machine\
**Phase 9:** Supplier coordination and restocking\
**Phase 10:** Concurrent orders and strong consistency\
**Phase 11:** Fault tolerance and recovery\
**Phase 12:** Fault-injection experiments\
**Phase 13:** Dashboard and final demo

### Finalization

**Phase 14:** Testing and performance evaluation\
**Phase 15:** Documentation and viva preparation

------------------------------------------------------------------------

## Core Inventory Invariant

The system must never allow:

``` text
stock < 0
```

For an order:

``` text
if requested_quantity <= current_stock:
    accept
else:
    reject
```

The final distributed implementation must preserve this invariant under
concurrent requests.

------------------------------------------------------------------------

## Distributed-System Model

The target cluster contains three warehouse nodes:

``` text
Node A -> Warehouse WH-001
Node B -> Warehouse WH-002
Node C -> Warehouse WH-003
```

Each node maintains its own local SQLite database.

The nodes form a logical replicated system using Raft. State-changing
commands are replicated and committed according to Raft's majority rules
before being applied to the inventory state machine.

------------------------------------------------------------------------

## LLM Scope

The LLM subsystem is responsible for:

-   demand analysis/prediction support,
-   reorder suggestions,
-   natural-language inventory summaries.

LLM calls are not part of the consensus-critical path.

Deterministic business logic remains responsible for inventory safety
and state transitions.

------------------------------------------------------------------------

## Running the LLM Demonstration

The LLM runs as its own gRPC service on port 50052, separate from the
application server on 50051. The application server reaches it over gRPC
and never talks to Ollama itself.

The automated test suite does **not** need any of this: it fakes the model
runtime, so `pytest` passes on a machine with no Ollama and no model
downloaded. The steps below are for seeing a real model answer.

### 1. Start Ollama

Ollama must be running locally and listening on its default port:

``` bash
ollama serve
```

On Windows and macOS the desktop app usually starts it already. Confirm it
is up:

``` bash
ollama list
```

### 2. Verify the model is available

``` bash
ollama pull phi3:mini
```

The pull is idempotent, so it is safe to run when the model is already
there. `ollama list` should then show the tag. To use a tag you already
have instead, set `OLLAMA_MODEL` (for example `phi3:latest`, which is the
same 3.8B Mini model under its default tag).

### 3. Start the LLM service

From the repository root, in its own terminal:

``` bash
python -m llm.server
```

It should log:

    LLMService listening on 127.0.0.1:50052
    Ollama runtime: http://127.0.0.1:11434
    Model: phi3:mini (timeout 60s)

The service starts whether or not Ollama is reachable; a runtime that is
down shows up as `UNAVAILABLE` on a request rather than as a failure to
boot.

### 4. Run the demonstration

In another terminal:

``` bash
python scripts/demo_llm_queries.py
```

This sends the three Milestone 1 analyses -- `DEMAND_FORECAST`,
`REORDER_RECOMMENDATION` and `INVENTORY_SUMMARY` -- over gRPC and prints
the request id, query type, model and answer for each.

Useful flags:

``` bash
python scripts/demo_llm_queries.py --only REORDER_RECOMMENDATION
python scripts/demo_llm_queries.py --show-context      # print what the model was sent
python scripts/demo_llm_queries.py --timeout 300       # slower CPUs
python scripts/demo_llm_queries.py --address host:port
```

The script carries mock context in the shape the application server builds
from SQLite, holds no database session, and cannot change any inventory
state. It exits 0 when every query is answered and 1 otherwise, and prints
what to check when the service cannot be reached.

Generation on a CPU takes a few seconds per query, and the first request
after starting Ollama also pays for loading the model.

------------------------------------------------------------------------

## Testing

Testing will cover:

-   inventory operations,
-   orders,
-   restocking,
-   gRPC,
-   demand prediction,
-   reorder recommendations,
-   Raft leader election,
-   log replication,
-   concurrent orders,
-   node failures,
-   node recovery,
-   state consistency,
-   performance.

------------------------------------------------------------------------

## Team

### Member 1 --- Inventory + gRPC

Inventory logic, SQLite, orders, restocking, protobuf, gRPC.

### Member 2 --- Raft + Distributed Systems

Raft node, election, replication, consensus, inter-node communication,
recovery.

### Member 3 --- LLM + Analytics + Testing

Historical data, demand prediction, reorder analysis, LLM reporting,
dashboard, experiments.

All members are responsible for understanding the complete architecture.

------------------------------------------------------------------------

## Development Philosophy

The project will be built incrementally.

We will not implement the complete distributed system in a single step.

Each phase must:

1.  Have a defined objective.
2.  Have explicit interfaces.
3.  Have tests.
4.  Produce demonstrable results.
5.  Be understood by the team before the next phase begins.

Claude Pro may be used extensively for implementation, but generated
code must be reviewed, tested, documented, and understood by the team.

------------------------------------------------------------------------

## Phase 0 Deliverables

-   [x] `PROJECT_SPEC.md`
-   [x] `ARCHITECTURE.md`
-   [x] `README.md`
-   [ ] Git repository initialized

------------------------------------------------------------------------

## Future Demonstration

The final demonstration should show:

1.  Normal order processing.
2.  Prevention of negative stock.
3.  Demand prediction.
4.  Reorder recommendation.
5.  LLM inventory summary.
6.  Three-node warehouse cluster.
7.  Raft leader election.
8.  Replicated inventory updates.
9.  Concurrent order handling.
10. Leader failure.
11. New leader election.
12. Failed-node recovery and synchronization.
