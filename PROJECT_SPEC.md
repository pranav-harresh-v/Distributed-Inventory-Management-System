# Distributed Inventory Management System

## Architecture

**Current Phase:** Phase 0 --- Project Understanding & Architecture

------------------------------------------------------------------------

## 1. Architecture Goals

The architecture is designed around four concerns:

1.  Inventory business logic.
2.  gRPC communication.
3.  Distributed consensus and replication.
4.  LLM-assisted analytics.

These concerns must remain modular so that each can be implemented and
tested independently.

------------------------------------------------------------------------

## 2. High-Level Architecture

``` text
                         +----------------+
                         |     Client     |
                         +-------+--------+
                                 |
                                gRPC
                                 |
                                 v
                    +-------------------------+
                    |    Raft Warehouse       |
                    |        Cluster          |
                    |                         |
                    | +-------+ +-------+    |
                    | |Node A | |Node B |    |
                    | |WH-001 | |WH-002 |    |
                    | +---+---+ +---+---+    |
                    |     \         /         |
                    |      \       /          |
                    |       +-----+           |
                    |       |Node C|           |
                    |       |WH-003|           |
                    |       +---+---+           |
                    +-----------+--------------+
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
                         | SQLite Store  |
                         +---------------+


             Historical Sales Data
                       |
                       v
                Demand Prediction
                       |
                       v
                Reorder Analysis
                       |
                       v
                LLM Reporting
```

------------------------------------------------------------------------

## 3. Logical Layers

### Layer 1 --- Client

Responsible for submitting requests and displaying responses.

Examples:

-   place order,
-   query inventory,
-   request restock,
-   request analytics.

### Layer 2 --- gRPC Service

Provides typed RPC interfaces between clients and nodes and, where
required, between cluster nodes.

### Layer 3 --- Raft

Provides:

-   leader election,
-   replicated log,
-   quorum-based commitment,
-   ordered state-changing commands,
-   recovery of replicas.

### Layer 4 --- Inventory State Machine

Applies committed commands deterministically.

Examples:

-   `PLACE_ORDER`
-   `RESTOCK`
-   other inventory-changing operations approved during implementation.

### Layer 5 --- Storage

SQLite provides local persistent storage for each node.

### Layer 6 --- Analytics / LLM

Consumes structured data and historical information for:

-   demand prediction,
-   reorder recommendations,
-   summarized reports.

LLM processing is outside the consensus-critical path.

------------------------------------------------------------------------

## 4. Warehouse Node Architecture

The target cluster contains three nodes:

``` text
Node A = Warehouse WH-001
Node B = Warehouse WH-002
Node C = Warehouse WH-003
```

Each node contains:

``` text
+--------------------------------+
| Warehouse Node                 |
|                                |
|  gRPC Server                   |
|       |                        |
|       v                        |
|  Raft Module <----> Other     |
|       |             Nodes      |
|       v                        |
|  Inventory State Machine       |
|       |                        |
|       v                        |
|  SQLite Storage                |
+--------------------------------+
```

Each node has its own local persistent database.

The cluster is logically one replicated system, but
physical/process-level storage is node-local.

------------------------------------------------------------------------

## 5. Why Three Nodes?

Three nodes provide a simple majority-based cluster:

``` text
Cluster = 3

Majority = 2
```

Therefore:

-   all 3 nodes available → normal operation,
-   1 node failed → quorum remains,
-   2 nodes failed → quorum is lost.

This makes leader failure and recovery easy to demonstrate while
remaining small enough for a student project.

------------------------------------------------------------------------

## 6. Database Architecture

Each warehouse node has its **own SQLite database**.

Conceptually:

``` text
Node A
  |
  +-- warehouse_A.db

Node B
  |
  +-- warehouse_B.db

Node C
  |
  +-- warehouse_C.db
```

We do not use one shared SQLite file across all nodes.

Instead, Raft replicates the logical state-changing commands, and each
node applies committed commands to its own local state/database.

This is essential to preserving the distributed-system model.

------------------------------------------------------------------------

## 7. Replication Model

The intended flow is:

``` text
Client
  |
  v
Leader
  |
  v
Create Raft Log Entry
  |
  +---------> Follower
  |
  +---------> Follower
  |
  v
Majority Acknowledgement
  |
  v
Commit Entry
  |
  v
Apply to State Machine
  |
  v
Update Local Database
```

Followers apply committed commands in the same committed order.

------------------------------------------------------------------------

## 8. Inventory State Machine

The state machine must be deterministic.

For an order:

``` text
Input:
PLACE_ORDER(product_id, quantity)

Check:
current_stock >= quantity

If false:
    reject operation

If true:
    current_stock -= quantity
    commit/apply successful state transition
```

The invariant is:

``` text
current_stock >= 0
```

The exact command/response semantics will be finalized during
implementation.

------------------------------------------------------------------------

## 9. gRPC Architecture

There are two communication categories.

### Client-to-node

``` text
Client
   |
  gRPC
   v
Warehouse Node
```

Used for application operations.

### Node-to-node

``` text
Node A <----gRPC----> Node B
   |                    |
   +------gRPC----------+
            |
          Node C
```

Used for required Raft RPCs and cluster communication.

------------------------------------------------------------------------

## 10. Protobuf Design

The `.proto` contract will eventually define application messages such
as:

``` text
Product
Order
InventoryRequest
InventoryResponse
RestockRequest
RestockResponse
```

Raft messages will include concepts corresponding to:

``` text
RequestVoteRequest
RequestVoteResponse

AppendEntriesRequest
AppendEntriesResponse
```

The exact fields and RPC definitions will be frozen before
implementation of each subsystem.

------------------------------------------------------------------------

## 11. Raft Architecture

Target state model:

``` text
                 +-----------+
                 |  Leader   |
                 +-----+-----+
                       |
             AppendEntries / Replication
                 +-----+-----+
                 |           |
                 v           v
            +--------+   +--------+
            |Follower|   |Follower|
            | Node B |   | Node C |
            +--------+   +--------+
```

A node may be:

``` text
Follower
Candidate
Leader
```

Raft manages:

-   terms,
-   election timeout,
-   votes,
-   heartbeats,
-   log replication,
-   commitment,
-   leader changes,
-   recovery.

------------------------------------------------------------------------

## 12. LLM Architecture

LLM functionality is deliberately separated from the consensus path.

``` text
Historical Sales
       |
       v
Demand Analysis
       |
       v
Demand Estimate
       |
       +------------------+
       |                  |
       v                  v
Reorder Logic       Inventory Analytics
       |                  |
       +--------+---------+
                |
                v
           LLM Reporting
```

The LLM should not be responsible for enforcing:

``` text
stock >= 0
```

and should not directly commit Raft commands.

------------------------------------------------------------------------

## 13. Order Processing

Target distributed flow:

``` text
Client
  |
  | PlaceOrder
  v
Raft Leader
  |
  | replicate command
  v
Raft Followers
  |
  | majority
  v
Commit
  |
  v
Inventory State Machine
  |
  +---- enough stock? ----+
  |                       |
 Yes                      No
  |                       |
  v                       v
Decrease stock           Reject
```

The implementation must carefully define whether validation occurs
before proposal, during state-machine application, or through a
deterministic reservation/command protocol. The final choice must
preserve consistency and the stock invariant.

------------------------------------------------------------------------

## 14. Concurrent Orders

Example:

``` text
Initial stock = 10

Order A = 7
Order B = 6
```

The system must establish a single valid committed ordering.

Valid outcome:

``` text
A commits -> stock = 3
B is rejected -> stock = 3
```

or the reverse ordering:

``` text
B commits -> stock = 4
A is rejected -> stock = 4
```

Invalid outcome:

``` text
A and B both succeed
13 units sold from stock of 10
```

------------------------------------------------------------------------

## 15. Restocking Architecture

Target flow:

``` text
Low Stock
   |
   v
Reorder Analysis
   |
   v
Supplier Request
   |
   v
Raft Leader
   |
   v
Replicated Restock Command
   |
   v
Majority Commit
   |
   v
Inventory State Machine
   |
   v
Updated Stock
```

The supplier is initially simulated rather than integrated with an
external commercial supplier.

------------------------------------------------------------------------

## 16. Failure Architecture

### Follower Failure

``` text
Leader A
 /     \
B       C
X       OK
```

The cluster retains quorum and should continue operating.

### Leader Failure

``` text
A = Leader
A = CRASHED

B + C
  |
  v
Election
  |
  v
New Leader
```

### Recovery

``` text
Failed Node
     |
   Restart
     |
     v
Reconnect
     |
     v
Catch up missing log/state
     |
     v
Return to follower role
```

------------------------------------------------------------------------

## 17. Persistence Model

Each node persists enough local state to survive process restarts.

At minimum, the architecture must account for:

-   inventory state,
-   Raft log/state required for recovery,
-   node identity/configuration,
-   order/restock information where required.

Exact persistence schema will be finalized before implementation.

------------------------------------------------------------------------

## 18. Observability

The final system should expose/log:

-   node ID,
-   node role,
-   current leader,
-   current term,
-   log index,
-   commit index,
-   inventory values,
-   request status,
-   node failures,
-   recovery events.

This information will support debugging, experiments, and
demonstrations.

------------------------------------------------------------------------

## 19. Architecture Constraints

The following constraints are frozen for the project unless deliberately
revised and documented:

1.  Three warehouse nodes.
2.  Each node has its own local SQLite database.
3.  Raft replicates logical state-changing commands.
4.  Inventory state transitions are deterministic.
5.  LLM calls are outside the consensus-critical path.
6.  gRPC is used for RPC communication.
7.  Raft is not implemented during Phase 0.
8.  The system is developed incrementally.
9.  Every major component must have tests.
10. Fault scenarios must be experimentally demonstrable.

------------------------------------------------------------------------

## 20. Open Decisions Before Implementation

These should be finalized before the relevant implementation phase:

-   exact SQLite schema,
-   exact protobuf fields,
-   request routing strategy,
-   exact Raft log persistence format,
-   client behavior when contacting a follower,
-   idempotency/request IDs,
-   exact demand-prediction method,
-   exact local LLM model,
-   dashboard scope,
-   Docker usage.

These are implementation/design decisions, not reasons to start coding
Raft during Phase 0.
