# Distributed Inventory Management System

## Project Specification

**Course:** Advanced Operating Systems (AOS)\
**Project:** Distributed Inventory Management System\
**Team Size:** 3\
**Current Phase:** Phase 0 --- Project Understanding & Architecture

------------------------------------------------------------------------

## 1. Problem Statement

Modern inventory systems may operate across multiple warehouses and must
process orders, restocking operations, and inventory updates correctly
even when multiple requests arrive concurrently or individual warehouse
nodes fail.

This project aims to build a distributed inventory management system
that initially provides correct inventory operations through gRPC and
LLM-assisted analytics, and is then extended into a multi-warehouse
distributed system using Raft consensus for replicated state, strong
consistency, and fault tolerance.

------------------------------------------------------------------------

## 2. Objectives

1.  Process customer orders while preventing negative inventory.
2.  Provide inventory operations through gRPC.
3.  Maintain inventory across multiple warehouse nodes.
4.  Use Raft consensus to replicate inventory-changing operations.
5.  Maintain consistent inventory state under concurrent orders.
6.  Coordinate supplier restocking operations through the distributed
    system.
7.  Provide demand prediction using mock historical sales data.
8.  Generate automated reorder recommendations.
9.  Generate summarized inventory analytics and reports using an LLM.
10. Demonstrate recovery from warehouse-node failures.
11. Measure and document correctness, consistency, latency, throughput,
    and recovery behavior.

------------------------------------------------------------------------

## 3. Official Milestones

### Milestone 1 --- Project Foundation, gRPC & LLM Integration

-   Order processing with correct stock decrement and prevention of
    negative stock.
-   LLM integration for demand prediction based on historical data
    (mock).
-   LLM integration for automated reorder suggestions.
-   LLM integration for summarized inventory analytics and reports.

### Milestone 2 --- Raft Consensus & Fault Tolerance

-   Multi-warehouse stock tracking with synchronized updates across
    nodes.
-   Supplier coordination for restocking, replicated via Raft.
-   Strong consistency for inventory state across all nodes under
    concurrent orders.
-   Fault tolerance and recovery for warehouse node crashes.

------------------------------------------------------------------------

## 4. Functional Requirements

### 4.1 Inventory

-   Add products.
-   Query product inventory.
-   Update inventory through valid operations.
-   Maintain non-negative stock.
-   Persist inventory state.

### 4.2 Orders

-   Accept valid orders.
-   Validate product existence and quantity.
-   Reject orders when requested quantity exceeds available stock.
-   Decrement stock atomically at the business-logic level.
-   Track order status.

### 4.3 Restocking

-   Create restock requests.
-   Increase inventory only for valid restock operations.
-   Track supplier/restock information.
-   In Milestone 2, replicate restock operations through Raft.

### 4.4 gRPC

-   Provide client-to-server inventory operations through gRPC.
-   Use Protocol Buffers as the service contract.
-   Use gRPC for required inter-node communication.

### 4.5 LLM / Analytics

-   Process mock historical sales data.
-   Produce demand estimates.
-   Generate reorder recommendations.
-   Summarize structured inventory analytics.
-   Keep LLM operations outside the consensus-critical state-transition
    path.

### 4.6 Distributed Consensus

-   Run a three-node warehouse cluster.
-   Support leader election.
-   Replicate inventory-changing commands.
-   Commit commands according to Raft majority rules.
-   Apply committed commands to the inventory state machine.
-   Recover nodes after failures.

------------------------------------------------------------------------

## 5. Non-Functional Requirements

### Correctness

The inventory invariant must always be preserved:

`stock >= 0`

### Consistency

Committed inventory-changing operations must be applied in a consistent
order across healthy replicas.

### Fault Tolerance

The cluster should continue making progress when a minority of nodes
fail, subject to Raft quorum requirements.

### Observability

The system should expose enough information to demonstrate:

-   current leader,
-   node state,
-   Raft term,
-   log/commit progress,
-   inventory state,
-   node failures and recovery.

### Reproducibility

Experiments should be repeatable using documented commands/scripts and
controlled test data.

------------------------------------------------------------------------

## 6. Core Design Principles

1.  Separate business logic from distributed consensus logic.
2.  Keep the Raft state machine deterministic.
3.  Do not place LLM calls in the consensus-critical path.
4.  Use explicit interfaces between modules.
5.  Build incrementally; do not implement the complete distributed
    system at once.
6.  Test every milestone before proceeding to the next major subsystem.
7.  Preserve raw/mock historical data separately from
    application-generated outputs.
8.  Every major design decision must be explainable during the AOS viva.

------------------------------------------------------------------------

## 7. Initial Technology Stack

  Component              Technology
  ---------------------- ---------------------------------------
  Language               Python 3
  RPC                    gRPC
  Interface definition   Protocol Buffers
  Consensus              Raft
  Persistent storage     SQLite
  LLM runtime            Ollama + local model
  Data processing        Pandas
  Testing                pytest
  Dashboard              Streamlit
  Visualization          Plotly
  Version control        Git + GitHub
  Environment            Python virtual environment
  Containerization       Docker, if useful during later stages

------------------------------------------------------------------------

## 8. Inventory Entities

### Product

-   `product_id`
-   `name`
-   `category`
-   `current_stock`
-   `reorder_threshold`
-   `safety_stock`

### Order

-   `order_id`
-   `product_id`
-   `quantity`
-   `warehouse_id`
-   `status`
-   `timestamp`

### Restock

-   `restock_id`
-   `product_id`
-   `quantity`
-   `supplier_id`
-   `warehouse_id`
-   `status`
-   `timestamp`

### Sales Record

-   `date`
-   `product_id`
-   `warehouse_id`
-   `quantity_sold`

------------------------------------------------------------------------

## 9. LLM Scope

The LLM subsystem has three main responsibilities:

### Demand Prediction

Use mock historical sales information to support demand estimation.

### Reorder Suggestions

Use structured inventory information and demand estimates to generate
useful reorder recommendations.

### Inventory Reporting

Convert structured analytics into human-readable summaries.

The LLM is advisory/analytical. It must not bypass deterministic
inventory validation or independently commit inventory-changing
operations.

------------------------------------------------------------------------

## 10. Raft Scope

Raft is responsible for distributed consensus and replication of
state-changing commands.

Responsibilities include:

-   follower/candidate/leader states,
-   election timeouts,
-   leader election,
-   terms,
-   heartbeats,
-   RequestVote RPC,
-   AppendEntries RPC,
-   replicated log,
-   commit index,
-   majority/quorum handling,
-   applying committed commands,
-   node recovery.

Raft is not responsible for demand prediction, natural-language
reporting, or business-level reorder policy.

------------------------------------------------------------------------

## 11. Failure Scenarios

The final system should demonstrate:

1.  Follower crash.
2.  Leader crash.
3.  New leader election.
4.  Continued operation with quorum available.
5.  Restart of a failed node.
6.  Catch-up/recovery of the restarted node.
7.  Concurrent orders for limited stock.
8.  Rejection of an order that would cause negative stock.
9.  Consistency of committed state across healthy nodes.

------------------------------------------------------------------------

## 12. Testing Strategy

### Unit Testing

Inventory, orders, restocking, prediction, and reorder logic.

### gRPC Testing

Client-server and required inter-node RPC behavior.

### Distributed Testing

Leader election, log replication, command commitment, and state-machine
application.

### Concurrency Testing

Concurrent orders against the same inventory item.

### Fault Injection

Controlled node shutdown/restart and leader failure.

### Performance Testing

Measure where meaningful:

-   request latency,
-   throughput,
-   replication/commit delay,
-   leader-election time,
-   recovery time,
-   behavior under concurrent load.

------------------------------------------------------------------------

## 13. Team Responsibilities

### Member 1 --- Inventory & gRPC

Primary ownership:

-   inventory business logic,
-   database layer,
-   order/restock operations,
-   protobuf definitions,
-   gRPC services and client.

### Member 2 --- Raft & Distributed Systems

Primary ownership:

-   Raft node,
-   leader election,
-   log replication,
-   consensus,
-   node-to-node communication,
-   failure recovery.

### Member 3 --- LLM, Analytics & Testing

Primary ownership:

-   historical data,
-   demand prediction,
-   reorder logic,
-   LLM reporting,
-   dashboard,
-   test scenarios and experiments.

All members must understand the overall architecture and core AOS
concepts.

------------------------------------------------------------------------

## 14. Development Phases

### Phase 0

Project understanding and architecture.

### Phase 1

Basic single-node inventory system.

### Phase 2

gRPC communication.

### Phase 3

Historical data and demand prediction.

### Phase 4

Automated reorder suggestions.

### Phase 5

Inventory analytics and LLM reports.

### Phase 6

Multi-warehouse architecture.

### Phase 7

Raft consensus.

### Phase 8

Replicated inventory state machine.

### Phase 9

Supplier coordination and replicated restocking.

### Phase 10

Concurrent orders and strong consistency.

### Phase 11

Fault tolerance and node recovery.

### Phase 12

Fault-injection experiments.

### Phase 13

Dashboard and final demonstration.

### Phase 14

Testing and performance evaluation.

### Phase 15

Documentation and viva preparation.

------------------------------------------------------------------------

## 15. Phase 0 Deliverables

-   `PROJECT_SPEC.md`
-   `ARCHITECTURE.md`
-   `README.md`
-   Git repository

Phase 0 must not implement Raft or complicated distributed logic.

------------------------------------------------------------------------

## 16. Definition of Done

Milestone 1 is complete when all four official Milestone 1 requirements
are demonstrably implemented and tested.

Milestone 2 is complete when all four official Milestone 2 requirements
are demonstrably implemented, tested, and supported by
fault-injection/consistency experiments.
