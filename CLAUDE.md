# Distributed Inventory Management System

## Source of Truth
The professor's Distributed Inventory Management System assignment is the source of truth.

## Mandatory Constraints
- Python exclusively.
- All inter-service communication must use gRPC.
- No REST APIs between services.
- No Flask/FastAPI/Django backend.
- SQLite is local to each application node.
- Never use one shared database across distributed nodes.
- Client must never access the database directly.
- Client communicates only with the application server through gRPC.
- Application server communicates with the LLM server through gRPC.
- LLM server must never modify inventory/database state directly.

## LLM Runtime (Milestone 1)
- LLM runtime is Ollama running Phi-3 locally.
- The Python LLM service talks to Ollama over local HTTP, then exposes its own functionality to the application server via gRPC only.
- Do not use llama.cpp or llama-cpp-python.

## Milestone 1
Implement:
- project foundation
- gRPC architecture
- authentication and sessions
- products
- warehouses
- inventory
- suppliers
- orders
- restocking
- demand forecasting
- reorder recommendations
- inventory analytics
- independent LLM server
- Streamlit client
- tests
- Docker Compose

## Milestone 2
Raft consensus will be added later.

Do NOT implement Raft logic during Milestone 1.

However, keep business operations deterministic so they can later become Raft log commands.

## Correctness
- Stock must never become negative.
- Order placement must be atomic.
- Failed orders must not partially modify inventory.
- Passwords must be hashed.
- Sessions must be authenticated.
- LLM output must never directly mutate system state.