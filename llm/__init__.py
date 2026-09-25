"""The LLM node: a gRPC service that answers questions about inventory.

This package runs as its own process, separate from the application server,
and is subject to two rules that the rest of the project depends on:

* **It never opens a database.** Nothing here imports :mod:`app.storage`. The
  facts an answer is grounded in arrive in the request, gathered from SQLite
  by the application server, so a model cannot read -- let alone change --
  inventory state.
* **It is advisory.** It returns prose. Every quantity the system acts on is
  computed deterministically by the application server, so a hallucinated
  number has no path into stock levels.
"""
