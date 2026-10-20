"""Raft consensus package for the Distributed Inventory Management System.

A separate package on purpose: it imports nothing from ``app``, ``client`` or
``llm`` (a test checks this), so a Raft node can start, hold elections and be
tested without the inventory application, its database or the LLM node.

Phase 6.1 provides the protocol (``proto/raft.proto``), validated configuration
(:mod:`raft.config`) and durable election state (:mod:`raft.storage`). Election
logic arrives in Phase 6.2 and log replication in Phase 7.
"""
