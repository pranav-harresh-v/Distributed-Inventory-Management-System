"""Streamlit client package.

Deliberately importable without Streamlit: :mod:`client.grpc_client` holds
the transport and has no UI dependency, so the client's logic can be tested
without launching a browser. Only :mod:`client.app` imports streamlit.
"""
