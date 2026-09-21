"""Developer smoke test for the Phase E gRPC surface.

Drives a running application server the way a real client will: log in, use
an authenticated RPC, log out, then prove the token is dead. This is a
throwaway verification tool, not the Streamlit client.

Usage (with the server already running)::

    python scripts/grpc_smoke_test.py
    python scripts/grpc_smoke_test.py --address 127.0.0.1:50051
    python scripts/grpc_smoke_test.py --username admin --password ...

Credentials default to the development seed account defined in
``scripts/seed_data.py`` (honouring ``SEED_ADMIN_PASSWORD``), so no
credential is duplicated here. Run ``python scripts/seed_data.py`` first if
the database has not been seeded.

Exits 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the project importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from proto import client_pb2, client_pb2_grpc
from scripts.seed_data import DEV_ADMIN_PASSWORD, DEV_ADMIN_USERNAME

DEFAULT_ADDRESS = "127.0.0.1:50051"
CONNECT_TIMEOUT_SECONDS = 5.0

#: How many leading characters of a token are safe to show. The token is a
#: live credential, so it is never printed in full -- enough to correlate
#: two log lines, not enough to reuse.
TOKEN_PREVIEW_CHARS = 6


def redact(token: str) -> str:
    """Render a token for display without exposing it."""
    if not token:
        return "<empty>"
    return f"{token[:TOKEN_PREVIEW_CHARS]}...<redacted, {len(token)} chars>"


def status_of(call, *args) -> tuple[grpc.StatusCode, object]:
    """Invoke an RPC and report its status code instead of raising."""
    try:
        return grpc.StatusCode.OK, call(*args)
    except grpc.RpcError as exc:
        return exc.code(), exc.details()


class Checks:
    """Tiny pass/fail recorder so the script exits non-zero on failure."""

    def __init__(self) -> None:
        self.failures = 0

    def record(self, description: str, ok: bool, detail: str = "") -> None:
        marker = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{marker}] {description}{suffix}")

    def expect_code(
        self, description: str, actual: grpc.StatusCode, expected: grpc.StatusCode
    ) -> None:
        ok = actual == expected
        # Only explain on failure; a passing line stays quiet.
        detail = "" if ok else f"expected {expected.name}, got {actual.name}"
        self.record(description, ok, detail)


def run(address: str, username: str, password: str) -> int:
    checks = Checks()
    print(f"Phase E smoke test against {address}")
    print(f"User: {username}\n")

    with grpc.insecure_channel(address) as channel:
        try:
            grpc.channel_ready_future(channel).result(timeout=CONNECT_TIMEOUT_SECONDS)
        except grpc.FutureTimeoutError:
            print(f"  [FAIL] Could not connect to {address}.")
            print("         Start the server first: python -m app.server")
            return 1

        stub = client_pb2_grpc.ClientServiceStub(channel)

        # 1-3. Log in and capture the token (kept in a local, never printed).
        print("1. login with the development seed user")
        code, result = status_of(
            stub.login, client_pb2.LoginRequest(username=username, password=password)
        )
        checks.expect_code("login returns OK", code, grpc.StatusCode.OK)
        if code != grpc.StatusCode.OK:
            print(f"         server said: {result}")
            print("         Has the database been seeded? python scripts/seed_data.py")
            return 1

        token = result.session_token
        checks.record("response reports success", result.success is True)
        checks.record("session token issued", bool(token), redact(token))
        checks.record(
            "user details returned",
            bool(result.user.username),
            f"id={result.user.id} {result.user.username} role={result.user.role}",
        )
        checks.record(
            "expiry returned",
            result.HasField("expires_at"),
            f"{result.expires_at.ToDatetime().isoformat()}Z",
        )

        # 4-5. A protected RPC must pass authentication and stop at the
        # Phase E placeholder -- UNIMPLEMENTED, not UNAUTHENTICATED.
        print("\n2. protected RPC with the valid token")
        code, detail = status_of(
            stub.get,
            client_pb2.GetRequest(session_token=token, resource="inventory"),
        )
        checks.expect_code(
            "get reaches UNIMPLEMENTED (so auth succeeded)",
            code,
            grpc.StatusCode.UNIMPLEMENTED,
        )

        # 6. Log out.
        print("\n3. logout")
        code, result = status_of(
            stub.logout, client_pb2.LogoutRequest(session_token=token)
        )
        checks.expect_code("logout returns OK", code, grpc.StatusCode.OK)
        if code == grpc.StatusCode.OK:
            checks.record("logout reports success", result.success is True)

        # 7-8. The same token must now be dead.
        print("\n4. protected RPC with the logged-out token")
        code, detail = status_of(
            stub.get,
            client_pb2.GetRequest(session_token=token, resource="inventory"),
        )
        checks.expect_code(
            "revoked token is rejected", code, grpc.StatusCode.UNAUTHENTICATED
        )

        # 9. Wrong password.
        print("\n5. login with an incorrect password")
        code, detail = status_of(
            stub.login,
            client_pb2.LoginRequest(username=username, password="definitely-wrong"),
        )
        checks.expect_code(
            "bad password is rejected", code, grpc.StatusCode.UNAUTHENTICATED
        )
        checks.record(
            "failure message is generic",
            isinstance(detail, str) and "password" in detail.lower(),
            str(detail),
        )

    print()
    if checks.failures:
        print(f"SMOKE TEST FAILED: {checks.failures} check(s) did not pass.")
        return 1
    print("SMOKE TEST PASSED: Phase E login/session/logout flow verified.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--address",
        default=os.environ.get("INVENTORY_GRPC_ADDRESS", DEFAULT_ADDRESS),
        help=f"host:port of the application server (default: {DEFAULT_ADDRESS})",
    )
    parser.add_argument(
        "--username",
        default=DEV_ADMIN_USERNAME,
        help="development account to log in as (default: the seeded admin)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("SEED_ADMIN_PASSWORD", DEV_ADMIN_PASSWORD),
        help="password for that account (default: SEED_ADMIN_PASSWORD, else the seed default)",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    return run(args.address, args.username, args.password)


if __name__ == "__main__":
    raise SystemExit(main())
