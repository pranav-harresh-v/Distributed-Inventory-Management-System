"""End-to-end smoke test for the Milestone 1 gRPC surface.

Drives a running deployment the way the Streamlit client does: log in, read
through an authenticated RPC, ask for an AI analysis, log out, then prove the
token is dead. It is a verification tool for a live system, not a substitute
for ``pytest`` -- the automated suite fakes the model and uses throwaway
databases; this one talks to whatever is actually running.

What it checks, in order::

    0. LLM node reachable          diagnostic only, see below
    1. application server reachable
    2. authenticated login
    3. a safe read through the application server
    4. an AI analysis through the application server
    5. logout, and the token dying with it

**The architecture is respected, not shortcut.** The AI check goes to the
*application server*, which authenticates the caller, reads its own SQLite
and then calls the LLM node itself::

    this script --gRPC :50051--> application server --gRPC :50052--> LLM node

Step 0 is the one exception and is deliberately limited: it opens a channel
to the LLM node to report whether it is listening, and makes **no service
call**. That is a diagnostic, so that "the AI check failed" can be told apart
from "the LLM node was never started". Nothing else here contacts :50052, and
nothing here opens a database. ``scripts/demo_llm_queries.py`` is the
complementary tool that *does* drive the LLM node directly, with mock
context.

**Read-only.** Every business operation used is a read. The script creates no
product, warehouse, stock row or order, so it is safe against a populated
demo database. The one unavoidable write is the session that logging in
issues, and step 5 revokes it.

Usage (with the servers already running)::

    python scripts/grpc_smoke_test.py
    python scripts/grpc_smoke_test.py --address 127.0.0.1:50051
    python scripts/grpc_smoke_test.py --skip-llm          # no LLM node running
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
import time
from pathlib import Path

# Make the project importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from proto import client_pb2, client_pb2_grpc
from scripts.seed_data import DEV_ADMIN_PASSWORD, DEV_ADMIN_USERNAME

DEFAULT_ADDRESS = "127.0.0.1:50051"
DEFAULT_LLM_ADDRESS = "127.0.0.1:50052"
CONNECT_TIMEOUT_SECONDS = 5.0

#: Generation on a CPU is slow, and the application server already allows the
#: LLM node its own timeout plus a margin. Too short a value here would abort
#: a request the backend is still happily serving.
DEFAULT_LLM_TIMEOUT_SECONDS = 180.0

#: The analysis used for the AI check. Chosen because it needs no arguments
#: and reads across every warehouse, so it works on any seeded database.
SMOKE_OPERATION = "INVENTORY_SUMMARY"

#: A read that exists on every deployment and mutates nothing.
SMOKE_RESOURCE = "products"

#: How many leading characters of a token are safe to show. The token is a
#: live credential, so it is never printed in full -- enough to correlate
#: two log lines, not enough to reuse.
TOKEN_PREVIEW_CHARS = 6

#: Answers can run long; the transcript shows the opening only.
ANSWER_PREVIEW_CHARS = 220


def redact(token: str) -> str:
    """Render a token for display without exposing it."""
    if not token:
        return "<empty>"
    return f"{token[:TOKEN_PREVIEW_CHARS]}...<redacted, {len(token)} chars>"


def status_of(call, *args, **kwargs) -> tuple[grpc.StatusCode, object]:
    """Invoke an RPC and report its status code instead of raising."""
    try:
        return grpc.StatusCode.OK, call(*args, **kwargs)
    except grpc.RpcError as exc:
        return exc.code(), exc.details()


class Checks:
    """Tiny pass/fail recorder so the script exits non-zero on failure."""

    def __init__(self) -> None:
        self.failures = 0
        self.passes = 0
        self.skipped = 0

    def record(self, description: str, ok: bool, detail: str = "") -> None:
        marker = "PASS" if ok else "FAIL"
        if ok:
            self.passes += 1
        else:
            self.failures += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{marker}] {description}{suffix}")

    def skip(self, description: str, reason: str) -> None:
        self.skipped += 1
        print(f"  [SKIP] {description}  ({reason})")

    def expect_code(
        self, description: str, actual: grpc.StatusCode, expected: grpc.StatusCode
    ) -> None:
        ok = actual == expected
        # Only explain on failure; a passing line stays quiet.
        detail = "" if ok else f"expected {expected.name}, got {actual.name}"
        self.record(description, ok, detail)


def check_llm_reachable(checks: Checks, address: str) -> bool:
    """Diagnostic only: is anything listening on the LLM node's port?

    Opens a channel and waits for it to become ready. It makes no RPC -- the
    application-level AI check below is the one that exercises the service,
    and it does so through the application server. The value of asking here
    is purely diagnostic: without it, a stopped LLM node and a broken
    application server produce the same failure later on.
    """
    print(f"0. LLM node reachable at {address}  [diagnostic]")
    with grpc.insecure_channel(address) as channel:
        try:
            grpc.channel_ready_future(channel).result(
                timeout=CONNECT_TIMEOUT_SECONDS
            )
        except grpc.FutureTimeoutError:
            checks.record(
                "LLM node is listening",
                False,
                "start it with: python -m llm.server",
            )
            return False
    checks.record("LLM node is listening", True)
    return True


def run(
    address: str,
    username: str,
    password: str,
    *,
    llm_address: str,
    skip_llm: bool,
    llm_timeout: float,
) -> int:
    checks = Checks()
    print(f"Milestone 1 smoke test against {address}")
    print(f"User: {username}\n")

    llm_up = False
    if skip_llm:
        print("0. LLM node reachable  [diagnostic]")
        checks.skip("LLM node is listening", "--skip-llm")
        print()
    else:
        llm_up = check_llm_reachable(checks, llm_address)
        print()

    with grpc.insecure_channel(address) as channel:
        print(f"1. application server reachable at {address}")
        try:
            grpc.channel_ready_future(channel).result(timeout=CONNECT_TIMEOUT_SECONDS)
        except grpc.FutureTimeoutError:
            checks.record("application server is listening", False)
            print(f"\nSMOKE TEST FAILED: could not connect to {address}.")
            print("Start the server first: python -m app.server")
            return 1
        checks.record("application server is listening", True)

        stub = client_pb2_grpc.ClientServiceStub(channel)

        # -- 2. Log in and capture the token (a local, never printed) ------
        print("\n2. login with the development seed user")
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

        # -- 3. A safe read through the application server -----------------
        #
        # Phase F routed `get` to real handlers, so this returns OK and rows.
        # Reading is all it does: nothing here writes.
        print(f"\n3. authenticated read ('{SMOKE_RESOURCE}')")
        code, result = status_of(
            stub.get,
            client_pb2.GetRequest(session_token=token, resource=SMOKE_RESOURCE),
        )
        checks.expect_code("get returns OK", code, grpc.StatusCode.OK)
        if code == grpc.StatusCode.OK:
            checks.record("read reports success", result.success is True)
            checks.record(
                "rows returned",
                len(result.items) > 0,
                f"{len(result.items)} product(s)"
                if result.items
                else "none -- seed the database: python scripts/seed_data.py",
            )

        # -- 4. An AI analysis, through the application server --------------
        #
        # Sent to :50051, not to the LLM node. The application server
        # authenticates, reads its own SQLite, computes the figures and makes
        # the :50052 call itself. Nothing here can reach the model directly.
        print(f"\n4. AI analysis through the application server ({SMOKE_OPERATION})")
        if skip_llm:
            checks.skip("AI analysis returns OK", "--skip-llm")
        elif not llm_up:
            checks.skip("AI analysis returns OK", "LLM node is not listening")
        else:
            print(f"   asking the model, up to {llm_timeout:g}s...")
            started = time.monotonic()
            code, result = status_of(
                stub.processBusinessRequest,
                client_pb2.BusinessRequest(
                    session_token=token, operation=SMOKE_OPERATION
                ),
                timeout=llm_timeout,
            )
            elapsed = time.monotonic() - started
            checks.expect_code("AI analysis returns OK", code, grpc.StatusCode.OK)

            if code == grpc.StatusCode.OK:
                data = dict(result.data)
                answer = str(data.get("answer", ""))
                checks.record("answer is not blank", bool(answer.strip()))
                checks.record(
                    "query type echoed",
                    data.get("query_type") == SMOKE_OPERATION,
                    str(data.get("query_type")),
                )
                checks.record(
                    "model attributed", bool(data.get("model")), str(data.get("model"))
                )
                # The deterministic half of the reply. Its presence is what
                # lets a reader tell a measured number from a written one.
                checks.record("deterministic context returned", bool(data.get("context")))
                checks.record(
                    "reply is marked advisory", data.get("advisory") is True
                )
                checks.record("request id returned", bool(result.request_id), result.request_id)
                print(f"\n   answered in {elapsed:.1f}s:")
                preview = answer.strip().replace("\n", " ")
                if len(preview) > ANSWER_PREVIEW_CHARS:
                    preview = preview[:ANSWER_PREVIEW_CHARS] + "..."
                print(f"   {preview}")
            else:
                print(f"         server said: {result}")
                if code == grpc.StatusCode.UNAVAILABLE:
                    print("         The LLM node answered the port but not the call.")
                    print("         Is Ollama running?  ollama list")
                elif code == grpc.StatusCode.DEADLINE_EXCEEDED:
                    print("         Generation outran the deadline; try --llm-timeout 300.")

        # -- 5. Log out, and prove the token died --------------------------
        print("\n5. logout")
        code, result = status_of(
            stub.logout, client_pb2.LogoutRequest(session_token=token)
        )
        checks.expect_code("logout returns OK", code, grpc.StatusCode.OK)
        if code == grpc.StatusCode.OK:
            checks.record("logout reports success", result.success is True)

        print("\n6. the revoked token is refused")
        code, detail = status_of(
            stub.get,
            client_pb2.GetRequest(session_token=token, resource=SMOKE_RESOURCE),
        )
        checks.expect_code(
            "revoked token is rejected", code, grpc.StatusCode.UNAUTHENTICATED
        )

        print("\n7. login with an incorrect password")
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
    print("-" * 68)
    summary = f"{checks.passes} passed, {checks.failures} failed"
    if checks.skipped:
        summary += f", {checks.skipped} skipped"

    if checks.failures:
        print(f"SMOKE TEST FAILED  ({summary})")
        return 1

    if checks.skipped:
        print(f"SMOKE TEST PASSED  ({summary})")
        print("The AI path was not exercised. Start the LLM node and re-run")
        print("without --skip-llm to verify it end to end.")
        return 0

    print(f"SMOKE TEST PASSED  ({summary})")
    print("Login, authenticated read, AI analysis and session revocation all")
    print("verified over gRPC, end to end.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--address",
        default=os.environ.get("INVENTORY_GRPC_ADDRESS", DEFAULT_ADDRESS),
        help=f"host:port of the application server (default: {DEFAULT_ADDRESS})",
    )
    parser.add_argument(
        "--llm-address",
        default=DEFAULT_LLM_ADDRESS,
        help=(
            "host:port of the LLM node, for the reachability diagnostic only "
            f"(default: {DEFAULT_LLM_ADDRESS})"
        ),
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="skip the LLM diagnostic and the AI analysis check",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        help=(
            "seconds to wait for the AI analysis "
            f"(default: {DEFAULT_LLM_TIMEOUT_SECONDS:g})"
        ),
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
    return run(
        args.address,
        args.username,
        args.password,
        llm_address=args.llm_address,
        skip_llm=args.skip_llm,
        llm_timeout=args.llm_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
