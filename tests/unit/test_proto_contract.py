"""Tests for the generated protobuf/gRPC contract itself.

These guard the wire contract rather than any behaviour: if someone edits
proto/client.proto and regenerates, a dropped RPC, a renamed method or a
leaked credential field fails here instead of at runtime.

The stubs are gitignored build artifacts, so a failure to import usually
means codegen has not been run: scripts/generate_proto.ps1
"""

from __future__ import annotations

import pytest

REQUIRED_RPCS = ["login", "logout", "post", "get", "processBusinessRequest"]

REQUIRED_MESSAGES = [
    "UserInfo",
    "StatusResponse",
    "DataItem",
    "LoginRequest",
    "LoginResponse",
    "LogoutRequest",
    "PostRequest",
    "GetRequest",
    "GetResponse",
    "BusinessRequest",
    "BusinessResponse",
]

#: Any field whose name contains one of these must never appear on a
#: message the server sends back to a client.
SECRET_FIELD_MARKERS = ("password", "hash", "secret", "salt")


# --- 1. the generated modules import --------------------------------------
def test_protobuf_modules_import() -> None:
    from proto import client_pb2, client_pb2_grpc

    assert client_pb2.DESCRIPTOR.name == "proto/client.proto"
    assert client_pb2_grpc.ClientServiceStub is not None


def test_generated_code_declares_proto3_and_the_expected_package() -> None:
    from google.protobuf import descriptor_pb2

    from proto import client_pb2

    file_proto = descriptor_pb2.FileDescriptorProto()
    client_pb2.DESCRIPTOR.CopyToProto(file_proto)
    assert file_proto.syntax == "proto3"
    assert client_pb2.DESCRIPTOR.package == "inventory.client.v1"


def test_well_known_types_are_wired_up() -> None:
    """Struct/Timestamp must resolve, or payload fields silently break."""
    from proto import client_pb2

    request = client_pb2.BusinessRequest()
    request.payload.update({"nested": {"n": 1}, "flag": True})
    restored = client_pb2.BusinessRequest.FromString(request.SerializeToString())
    assert restored.payload["nested"]["n"] == 1
    assert restored.payload["flag"] is True


# --- 2. the service exposes exactly the required RPCs ---------------------
def test_service_exposes_every_required_rpc() -> None:
    from proto import client_pb2

    service = client_pb2.DESCRIPTOR.services_by_name["ClientService"]
    assert sorted(m.name for m in service.methods) == sorted(REQUIRED_RPCS)


@pytest.mark.parametrize("rpc", REQUIRED_RPCS)
def test_stub_and_servicer_expose_each_rpc(rpc: str) -> None:
    """The lowercase assignment names must survive code generation."""
    from proto import client_pb2_grpc

    assert hasattr(client_pb2_grpc.ClientServiceServicer, rpc)
    assert hasattr(client_pb2_grpc.ClientServiceStub, "__init__")
    service = __import__(
        "proto.client_pb2", fromlist=["DESCRIPTOR"]
    ).DESCRIPTOR.services_by_name["ClientService"]
    assert rpc in [m.name for m in service.methods]


def test_rpc_request_and_response_types() -> None:
    """Pins the assignment's return shapes, e.g. post -> status."""
    from proto import client_pb2

    service = client_pb2.DESCRIPTOR.services_by_name["ClientService"]
    actual = {m.name: (m.input_type.name, m.output_type.name) for m in service.methods}
    assert actual == {
        "login": ("LoginRequest", "LoginResponse"),
        "logout": ("LogoutRequest", "StatusResponse"),
        "post": ("PostRequest", "StatusResponse"),
        "get": ("GetRequest", "GetResponse"),
        "processBusinessRequest": ("BusinessRequest", "BusinessResponse"),
    }


def test_the_app_implements_every_declared_rpc() -> None:
    """The servicer must not leave a declared RPC unimplemented."""
    from app.api.client_service import ClientService
    from proto import client_pb2

    service = client_pb2.DESCRIPTOR.services_by_name["ClientService"]
    for method in service.methods:
        implementation = getattr(ClientService, method.name, None)
        assert implementation is not None, f"{method.name} is not implemented"
        # An unimplemented inherited stub would just set UNIMPLEMENTED and
        # return; ours must be defined on ClientService itself.
        assert method.name in vars(ClientService)


# --- messages --------------------------------------------------------------
@pytest.mark.parametrize("message", REQUIRED_MESSAGES)
def test_required_message_exists(message: str) -> None:
    from proto import client_pb2

    assert message in client_pb2.DESCRIPTOR.message_types_by_name


def test_login_response_carries_optional_token_presence() -> None:
    """`Optional(token)` needs real presence, not just an empty string."""
    from proto import client_pb2

    absent = client_pb2.LoginResponse(success=False, message="denied")
    assert absent.HasField("session_token") is False
    present = client_pb2.LoginResponse(success=True, session_token="")
    assert present.HasField("session_token") is True


# --- 16. no credential field exists on any response -----------------------
@pytest.mark.parametrize(
    "message",
    ["LoginResponse", "StatusResponse", "GetResponse", "BusinessResponse", "UserInfo"],
)
def test_no_response_message_declares_a_credential_field(message: str) -> None:
    """Structural guarantee: there is nowhere for a secret to be placed."""
    from proto import client_pb2

    fields = [
        f.name for f in client_pb2.DESCRIPTOR.message_types_by_name[message].fields
    ]
    leaks = [f for f in fields if any(m in f.lower() for m in SECRET_FIELD_MARKERS)]
    assert leaks == [], f"{message} exposes {leaks}"


def test_user_info_matches_the_safe_phase_d_columns() -> None:
    from proto import client_pb2

    fields = {
        f.name for f in client_pb2.DESCRIPTOR.message_types_by_name["UserInfo"].fields
    }
    assert fields == {"id", "username", "email", "role", "is_active"}
    assert "password_hash" not in fields
