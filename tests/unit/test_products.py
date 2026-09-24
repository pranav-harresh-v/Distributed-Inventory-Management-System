"""Unit tests for the product catalogue (Phase F step 23).

Two levels are covered here: the domain rules in app/domain/inventory.py,
and the status-code mapping the servicer applies to them. End-to-end
coverage over a real channel lives in
tests/integration/test_grpc_products.py.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth, inventory
from app.storage.database import session_scope
from app.storage.models import Product, User
from app.storage.repositories import ProductRepository
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

VALID = {"sku": "SKU-100", "name": "Wireless Mouse", "unit_price": 24.99}


# --- helpers ---------------------------------------------------------------
def make_product(session_factory: sessionmaker[DbSession], **overrides: Any) -> Product:
    payload = {**VALID, **overrides}
    with session_scope(session_factory) as db:
        product = inventory.create_product(db, payload)
        db.flush()
        return product


def fetch(session_factory: sessionmaker[DbSession], sku: str) -> Product | None:
    with session_factory() as db:
        return ProductRepository(db).get_by_sku(sku)


class FakeContext:
    """ServicerContext stand-in whose abort raises, as grpc's does."""

    def __init__(self) -> None:
        self.code: grpc.StatusCode | None = None
        self.details: str | None = None

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.code = code
        self.details = details
        raise _Aborted(code, details)


class _Aborted(Exception):
    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(details)
        self.code = code
        self.details = details


@pytest.fixture
def service(session_factory: sessionmaker[DbSession]) -> ClientService:
    return ClientService(session_factory)


@pytest.fixture
def token(session_factory: sessionmaker[DbSession], auth_user: User) -> str:
    with session_scope(session_factory) as db:
        return auth.login(db, auth_user.username, TEST_PASSWORD).raw_token


def post(service: ClientService, token: str, resource: str, payload: dict[str, Any]):
    request = client_pb2.PostRequest(session_token=token, resource=resource)
    request.payload.update(payload)
    return service.post(request, FakeContext())


def get(service: ClientService, token: str, resource: str, params: dict[str, Any]):
    request = client_pb2.GetRequest(session_token=token, resource=resource)
    request.params.update(params)
    return service.get(request, FakeContext())


# --- create ----------------------------------------------------------------
def test_create_product_successfully(session_factory: sessionmaker[DbSession]) -> None:
    product = make_product(session_factory)
    assert product.id is not None

    stored = fetch(session_factory, "SKU-100")
    assert stored is not None
    assert stored.name == "Wireless Mouse"
    assert stored.unit_price == Decimal("24.99")
    assert stored.is_active is True
    assert stored.description is None


def test_create_accepts_all_optional_fields(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory, description="  Bluetooth mouse  ", is_active=False)
    stored = fetch(session_factory, "SKU-100")
    assert stored.description == "Bluetooth mouse"  # trimmed
    assert stored.is_active is False


def test_price_arrives_as_a_float_but_is_stored_exactly(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Struct carries numbers as doubles; the stored value must stay exact."""
    make_product(session_factory, unit_price=1234567.89)
    assert fetch(session_factory, "SKU-100").unit_price == Decimal("1234567.89")


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"name": "X", "unit_price": 1}, "missing sku"),
        ({"sku": "S", "unit_price": 1}, "missing name"),
        ({"sku": "S", "name": "X"}, "missing unit_price"),
        ({"sku": "  ", "name": "X", "unit_price": 1}, "blank sku"),
        ({"sku": "S", "name": "", "unit_price": 1}, "blank name"),
        ({"sku": "S", "name": "X", "unit_price": -1}, "negative price"),
        ({"sku": "S", "name": "X", "unit_price": "abc"}, "non-numeric price"),
        ({"sku": "S", "name": "X", "unit_price": True}, "bool price"),
        ({"sku": "S", "name": "X", "unit_price": None}, "null price"),
        ({"sku": "S", "name": "X", "unit_price": 1e12}, "price too large"),
        ({"sku": "S" * 65, "name": "X", "unit_price": 1}, "sku too long"),
        ({"sku": "S", "name": "X" * 256, "unit_price": 1}, "name too long"),
        ({"sku": "S", "name": 5, "unit_price": 1}, "non-string name"),
        (
            {"sku": "S", "name": "X", "unit_price": 1, "is_active": "yes"},
            "non-bool flag",
        ),
        ({"sku": "S", "name": "X", "unit_price": 1, "id": 4}, "id is not settable"),
        ({"sku": "S", "name": "X", "unit_price": 1, "colour": "red"}, "unknown field"),
    ],
)
def test_create_product_with_invalid_data(
    session_factory: sessionmaker[DbSession], payload: dict[str, Any], reason: str
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_product(db, payload)


def test_create_rejects_non_mapping_payload(
    session_factory: sessionmaker[DbSession],
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_product(db, ["not", "a", "mapping"])


def test_duplicate_sku_rejected(session_factory: sessionmaker[DbSession]) -> None:
    make_product(session_factory)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.create_product(db, {**VALID, "name": "Different name"})


def test_duplicate_sku_leaves_exactly_one_row(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory)
    with pytest.raises(inventory.DuplicateError), session_scope(session_factory) as db:
        inventory.create_product(db, {**VALID, "name": "Second"})

    with session_factory() as db:
        products = ProductRepository(db).list()
    assert len(products) == 1
    assert products[0].name == "Wireless Mouse"


# --- get -------------------------------------------------------------------
def test_get_existing_product_by_id(session_factory: sessionmaker[DbSession]) -> None:
    created = make_product(session_factory)
    with session_factory() as db:
        assert inventory.get_product(db, product_id=created.id).sku == "SKU-100"


def test_get_existing_product_by_sku(session_factory: sessionmaker[DbSession]) -> None:
    make_product(session_factory)
    with session_factory() as db:
        assert inventory.get_product(db, sku="SKU-100").name == "Wireless Mouse"


def test_get_nonexistent_product(session_factory: sessionmaker[DbSession]) -> None:
    with session_factory() as db:
        with pytest.raises(inventory.NotFoundError):
            inventory.get_product(db, product_id=4242)
        with pytest.raises(inventory.NotFoundError):
            inventory.get_product(db, sku="NOPE")


@pytest.mark.parametrize(
    "params",
    [{}, {"id": 1, "sku": "SKU-100"}, {"id": "abc"}, {"id": 1.5}, {"id": 0}, {"x": 1}],
)
def test_get_rejects_bad_identifier_params(
    session_factory: sessionmaker[DbSession], params: dict[str, Any]
) -> None:
    with session_factory() as db, pytest.raises(inventory.ValidationError):
        inventory.get_product_from_params(db, params)


# --- list ------------------------------------------------------------------
def test_list_products(session_factory: sessionmaker[DbSession]) -> None:
    for index in range(3):
        make_product(session_factory, sku=f"SKU-{index}", name=f"Item {index}")
    with session_factory() as db:
        assert len(inventory.list_products(db)) == 3


def test_list_is_empty_when_no_products(
    session_factory: sessionmaker[DbSession],
) -> None:
    with session_factory() as db:
        assert list(inventory.list_products(db)) == []


def test_list_ordering_is_deterministic(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Insertion order deliberately differs from alphabetical order."""
    for sku in ["SKU-Z", "SKU-A", "SKU-M"]:
        make_product(session_factory, sku=sku, name=f"Item {sku}")

    with session_factory() as db:
        first = [p.sku for p in inventory.list_products(db)]
        second = [p.sku for p in inventory.list_products(db)]
    assert first == second == ["SKU-Z", "SKU-A", "SKU-M"]  # primary-key order


def test_list_honours_active_only_and_paging(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory, sku="SKU-1", name="A")
    make_product(session_factory, sku="SKU-2", name="B", is_active=False)
    make_product(session_factory, sku="SKU-3", name="C")

    with session_factory() as db:
        assert [p.sku for p in inventory.list_products(db, {"active_only": True})] == [
            "SKU-1",
            "SKU-3",
        ]
        assert [p.sku for p in inventory.list_products(db, {"limit": 2})] == [
            "SKU-1",
            "SKU-2",
        ]
        assert [p.sku for p in inventory.list_products(db, {"offset": 2})] == ["SKU-3"]


@pytest.mark.parametrize("params", [{"limit": 501}, {"limit": "x"}, {"unknown": 1}])
def test_list_rejects_bad_params(
    session_factory: sessionmaker[DbSession], params: dict[str, Any]
) -> None:
    with session_factory() as db, pytest.raises(inventory.ValidationError):
        inventory.list_products(db, params)


# --- serialization ---------------------------------------------------------
def test_serialize_product_is_struct_safe(
    session_factory: sessionmaker[DbSession],
) -> None:
    """A Struct rejects Decimal and datetime, so neither may survive."""
    make_product(session_factory, description="A mouse")
    with session_factory() as db:
        payload = inventory.serialize_product(inventory.get_product(db, sku="SKU-100"))

    assert payload["unit_price"] == "24.99"  # string, not float
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"].endswith("+00:00")
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool, type(None)))

    struct = client_pb2.DataItem(id="1")
    struct.data.update(payload)  # would raise on a Decimal or datetime
    assert struct.data["sku"] == "SKU-100"


# --- update ----------------------------------------------------------------
def test_update_product(session_factory: sessionmaker[DbSession]) -> None:
    make_product(session_factory)
    with session_scope(session_factory) as db:
        inventory.update_product(
            db, {"sku_key": "SKU-100", "name": "Renamed", "unit_price": 30}
        )

    stored = fetch(session_factory, "SKU-100")
    assert stored.name == "Renamed"
    assert stored.unit_price == Decimal("30.00")


def test_update_by_id(session_factory: sessionmaker[DbSession]) -> None:
    created = make_product(session_factory)
    with session_scope(session_factory) as db:
        inventory.update_product(db, {"id": created.id, "is_active": False})
    assert fetch(session_factory, "SKU-100").is_active is False


def test_partial_update_preserves_unchanged_fields(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory, description="Original text", is_active=True)
    with session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-100", "name": "Only the name"})

    stored = fetch(session_factory, "SKU-100")
    assert stored.name == "Only the name"
    assert stored.description == "Original text"
    assert stored.unit_price == Decimal("24.99")
    assert stored.is_active is True
    assert stored.sku == "SKU-100"


def test_update_can_change_sku(session_factory: sessionmaker[DbSession]) -> None:
    make_product(session_factory)
    with session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-100", "sku": "SKU-RENAMED"})
    assert fetch(session_factory, "SKU-100") is None
    assert fetch(session_factory, "SKU-RENAMED") is not None


def test_update_to_a_taken_sku_is_rejected(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory, sku="SKU-1", name="First")
    make_product(session_factory, sku="SKU-2", name="Second")

    with pytest.raises(inventory.DuplicateError), session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-2", "sku": "SKU-1"})

    assert fetch(session_factory, "SKU-2").name == "Second"


def test_update_nonexistent_product(session_factory: sessionmaker[DbSession]) -> None:
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "GHOST", "name": "X"})

    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.update_product(db, {"id": 9999, "name": "X"})


def test_update_rejects_empty_change_set(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory)
    with (
        pytest.raises(inventory.ValidationError, match="No fields to update"),
        session_scope(session_factory) as db,
    ):
        inventory.update_product(db, {"sku_key": "SKU-100"})


def test_update_requires_an_identifier(
    session_factory: sessionmaker[DbSession],
) -> None:
    with pytest.raises(inventory.ValidationError), session_scope(session_factory) as db:
        inventory.update_product(db, {"name": "X"})


@pytest.mark.parametrize(
    "illegal",
    ["id", "created_at", "updated_at", "password_hash", "__class__", "quantity"],
)
def test_illegal_or_nonmutable_field_cannot_be_changed(
    session_factory: sessionmaker[DbSession], illegal: str
) -> None:
    """A caller must never reach an arbitrary attribute via setattr."""
    created = make_product(session_factory)
    before = fetch(session_factory, "SKU-100")
    original_created_at = before.created_at

    with pytest.raises(inventory.ValidationError), session_scope(session_factory) as db:
        inventory.update_product(db, {"id": created.id, illegal: "tampered"})

    after = fetch(session_factory, "SKU-100")
    assert after.created_at == original_created_at
    assert after.name == "Wireless Mouse"


def test_update_validates_values_not_just_names(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory)
    with pytest.raises(inventory.ValidationError), session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-100", "unit_price": -5})
    assert fetch(session_factory, "SKU-100").unit_price == Decimal("24.99")


# --- failure leaves no partial state ---------------------------------------
def test_failed_operation_leaves_no_partial_state(
    session_factory: sessionmaker[DbSession],
) -> None:
    """A create that succeeds then a failure in the same scope rolls back both."""
    make_product(session_factory, sku="SKU-EXISTING", name="Existing")

    with pytest.raises(inventory.DuplicateError), session_scope(session_factory) as db:
        inventory.create_product(
            db, {"sku": "SKU-NEW", "name": "New", "unit_price": 1.0}
        )
        # Same transaction; this duplicate must undo the row above too.
        inventory.create_product(
            db, {"sku": "SKU-EXISTING", "name": "Clash", "unit_price": 1.0}
        )

    with session_factory() as db:
        skus = [p.sku for p in ProductRepository(db).list()]
    assert skus == ["SKU-EXISTING"]


def test_failed_update_does_not_persist_partial_changes(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_product(session_factory)

    with pytest.raises(inventory.ValidationError), session_scope(session_factory) as db:
        inventory.update_product(db, {"sku_key": "SKU-100", "name": "Half applied"})
        inventory.update_product(db, {"sku_key": "SKU-100", "unit_price": -1})

    assert fetch(session_factory, "SKU-100").name == "Wireless Mouse"


# --- servicer status mapping -----------------------------------------------
def test_servicer_create_and_read_back(service: ClientService, token: str) -> None:
    response = post(service, token, "product.create", VALID)
    assert response.success is True
    assert "SKU-100" in response.message

    read = get(service, token, "product", {"sku": "SKU-100"})
    assert read.success is True
    assert len(read.items) == 1
    assert read.items[0].id == "1"
    assert read.items[0].data["unit_price"] == "24.99"


def test_servicer_list_returns_serialized_items(
    service: ClientService, token: str
) -> None:
    post(service, token, "product.create", VALID)
    post(
        service, token, "product.create", {**VALID, "sku": "SKU-200", "name": "Second"}
    )

    listed = get(service, token, "products", {})
    assert [item.data["sku"] for item in listed.items] == ["SKU-100", "SKU-200"]


@pytest.mark.parametrize(
    ("resource", "payload", "expected"),
    [
        ("product.create", {"name": "X"}, grpc.StatusCode.INVALID_ARGUMENT),
        (
            "product.update",
            {"sku_key": "GHOST", "name": "X"},
            grpc.StatusCode.NOT_FOUND,
        ),
        ("no.such.action", {}, grpc.StatusCode.INVALID_ARGUMENT),
    ],
)
def test_servicer_post_status_mapping(
    service: ClientService,
    token: str,
    resource: str,
    payload: dict[str, Any],
    expected: grpc.StatusCode,
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, resource, payload)
    assert excinfo.value.code == expected


def test_servicer_duplicate_maps_to_already_exists(
    service: ClientService, token: str
) -> None:
    post(service, token, "product.create", VALID)
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "product.create", VALID)
    assert excinfo.value.code == grpc.StatusCode.ALREADY_EXISTS


def test_servicer_get_missing_maps_to_not_found(
    service: ClientService, token: str
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        get(service, token, "product", {"sku": "GHOST"})
    assert excinfo.value.code == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize("resource", ["product.create", "product"])
def test_product_actions_still_require_authentication(
    service: ClientService, resource: str
) -> None:
    """Phase E session validation still gates every catalogue action."""
    with pytest.raises(_Aborted) as excinfo:
        if resource.startswith("product."):
            post(service, "bogus-token", resource, VALID)
        else:
            get(service, "bogus-token", resource, {"sku": "SKU-100"})
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED
