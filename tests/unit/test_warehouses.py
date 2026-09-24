"""Unit tests for warehouses (Phase F step 24).

Covers the domain rules in app/domain/inventory.py and the status codes the
servicer maps them onto. End-to-end coverage lives in
tests/integration/test_grpc_warehouses.py.
"""

from __future__ import annotations

from typing import Any

import grpc
import pytest
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth, inventory
from app.storage.database import session_scope
from app.storage.models import User, Warehouse
from app.storage.repositories import WarehouseRepository
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

VALID = {"code": "WH-001", "name": "North Depot", "location": "Bengaluru, IN"}


# --- helpers ---------------------------------------------------------------
def make_warehouse(
    session_factory: sessionmaker[DbSession], **overrides: Any
) -> Warehouse:
    payload = {**VALID, **overrides}
    with session_scope(session_factory) as db:
        warehouse = inventory.create_warehouse(db, payload)
        db.flush()
        return warehouse


def fetch(session_factory: sessionmaker[DbSession], code: str) -> Warehouse | None:
    with session_factory() as db:
        return WarehouseRepository(db).get_by_code(code)


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
def test_create_warehouse_successfully(
    session_factory: sessionmaker[DbSession],
) -> None:
    warehouse = make_warehouse(session_factory)
    assert warehouse.id is not None

    stored = fetch(session_factory, "WH-001")
    assert stored is not None
    assert stored.name == "North Depot"
    assert stored.location == "Bengaluru, IN"
    assert stored.is_active is True


def test_create_accepts_is_active_and_trims_text(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_warehouse(
        session_factory, code="  WH-002  ", name="  South  ", is_active=False
    )
    stored = fetch(session_factory, "WH-002")
    assert stored is not None
    assert stored.name == "South"
    assert stored.is_active is False


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"name": "N", "location": "L"}, "missing code"),
        ({"code": "C", "location": "L"}, "missing name"),
        ({"code": "C", "name": "N"}, "missing location"),
        ({"code": "  ", "name": "N", "location": "L"}, "blank code"),
        ({"code": "C", "name": "", "location": "L"}, "blank name"),
        ({"code": "C", "name": "N", "location": "  "}, "blank location"),
        ({"code": "C" * 33, "name": "N", "location": "L"}, "code too long"),
        ({"code": "C", "name": "N" * 256, "location": "L"}, "name too long"),
        ({"code": "C", "name": "N", "location": "L" * 256}, "location too long"),
        ({"code": 7, "name": "N", "location": "L"}, "non-string code"),
        ({"code": "C", "name": None, "location": "L"}, "null name"),
        ({"code": "C", "name": "N", "location": "L", "is_active": "yes"}, "bad flag"),
        ({"code": "C", "name": "N", "location": "L", "id": 3}, "id not settable"),
        ({"code": "C", "name": "N", "location": "L", "capacity": 10}, "unknown field"),
    ],
)
def test_create_warehouse_with_invalid_data(
    session_factory: sessionmaker[DbSession], payload: dict[str, Any], reason: str
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, payload)


def test_create_rejects_non_mapping_payload(
    session_factory: sessionmaker[DbSession],
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, ["not", "a", "mapping"])


def test_invalid_create_writes_nothing(
    session_factory: sessionmaker[DbSession],
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, {"code": "WH-X", "name": "N"})

    with session_factory() as db:
        assert WarehouseRepository(db).list() == []


# --- uniqueness ------------------------------------------------------------
def test_duplicate_code_rejected(session_factory: sessionmaker[DbSession]) -> None:
    make_warehouse(session_factory)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, {**VALID, "name": "Another depot"})


def test_duplicate_code_leaves_exactly_one_row(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_warehouse(session_factory)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, {**VALID, "name": "Second"})

    with session_factory() as db:
        warehouses = WarehouseRepository(db).list()
    assert len(warehouses) == 1
    assert warehouses[0].name == "North Depot"


def test_duplicate_name_is_allowed(session_factory: sessionmaker[DbSession]) -> None:
    """warehouses.name is NOT unique in the schema, so this must succeed."""
    make_warehouse(session_factory, code="WH-A")
    make_warehouse(session_factory, code="WH-B")

    with session_factory() as db:
        names = [w.name for w in WarehouseRepository(db).list()]
    assert names == ["North Depot", "North Depot"]


# --- get -------------------------------------------------------------------
def test_get_existing_warehouse_by_id(
    session_factory: sessionmaker[DbSession],
) -> None:
    created = make_warehouse(session_factory)
    with session_factory() as db:
        assert inventory.get_warehouse(db, warehouse_id=created.id).code == "WH-001"


def test_get_existing_warehouse_by_code(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_warehouse(session_factory)
    with session_factory() as db:
        assert inventory.get_warehouse(db, code="WH-001").name == "North Depot"


def test_get_missing_warehouse(session_factory: sessionmaker[DbSession]) -> None:
    with session_factory() as db:
        with pytest.raises(inventory.NotFoundError):
            inventory.get_warehouse(db, warehouse_id=4242)
        with pytest.raises(inventory.NotFoundError):
            inventory.get_warehouse(db, code="NOPE")


@pytest.mark.parametrize(
    "params",
    [{}, {"id": 1, "code": "WH-001"}, {"id": "abc"}, {"id": 0}, {"sku": "WH-001"}],
)
def test_get_rejects_bad_identifier_params(
    session_factory: sessionmaker[DbSession], params: dict[str, Any]
) -> None:
    with session_factory() as db, pytest.raises(inventory.ValidationError):
        inventory.get_warehouse_from_params(db, params)


# --- list ------------------------------------------------------------------
def test_list_warehouses(session_factory: sessionmaker[DbSession]) -> None:
    for index in range(3):
        make_warehouse(session_factory, code=f"WH-{index}")
    with session_factory() as db:
        assert len(inventory.list_warehouses(db)) == 3


def test_list_is_empty_when_none_exist(
    session_factory: sessionmaker[DbSession],
) -> None:
    with session_factory() as db:
        assert list(inventory.list_warehouses(db)) == []


def test_list_ordering_is_deterministic(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Insertion order deliberately differs from alphabetical order."""
    for code in ["WH-Z", "WH-A", "WH-M"]:
        make_warehouse(session_factory, code=code)

    with session_factory() as db:
        first = [w.code for w in inventory.list_warehouses(db)]
        second = [w.code for w in inventory.list_warehouses(db)]
    assert first == second == ["WH-Z", "WH-A", "WH-M"]  # primary-key order


def test_list_honours_active_only_and_paging(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_warehouse(session_factory, code="WH-1")
    make_warehouse(session_factory, code="WH-2", is_active=False)
    make_warehouse(session_factory, code="WH-3")

    with session_factory() as db:
        assert [
            w.code for w in inventory.list_warehouses(db, {"active_only": True})
        ] == [
            "WH-1",
            "WH-3",
        ]
        assert [w.code for w in inventory.list_warehouses(db, {"limit": 2})] == [
            "WH-1",
            "WH-2",
        ]
        assert [w.code for w in inventory.list_warehouses(db, {"offset": 2})] == [
            "WH-3"
        ]


@pytest.mark.parametrize("params", [{"limit": 501}, {"limit": "x"}, {"unknown": 1}])
def test_list_rejects_bad_params(
    session_factory: sessionmaker[DbSession], params: dict[str, Any]
) -> None:
    with session_factory() as db, pytest.raises(inventory.ValidationError):
        inventory.list_warehouses(db, params)


# --- serialization ---------------------------------------------------------
def test_serialize_warehouse_is_struct_safe(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_warehouse(session_factory)
    with session_factory() as db:
        payload = inventory.serialize_warehouse(
            inventory.get_warehouse(db, code="WH-001")
        )

    assert payload["code"] == "WH-001"
    assert isinstance(payload["created_at"], str)
    assert payload["created_at"].endswith("+00:00")
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool, type(None)))

    item = client_pb2.DataItem(id="1")
    item.data.update(payload)  # would raise on a datetime
    assert item.data["location"] == "Bengaluru, IN"


# --- rollback --------------------------------------------------------------
def test_rollback_on_failure_leaves_no_partial_state(
    session_factory: sessionmaker[DbSession],
) -> None:
    """A good write followed by a failure in the same scope undoes both."""
    make_warehouse(session_factory, code="WH-EXISTING")

    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.create_warehouse(db, {**VALID, "code": "WH-NEW"})
        inventory.create_warehouse(db, {**VALID, "code": "WH-EXISTING"})

    with session_factory() as db:
        assert [w.code for w in WarehouseRepository(db).list()] == ["WH-EXISTING"]


def test_rollback_does_not_affect_products(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Step 23 must keep working alongside step 24, including on rollback."""
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_product(
            db, {"sku": "SKU-1", "name": "Widget", "unit_price": 5.0}
        )
        inventory.create_warehouse(db, {"code": "WH-BAD"})  # missing fields

    with session_factory() as db:
        assert WarehouseRepository(db).list() == []
        from app.storage.repositories import ProductRepository

        assert ProductRepository(db).list() == []


# --- servicer status mapping -----------------------------------------------
def test_servicer_create_and_read_back(service: ClientService, token: str) -> None:
    response = post(service, token, "warehouse.create", VALID)
    assert response.success is True
    assert "WH-001" in response.message

    read = get(service, token, "warehouse", {"code": "WH-001"})
    assert read.success is True
    assert len(read.items) == 1
    assert read.items[0].id == "1"
    assert read.items[0].data["location"] == "Bengaluru, IN"


def test_servicer_list_returns_serialized_items(
    service: ClientService, token: str
) -> None:
    post(service, token, "warehouse.create", VALID)
    post(service, token, "warehouse.create", {**VALID, "code": "WH-002"})

    listed = get(service, token, "warehouses", {})
    assert [item.data["code"] for item in listed.items] == ["WH-001", "WH-002"]


@pytest.mark.parametrize(
    ("resource", "payload", "expected"),
    [
        ("warehouse.create", {"name": "X"}, grpc.StatusCode.INVALID_ARGUMENT),
        ("warehouse.update", VALID, grpc.StatusCode.INVALID_ARGUMENT),
    ],
)
def test_servicer_post_status_mapping(
    service: ClientService,
    token: str,
    resource: str,
    payload: dict[str, Any],
    expected: grpc.StatusCode,
) -> None:
    """`warehouse.update` is not a registered action in this step."""
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, resource, payload)
    assert excinfo.value.code == expected


def test_servicer_duplicate_maps_to_already_exists(
    service: ClientService, token: str
) -> None:
    post(service, token, "warehouse.create", VALID)
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "warehouse.create", VALID)
    assert excinfo.value.code == grpc.StatusCode.ALREADY_EXISTS


def test_servicer_get_missing_maps_to_not_found(
    service: ClientService, token: str
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        get(service, token, "warehouse", {"code": "GHOST"})
    assert excinfo.value.code == grpc.StatusCode.NOT_FOUND


@pytest.mark.parametrize("resource", ["warehouse.create", "warehouse", "warehouses"])
def test_warehouse_actions_require_authentication(
    service: ClientService, resource: str
) -> None:
    """Phase E session validation still gates every warehouse action."""
    with pytest.raises(_Aborted) as excinfo:
        if resource.endswith(".create"):
            post(service, "bogus-token", resource, VALID)
        else:
            get(service, "bogus-token", resource, {})
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


def test_products_and_warehouses_coexist(service: ClientService, token: str) -> None:
    """Step 23 routes must keep working now that step 24 shares the table."""
    post(service, token, "warehouse.create", VALID)
    post(
        service,
        token,
        "product.create",
        {"sku": "SKU-1", "name": "Widget", "unit_price": 9.99},
    )

    assert len(get(service, token, "warehouses", {}).items) == 1
    products = get(service, token, "products", {})
    assert len(products.items) == 1
    assert products.items[0].data["unit_price"] == "9.99"
