"""Unit tests for suppliers and supplier/product terms (Phase F step 26).

Note the difference in guarantee strength between the two tables, which
these tests make explicit: `supplier_products` has a real UNIQUE constraint
and three CHECKs, while `suppliers` has neither, so supplier-name
uniqueness is an application rule only.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import grpc
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.api.client_service import ClientService
from app.domain import auth, inventory
from app.storage.database import session_scope
from app.storage.models import Supplier, SupplierProduct, User
from app.storage.repositories import SupplierProductRepository, SupplierRepository
from proto import client_pb2
from tests.conftest import TEST_PASSWORD

VALID = {"name": "Nimbus Components"}
TERMS = {"unit_cost": 14.20, "lead_time_days": 5}


# --- fixtures --------------------------------------------------------------
@pytest.fixture
def catalogue(session_factory: sessionmaker[DbSession]) -> None:
    """Two products for supplier links to point at."""
    with session_scope(session_factory) as db:
        for sku in ["SKU-1", "SKU-2"]:
            inventory.create_product(
                db, {"sku": sku, "name": f"Item {sku}", "unit_price": 10.0}
            )


def make_supplier(
    session_factory: sessionmaker[DbSession], **overrides: Any
) -> Supplier:
    with session_scope(session_factory) as db:
        supplier = inventory.create_supplier(db, {**VALID, **overrides})
        db.flush()
        return supplier


def associate(
    session_factory: sessionmaker[DbSession],
    *,
    supplier: str = "Nimbus Components",
    sku: str = "SKU-1",
    **overrides: Any,
) -> SupplierProduct:
    payload = {"supplier_name": supplier, "sku": sku, **TERMS, **overrides}
    with session_scope(session_factory) as db:
        link = inventory.associate_supplier_product(db, payload)
        db.flush()
        return link


def read_link(
    session_factory: sessionmaker[DbSession], supplier_id: int = 1, product_id: int = 1
) -> SupplierProduct | None:
    with session_factory() as db:
        return SupplierProductRepository(db).get_by_supplier_and_product(
            supplier_id, product_id
        )


class FakeContext:
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


# --- create supplier -------------------------------------------------------
def test_create_supplier(session_factory: sessionmaker[DbSession]) -> None:
    supplier = make_supplier(session_factory)
    assert supplier.id is not None

    with session_factory() as db:
        stored = SupplierRepository(db).get_by_name("Nimbus Components")
    assert stored is not None
    assert stored.is_active is True
    assert stored.email is None


def test_create_supplier_with_contact_details(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_supplier(
        session_factory,
        email="  sales@nimbus.example.com  ",
        phone="+91-80-5550-0101",
        address="12 Industrial Layout",
        is_active=False,
    )
    with session_factory() as db:
        stored = SupplierRepository(db).get_by_name("Nimbus Components")
    assert stored.email == "sales@nimbus.example.com"  # trimmed
    assert stored.phone == "+91-80-5550-0101"
    assert stored.address == "12 Industrial Layout"
    assert stored.is_active is False


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({}, "missing name"),
        ({"name": "   "}, "blank name"),
        ({"name": ""}, "empty name"),
        ({"name": 7}, "non-string name"),
        ({"name": "N" * 256}, "name too long"),
        ({"name": "N", "email": "e" * 256}, "email too long"),
        ({"name": "N", "phone": "p" * 65}, "phone too long"),
        ({"name": "N", "is_active": "yes"}, "non-bool flag"),
        ({"name": "N", "id": 3}, "id not settable"),
        ({"name": "N", "rating": 5}, "unknown field"),
    ],
)
def test_invalid_supplier_rejected(
    session_factory: sessionmaker[DbSession], payload: dict[str, Any], reason: str
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_supplier(db, payload)

    with session_factory() as db:
        assert SupplierRepository(db).list() == []


def test_create_rejects_non_mapping_payload(
    session_factory: sessionmaker[DbSession],
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_supplier(db, ["not", "a", "mapping"])


def test_duplicate_supplier_rejected(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Application-level rule: suppliers has no unique index."""
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.create_supplier(db, {**VALID, "email": "other@example.com"})

    with session_factory() as db:
        assert len(SupplierRepository(db).list()) == 1


def test_supplier_name_uniqueness_is_not_enforced_by_the_schema(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Documents the real guarantee: the database would accept a duplicate.

    Unlike products.sku or warehouses.code, suppliers.name has no unique
    index, so the rule above lives in the domain layer only.
    """
    make_supplier(session_factory)
    with session_factory() as db:
        db.add(Supplier(name="Nimbus Components"))
        db.flush()  # no IntegrityError: the schema permits this
        assert len(SupplierRepository(db).list()) == 2
        db.rollback()


# --- list suppliers --------------------------------------------------------
def test_list_suppliers(session_factory: sessionmaker[DbSession]) -> None:
    for name in ["Alpha", "Beta", "Gamma"]:
        make_supplier(session_factory, name=name)
    with session_factory() as db:
        assert len(inventory.list_suppliers(db)) == 3


def test_list_is_empty_when_none_exist(
    session_factory: sessionmaker[DbSession],
) -> None:
    with session_factory() as db:
        assert list(inventory.list_suppliers(db)) == []


def test_list_ordering_is_deterministic(
    session_factory: sessionmaker[DbSession],
) -> None:
    """Insertion order deliberately differs from alphabetical order."""
    for name in ["Zeta", "Alpha", "Mu"]:
        make_supplier(session_factory, name=name)

    with session_factory() as db:
        first = [s.name for s in inventory.list_suppliers(db)]
        second = [s.name for s in inventory.list_suppliers(db)]
    assert first == second == ["Zeta", "Alpha", "Mu"]  # primary-key order


def test_list_honours_active_only_and_paging(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_supplier(session_factory, name="One")
    make_supplier(session_factory, name="Two", is_active=False)
    make_supplier(session_factory, name="Three")

    with session_factory() as db:
        assert [
            s.name for s in inventory.list_suppliers(db, {"active_only": True})
        ] == [
            "One",
            "Three",
        ]
        assert [s.name for s in inventory.list_suppliers(db, {"limit": 2})] == [
            "One",
            "Two",
        ]


# --- association -----------------------------------------------------------
def test_associate_supplier_and_product(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    link = associate(session_factory)
    assert link.id is not None

    stored = read_link(session_factory)
    assert stored is not None
    assert stored.supplier.name == "Nimbus Components"
    assert stored.product.sku == "SKU-1"


def test_association_can_use_ids(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    with session_scope(session_factory) as db:
        inventory.associate_supplier_product(
            db, {"supplier_id": 1, "product_id": 1, **TERMS}
        )
    assert read_link(session_factory) is not None


def test_optional_terms_use_schema_defaults(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory)
    stored = read_link(session_factory)
    assert stored.minimum_order_qty == 1
    assert stored.is_preferred is False
    assert stored.supplier_sku is None


def test_optional_terms_can_be_supplied(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(
        session_factory,
        supplier_sku="NC-MOU-01",
        minimum_order_qty=25,
        is_preferred=True,
    )
    stored = read_link(session_factory)
    assert stored.supplier_sku == "NC-MOU-01"
    assert stored.minimum_order_qty == 25
    assert stored.is_preferred is True


def test_missing_supplier_is_not_found(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.associate_supplier_product(
            db, {"supplier_name": "GHOST", "sku": "SKU-1", **TERMS}
        )


def test_missing_product_is_not_found(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    with pytest.raises(inventory.NotFoundError), session_scope(session_factory) as db:
        inventory.associate_supplier_product(
            db, {"supplier_name": "Nimbus Components", "sku": "GHOST", **TERMS}
        )


def test_duplicate_association_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory)

    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS},
        )

    with session_factory() as db:
        assert len(SupplierProductRepository(db).list()) == 1


def test_duplicate_association_does_not_overwrite_terms(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory, unit_cost=14.20, lead_time_days=5)

    with (
        pytest.raises(inventory.DuplicateError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                "unit_cost": 999.99,
                "lead_time_days": 99,
            },
        )

    stored = read_link(session_factory)
    assert stored.unit_cost == Decimal("14.20")
    assert stored.lead_time_days == 5


def test_same_supplier_may_serve_several_products(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory, sku="SKU-1")
    associate(session_factory, sku="SKU-2")

    with session_factory() as db:
        assert len(SupplierProductRepository(db).list()) == 2


def test_several_suppliers_may_serve_one_product(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory, name="Nimbus Components")
    make_supplier(session_factory, name="Vertex Peripherals")
    associate(session_factory, supplier="Nimbus Components")
    associate(session_factory, supplier="Vertex Peripherals")

    with session_factory() as db:
        assert len(SupplierProductRepository(db).list_for_product(1)) == 2


# --- price -----------------------------------------------------------------
def test_supplier_price_persisted_correctly(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory, unit_cost=182.40)
    assert read_link(session_factory).unit_cost == Decimal("182.40")


def test_price_survives_the_double_round_trip(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    """A Struct carries numbers as doubles; the stored value stays exact."""
    make_supplier(session_factory)
    associate(session_factory, unit_cost=1234567.89)
    assert read_link(session_factory).unit_cost == Decimal("1234567.89")


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        (12.345, "12.35"),  # half-up, not truncated to 12.34
        (12.344, "12.34"),
        (2.675, "2.68"),  # the classic binary-float trap, handled via str()
        (10, "10.00"),
        ("7.5", "7.50"),
        (0, "0.00"),
    ],
)
def test_price_rounding_is_explicit_and_half_up(
    session_factory: sessionmaker[DbSession],
    catalogue: None,
    supplied: Any,
    expected: str,
) -> None:
    """NUMERIC(12,2) holds two decimals; excess digits round half-up.

    Pinned deliberately so the behaviour is documented rather than silent.
    """
    make_supplier(session_factory)
    associate(session_factory, unit_cost=supplied)
    assert read_link(session_factory).unit_cost == Decimal(expected)


@pytest.mark.parametrize(
    "unit_cost", [-0.01, -100, "abc", None, True, float("inf"), 1e12]
)
def test_invalid_price_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None, unit_cost: Any
) -> None:
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                "unit_cost": unit_cost,
                "lead_time_days": 5,
            },
        )

    assert read_link(session_factory) is None


def test_missing_price_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {"supplier_name": "Nimbus Components", "sku": "SKU-1", "lead_time_days": 5},
        )


# --- lead time -------------------------------------------------------------
def test_supplier_lead_time_persisted_correctly(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory, lead_time_days=14)
    assert read_link(session_factory).lead_time_days == 14


def test_zero_lead_time_is_valid(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    """The schema CHECK is `>= 0`, so same-day supply is legal."""
    make_supplier(session_factory)
    associate(session_factory, lead_time_days=0)
    assert read_link(session_factory).lead_time_days == 0


@pytest.mark.parametrize("lead_time", [-1, -30, 1.5, "soon", None, True])
def test_invalid_lead_time_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None, lead_time: Any
) -> None:
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                "unit_cost": 5.0,
                "lead_time_days": lead_time,
            },
        )

    assert read_link(session_factory) is None


def test_missing_lead_time_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db, {"supplier_name": "Nimbus Components", "sku": "SKU-1", "unit_cost": 5.0}
        )


@pytest.mark.parametrize("moq", [0, -1])
def test_invalid_minimum_order_quantity_rejected(
    session_factory: sessionmaker[DbSession], catalogue: None, moq: int
) -> None:
    """The schema CHECK on minimum_order_qty is `> 0`."""
    make_supplier(session_factory)
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db,
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                **TERMS,
                "minimum_order_qty": moq,
            },
        )


# --- the database constraints, exercised directly --------------------------
def test_database_rejects_negative_cost_and_lead_time(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    """The CHECKs hold even when the domain layer is bypassed."""
    make_supplier(session_factory)
    for field, value in [
        ("unit_cost", Decimal("-0.01")),
        ("lead_time_days", -1),
        ("minimum_order_qty", 0),
    ]:
        row = {
            "supplier_id": 1,
            "product_id": 1,
            "unit_cost": Decimal("1.00"),
            "lead_time_days": 1,
        }
        row[field] = value
        with pytest.raises(IntegrityError), session_factory() as db:
            db.add(SupplierProduct(**row))
            db.flush()


def test_database_rejects_a_duplicate_pair(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory)
    with pytest.raises(IntegrityError), session_factory() as db:
        db.add(
            SupplierProduct(
                supplier_id=1,
                product_id=1,
                unit_cost=Decimal("1.00"),
                lead_time_days=1,
            )
        )
        db.flush()


# --- failure leaves no partial data ----------------------------------------
def test_association_failure_does_not_leave_partial_data(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    """A good link followed by a failure in the same scope undoes both."""
    make_supplier(session_factory)

    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.associate_supplier_product(
            db, {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS}
        )
        inventory.associate_supplier_product(
            db,
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-2",
                "unit_cost": -5,
                "lead_time_days": 1,
            },
        )

    with session_factory() as db:
        assert SupplierProductRepository(db).list() == []


def test_supplier_creation_rolls_back_with_its_transaction(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    with (
        pytest.raises(inventory.ValidationError),
        session_scope(session_factory) as db,
    ):
        inventory.create_supplier(db, {"name": "Rollback Co"})
        inventory.create_supplier(db, {"name": ""})  # invalid

    with session_factory() as db:
        assert SupplierRepository(db).list() == []


# --- serialization ---------------------------------------------------------
def test_serialize_supplier_is_struct_safe(
    session_factory: sessionmaker[DbSession],
) -> None:
    make_supplier(session_factory, email="a@b.com")
    with session_factory() as db:
        payload = inventory.serialize_supplier(
            inventory.get_supplier(db, name="Nimbus Components")
        )

    assert payload["name"] == "Nimbus Components"
    assert isinstance(payload["created_at"], str)
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool, type(None)))

    item = client_pb2.DataItem(id="1")
    item.data.update(payload)
    assert item.data["email"] == "a@b.com"


def test_serialize_supplier_product_is_struct_safe(
    session_factory: sessionmaker[DbSession], catalogue: None
) -> None:
    make_supplier(session_factory)
    associate(session_factory, unit_cost=14.20, lead_time_days=5)

    with session_factory() as db:
        link = SupplierProductRepository(db).get_by_supplier_and_product(1, 1)
        payload = inventory.serialize_supplier_product(link)

    assert payload["unit_cost"] == "14.20"  # exact string, never a double
    assert payload["lead_time_days"] == 5
    assert payload["sku"] == "SKU-1"
    for value in payload.values():
        assert isinstance(value, (str, int, float, bool, type(None)))

    item = client_pb2.DataItem(id="1")
    item.data.update(payload)
    assert item.data["unit_cost"] == "14.20"


# --- servicer status mapping -----------------------------------------------
def test_servicer_create_and_list(service: ClientService, token: str) -> None:
    response = post(service, token, "supplier.create", VALID)
    assert response.success is True
    assert "Nimbus Components" in response.message

    listed = get(service, token, "suppliers", {})
    assert [item.data["name"] for item in listed.items] == ["Nimbus Components"]
    assert listed.items[0].id == "1"


def test_servicer_associate(
    service: ClientService, token: str, catalogue: None
) -> None:
    post(service, token, "supplier.create", VALID)
    response = post(
        service,
        token,
        "supplier.associate_product",
        {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS},
    )
    assert response.success is True
    assert "14.20" in response.message
    assert "5-day" in response.message


@pytest.mark.parametrize(
    ("resource", "payload", "expected"),
    [
        ("supplier.create", {}, grpc.StatusCode.INVALID_ARGUMENT),
        (
            "supplier.associate_product",
            {"supplier_name": "GHOST", "sku": "SKU-1", **TERMS},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            "supplier.associate_product",
            {"supplier_name": "Nimbus Components", "sku": "GHOST", **TERMS},
            grpc.StatusCode.NOT_FOUND,
        ),
        (
            "supplier.associate_product",
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                "unit_cost": -1,
                "lead_time_days": 1,
            },
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
    ],
)
def test_servicer_post_status_mapping(
    service: ClientService,
    token: str,
    catalogue: None,
    resource: str,
    payload: dict[str, Any],
    expected: grpc.StatusCode,
) -> None:
    post(service, token, "supplier.create", VALID)
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, resource, payload)
    assert excinfo.value.code == expected


def test_servicer_duplicate_supplier_is_already_exists(
    service: ClientService, token: str
) -> None:
    post(service, token, "supplier.create", VALID)
    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "supplier.create", VALID)
    assert excinfo.value.code == grpc.StatusCode.ALREADY_EXISTS


def test_servicer_duplicate_association_is_already_exists(
    service: ClientService, token: str, catalogue: None
) -> None:
    post(service, token, "supplier.create", VALID)
    payload = {"supplier_name": "Nimbus Components", "sku": "SKU-1", **TERMS}
    post(service, token, "supplier.associate_product", payload)

    with pytest.raises(_Aborted) as excinfo:
        post(service, token, "supplier.associate_product", payload)
    assert excinfo.value.code == grpc.StatusCode.ALREADY_EXISTS


def test_errors_never_expose_sqlalchemy_internals(
    service: ClientService, token: str, catalogue: None
) -> None:
    post(service, token, "supplier.create", VALID)
    with pytest.raises(_Aborted) as excinfo:
        post(
            service,
            token,
            "supplier.associate_product",
            {
                "supplier_name": "Nimbus Components",
                "sku": "SKU-1",
                "unit_cost": -1,
                "lead_time_days": 1,
            },
        )
    details = excinfo.value.details.lower()
    for leak in ("sqlalchemy", "traceback", "select", "sqlite", "integrityerror"):
        assert leak not in details


@pytest.mark.parametrize("resource", ["supplier.create", "suppliers"])
def test_supplier_actions_require_authentication(
    service: ClientService, resource: str
) -> None:
    with pytest.raises(_Aborted) as excinfo:
        if resource == "supplier.create":
            post(service, "bogus-token", resource, VALID)
        else:
            get(service, "bogus-token", resource, {})
    assert excinfo.value.code == grpc.StatusCode.UNAUTHENTICATED


def test_earlier_steps_still_work(
    service: ClientService, token: str, catalogue: None
) -> None:
    """Steps 23-25 must keep working alongside step 26."""
    post(
        service,
        token,
        "warehouse.create",
        {"code": "WH-1", "name": "North", "location": "IN"},
    )
    post(
        service,
        token,
        "inventory.set_initial_stock",
        {"warehouse_code": "WH-1", "sku": "SKU-1", "quantity": 5},
    )
    post(service, token, "supplier.create", VALID)

    assert len(get(service, token, "products", {}).items) == 2
    assert len(get(service, token, "warehouses", {}).items) == 1
    assert len(get(service, token, "suppliers", {}).items) == 1
    stock = get(
        service, token, "inventory.stock", {"warehouse_code": "WH-1", "sku": "SKU-1"}
    )
    assert stock.items[0].data["quantity"] == 5
