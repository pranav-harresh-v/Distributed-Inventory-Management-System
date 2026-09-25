"""Inventory catalogue business logic.

Phase F step 23 implements the product catalogue here. Warehouses and stock
levels join this module in later steps, which is why the error types are
named for the catalogue rather than for products specifically.

Layering: this module owns validation and business rules, talks to the
database only through :mod:`app.storage.repositories`, and knows nothing
about gRPC. The caller owns the transaction (normally
:func:`app.storage.database.session_scope`), so every function here flushes
but never commits -- a failure propagates and the caller's scope rolls back.

Money crossing the wire is a special case. ``google.protobuf.Struct`` holds
JSON types only: it rejects :class:`~decimal.Decimal` outright and stores
numbers as doubles. Prices therefore travel as **strings** and are rebuilt
with ``Decimal(str(value))``, so no binary float ever reaches the database.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Final

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession

from app.storage.models import (
    CURRENCY_EXPONENT,
    Inventory,
    Product,
    Supplier,
    SupplierProduct,
    Warehouse,
    ensure_utc,
)
from app.storage.repositories import (
    InventoryRepository,
    ProductRepository,
    SupplierProductRepository,
    SupplierRepository,
    WarehouseRepository,
)

__all__ = [
    "PRODUCT_MUTABLE_FIELDS",
    "WAREHOUSE_CREATE_FIELDS",
    "CatalogError",
    "DuplicateError",
    "NotFoundError",
    "ValidationError",
    "add_stock",
    "associate_supplier_product",
    "create_product",
    "create_supplier",
    "create_warehouse",
    "get_product",
    "get_stock",
    "get_supplier",
    "get_warehouse",
    "is_low_stock",
    "list_low_stock",
    "list_products",
    "list_suppliers",
    "list_warehouse_inventory",
    "list_warehouses",
    "serialize_inventory",
    "serialize_product",
    "serialize_supplier",
    "serialize_supplier_product",
    "serialize_warehouse",
    "set_initial_stock",
    "update_product",
]

# --------------------------------------------------------------------------
# Field rules, derived from the Phase D column definitions
# --------------------------------------------------------------------------
SKU_MAX_LENGTH: Final[int] = 64  # products.sku  VARCHAR(64)
NAME_MAX_LENGTH: Final[int] = 255  # products.name VARCHAR(255)

#: NUMERIC(12, 2) leaves ten digits ahead of the decimal point.
MAX_UNIT_PRICE: Final[Decimal] = Decimal("9999999999.99")

#: The only columns a client may change through update_product. `id`,
#: `created_at` and `updated_at` are deliberately absent: identity and
#: audit timestamps are the database's to manage, not a caller's.
PRODUCT_MUTABLE_FIELDS: Final[frozenset[str]] = frozenset(
    {"sku", "name", "description", "unit_price", "is_active"}
)

#: Accepted on create. Same set, because every mutable column is also
#: settable at creation time.
PRODUCT_CREATE_FIELDS: Final[frozenset[str]] = PRODUCT_MUTABLE_FIELDS

WAREHOUSE_CODE_MAX_LENGTH: Final[int] = 32  # warehouses.code     VARCHAR(32)
WAREHOUSE_NAME_MAX_LENGTH: Final[int] = 255  # warehouses.name     VARCHAR(255)
WAREHOUSE_LOCATION_MAX_LENGTH: Final[int] = 255  # warehouses.location VARCHAR(255)

#: Settable when creating a warehouse. `code` is the only column the schema
#: makes unique -- `warehouses.name` is deliberately not unique, so two
#: depots may share a name, and nothing here invents a constraint for it.
WAREHOUSE_CREATE_FIELDS: Final[frozenset[str]] = frozenset(
    {"code", "name", "location", "is_active"}
)

_DEFAULT_LIST_LIMIT: Final[int] = 100
_MAX_LIST_LIMIT: Final[int] = 500


class CatalogError(Exception):
    """Base class for catalogue failures."""


class ValidationError(CatalogError):
    """The request is malformed or carries an out-of-range value."""


class NotFoundError(CatalogError):
    """The referenced record does not exist."""


class DuplicateError(CatalogError):
    """A uniqueness constraint would be violated."""


# --------------------------------------------------------------------------
# Input coercion
# --------------------------------------------------------------------------
def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError(f"'{label}' must be an object.")
    return payload


def _reject_unknown_fields(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    """Fail loudly on fields we will not act on.

    Silently ignoring an unknown key would let a caller believe a write
    happened. Naming the offending keys also catches a client that tries to
    set `id` or `created_at`.
    """
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValidationError(
            f"Unknown or non-editable field(s): {', '.join(unknown)}. "
            f"Editable fields are: {', '.join(sorted(allowed))}."
        )


def _coerce_text(
    value: Any, field: str, max_length: int, *, allow_none: bool = False
) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"'{field}' is required.")
    if not isinstance(value, str):
        raise ValidationError(f"'{field}' must be a string.")
    cleaned = value.strip()
    if not cleaned:
        if allow_none:
            return None
        raise ValidationError(f"'{field}' must not be blank.")
    if len(cleaned) > max_length:
        raise ValidationError(f"'{field}' must be at most {max_length} characters.")
    return cleaned


def _coerce_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"'{field}' must be true or false.")
    return value


def _coerce_price(value: Any, field: str = "unit_price") -> Decimal:
    """Convert a wire value to an exact 2dp Decimal.

    ``bool`` is rejected explicitly because it is a subclass of ``int`` and
    would otherwise silently become a price of 0.00 or 1.00.
    """
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"'{field}' must be a number.")
    if isinstance(value, Decimal):
        candidate = value
    elif isinstance(value, (int, float, str)):
        try:
            # str() first: Decimal(24.99) would capture the binary float's
            # full error tail, Decimal("24.99") is exact.
            candidate = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as exc:
            raise ValidationError(f"'{field}' must be a number.") from exc
    else:
        raise ValidationError(f"'{field}' must be a number.")

    if not candidate.is_finite():
        raise ValidationError(f"'{field}' must be a finite number.")
    if candidate < 0:
        raise ValidationError(f"'{field}' must not be negative.")

    quantized = candidate.quantize(CURRENCY_EXPONENT, rounding=ROUND_HALF_UP)
    if quantized > MAX_UNIT_PRICE:
        raise ValidationError(f"'{field}' must not exceed {MAX_UNIT_PRICE}.")
    return quantized


#: Practical ceiling for stock counts, well inside SQLite's INTEGER range.
MAX_QUANTITY: Final[int] = 2**31 - 1


def _coerce_int(value: Any, field: str, *, minimum: int) -> int:
    """Convert a wire value to an int of at least ``minimum``.

    Struct carries every number as a double, so an id arrives as 1.0 and a
    quantity as 5.0. ``bool`` is rejected explicitly: it subclasses ``int``
    and would otherwise pass as 0 or 1.
    """
    if isinstance(value, bool) or value is None:
        raise ValidationError(f"'{field}' must be an integer.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValidationError(f"'{field}' must be a whole number.")
        number = int(value)
    elif isinstance(value, str):
        try:
            number = int(value.strip())
        except ValueError as exc:
            raise ValidationError(f"'{field}' must be an integer.") from exc
    else:
        raise ValidationError(f"'{field}' must be an integer.")

    if number < minimum:
        if minimum == 1:
            raise ValidationError(f"'{field}' must be positive.")
        if minimum == 0:
            raise ValidationError(f"'{field}' must not be negative.")
        raise ValidationError(f"'{field}' must be at least {minimum}.")
    if number > MAX_QUANTITY:
        raise ValidationError(f"'{field}' must not exceed {MAX_QUANTITY}.")
    return number


def _coerce_identifier(value: Any, field: str) -> int:
    return _coerce_int(value, field, minimum=1)


def _coerce_product_field(field: str, value: Any) -> Any:
    """Validate one product field by name."""
    if field == "sku":
        return _coerce_text(value, "sku", SKU_MAX_LENGTH)
    if field == "name":
        return _coerce_text(value, "name", NAME_MAX_LENGTH)
    if field == "description":
        # TEXT column: no length cap, and blank collapses to NULL.
        return _coerce_text(value, "description", 10_000, allow_none=True)
    if field == "unit_price":
        return _coerce_price(value)
    if field == "is_active":
        return _coerce_bool(value, "is_active")
    raise ValidationError(f"Unsupported field '{field}'.")


def _coerce_warehouse_field(field: str, value: Any) -> Any:
    """Validate one warehouse field by name."""
    if field == "code":
        return _coerce_text(value, "code", WAREHOUSE_CODE_MAX_LENGTH)
    if field == "name":
        return _coerce_text(value, "name", WAREHOUSE_NAME_MAX_LENGTH)
    if field == "location":
        return _coerce_text(value, "location", WAREHOUSE_LOCATION_MAX_LENGTH)
    if field == "is_active":
        return _coerce_bool(value, "is_active")
    raise ValidationError(f"Unsupported field '{field}'.")


def _translate_integrity_error(
    exc: IntegrityError, *, entity: str, conflict: str | None = None
) -> CatalogError:
    """Turn a database constraint violation into a domain error.

    The kind of constraint decides the domain error, so a client learns
    *what* went wrong without ever seeing SQL. ``entity`` and ``conflict``
    only shape the wording.
    """
    detail = str(exc.orig) if exc.orig is not None else str(exc)
    if "UNIQUE" in detail.upper():
        label = conflict or "value"
        return DuplicateError(f"A {entity} with that {label} already exists.")
    if "CHECK" in detail.upper():
        return ValidationError(f"The {entity} violates a database constraint.")
    return ValidationError(f"The {entity} could not be saved.")


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------
def _coerce_offset(value: Any) -> int:
    """Paging offset: any whole number from zero upwards."""
    return _coerce_int(value, "offset", minimum=0)


def _list_arguments(params: Any) -> dict[str, Any]:
    """Validate the paging/filter params shared by every list operation.

    Returned as keyword arguments for ``BaseRepository.list``, whose
    primary-key ordering is what makes the result deterministic.
    """
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, frozenset({"active_only", "limit", "offset"}))

    active_only = (
        _coerce_bool(data["active_only"], "active_only")
        if "active_only" in data
        else False
    )
    limit = (
        _coerce_identifier(data["limit"], "limit")
        if "limit" in data
        else _DEFAULT_LIST_LIMIT
    )
    if limit > _MAX_LIST_LIMIT:
        raise ValidationError(f"'limit' must not exceed {_MAX_LIST_LIMIT}.")
    offset = _coerce_offset(data["offset"]) if "offset" in data else None

    return {"active_only": active_only, "limit": limit, "offset": offset}


def _isoformat(value: Any) -> str | None:
    return ensure_utc(value).isoformat() if value is not None else None


def serialize_product(product: Product) -> dict[str, Any]:
    """Project a Product row onto protobuf-Struct-safe primitives.

    No ORM object, Decimal or datetime ever reaches the wire: prices become
    exact decimal strings (a double would round 1234567.89) and timestamps
    become ISO-8601 UTC strings.
    """
    return {
        "id": product.id,
        "sku": product.sku,
        "name": product.name,
        "description": product.description,
        "unit_price": str(product.unit_price),
        "is_active": product.is_active,
        "created_at": _isoformat(product.created_at),
        "updated_at": _isoformat(product.updated_at),
    }


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------
def create_product(db: DbSession, payload: Any) -> Product:
    """Create a product from a client payload.

    Validates first so bad input is rejected before touching the database,
    then lets the schema's own constraints have the final word.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(data, PRODUCT_CREATE_FIELDS)

    for required in ("sku", "name", "unit_price"):
        if required not in data:
            raise ValidationError(f"'{required}' is required.")

    fields = {field: _coerce_product_field(field, data[field]) for field in data}
    repository = ProductRepository(db)

    # Check first for a clear, cheap error; the unique index below is what
    # actually guarantees it under concurrency.
    if repository.get_by_sku(fields["sku"]) is not None:
        raise DuplicateError(f"A product with sku '{fields['sku']}' already exists.")

    try:
        return repository.create(
            sku=fields["sku"],
            name=fields["name"],
            unit_price=fields["unit_price"],
            description=fields.get("description"),
            is_active=fields.get("is_active", True),
        )
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc, entity="product", conflict=f"sku '{fields['sku']}'"
        ) from exc


def get_product(
    db: DbSession, *, product_id: int | None = None, sku: str | None = None
) -> Product:
    """Fetch one product by id or sku. Exactly one must be supplied."""
    if (product_id is None) == (sku is None):
        raise ValidationError("Provide exactly one of 'id' or 'sku'.")

    repository = ProductRepository(db)
    product = (
        repository.get(product_id)
        if product_id is not None
        else repository.get_by_sku(sku)
    )
    if product is None:
        label = f"id {product_id}" if product_id is not None else f"sku '{sku}'"
        raise NotFoundError(f"No product with {label}.")
    return product


def get_product_from_params(db: DbSession, params: Any) -> Product:
    """Resolve the identifier out of a client ``params`` mapping.

    Supplying both 'id' and 'sku' is rejected rather than silently
    preferring one: the two could disagree, and guessing which the caller
    meant would return a product they did not ask for.
    """
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, frozenset({"id", "sku"}))

    supplied = {"id", "sku"} & set(data)
    if len(supplied) != 1:
        raise ValidationError("Provide exactly one of 'id' or 'sku'.")

    if "id" in supplied:
        return get_product(db, product_id=_coerce_identifier(data["id"], "id"))
    return get_product(db, sku=_coerce_text(data["sku"], "sku", SKU_MAX_LENGTH))


def list_products(db: DbSession, params: Any = None) -> Sequence[Product]:
    """List products in a deterministic order.

    Ordering is by primary key, inherited from ``BaseRepository.list``, so
    repeated calls and paged calls agree with each other.
    """
    return ProductRepository(db).list(**_list_arguments(params))


def update_product(db: DbSession, payload: Any) -> Product:
    """Apply a partial update to one product.

    Only fields named in :data:`PRODUCT_MUTABLE_FIELDS` are applied, and
    each is validated before assignment -- a caller cannot reach an
    arbitrary attribute. Fields left out keep their current values.
    """
    data = _require_mapping(payload, "payload")

    identifier_keys = {"id", "sku_key"} & set(data)
    if not identifier_keys:
        raise ValidationError("Provide 'id' or 'sku_key' to identify the product.")
    if len(identifier_keys) > 1:
        raise ValidationError("Provide only one of 'id' or 'sku_key'.")

    changes = {key: value for key, value in data.items() if key not in identifier_keys}
    _reject_unknown_fields(changes, PRODUCT_MUTABLE_FIELDS)
    if not changes:
        raise ValidationError("No fields to update.")

    if "id" in data:
        product = get_product(db, product_id=_coerce_identifier(data["id"], "id"))
    else:
        product = get_product(
            db, sku=_coerce_text(data["sku_key"], "sku_key", SKU_MAX_LENGTH)
        )

    validated = {
        field: _coerce_product_field(field, value) for field, value in changes.items()
    }

    new_sku = validated.get("sku")
    if new_sku is not None and new_sku != product.sku:
        clash = ProductRepository(db).get_by_sku(new_sku)
        if clash is not None and clash.id != product.id:
            raise DuplicateError(f"A product with sku '{new_sku}' already exists.")

    for field, value in validated.items():
        setattr(product, field, value)

    try:
        return ProductRepository(db).save(product)
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc, entity="product", conflict=f"sku '{new_sku or product.sku}'"
        ) from exc


# --------------------------------------------------------------------------
# Warehouses
# --------------------------------------------------------------------------
def serialize_warehouse(warehouse: Warehouse) -> dict[str, Any]:
    """Project a Warehouse row onto protobuf-Struct-safe primitives."""
    return {
        "id": warehouse.id,
        "code": warehouse.code,
        "name": warehouse.name,
        "location": warehouse.location,
        "is_active": warehouse.is_active,
        "created_at": _isoformat(warehouse.created_at),
        "updated_at": _isoformat(warehouse.updated_at),
    }


def create_warehouse(db: DbSession, payload: Any) -> Warehouse:
    """Create a warehouse from a client payload.

    ``code`` is the natural key and the only uniqueness the schema enforces.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(data, WAREHOUSE_CREATE_FIELDS)

    for required in ("code", "name", "location"):
        if required not in data:
            raise ValidationError(f"'{required}' is required.")

    fields = {field: _coerce_warehouse_field(field, data[field]) for field in data}
    repository = WarehouseRepository(db)

    # Cheap, clear pre-check; the unique index is what actually guarantees
    # it when two callers race.
    if repository.get_by_code(fields["code"]) is not None:
        raise DuplicateError(
            f"A warehouse with code '{fields['code']}' already exists."
        )

    try:
        return repository.create(
            code=fields["code"],
            name=fields["name"],
            location=fields["location"],
            is_active=fields.get("is_active", True),
        )
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc, entity="warehouse", conflict=f"code '{fields['code']}'"
        ) from exc


def get_warehouse(
    db: DbSession, *, warehouse_id: int | None = None, code: str | None = None
) -> Warehouse:
    """Fetch one warehouse by id or code. Exactly one must be supplied."""
    if (warehouse_id is None) == (code is None):
        raise ValidationError("Provide exactly one of 'id' or 'code'.")

    repository = WarehouseRepository(db)
    warehouse = (
        repository.get(warehouse_id)
        if warehouse_id is not None
        else repository.get_by_code(code)
    )
    if warehouse is None:
        label = f"id {warehouse_id}" if warehouse_id is not None else f"code '{code}'"
        raise NotFoundError(f"No warehouse with {label}.")
    return warehouse


def get_warehouse_from_params(db: DbSession, params: Any) -> Warehouse:
    """Resolve the identifier out of a client ``params`` mapping.

    Supplying both is rejected rather than silently preferring one, for the
    same reason as products: the two could disagree.
    """
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, frozenset({"id", "code"}))

    supplied = {"id", "code"} & set(data)
    if len(supplied) != 1:
        raise ValidationError("Provide exactly one of 'id' or 'code'.")

    if "id" in supplied:
        return get_warehouse(db, warehouse_id=_coerce_identifier(data["id"], "id"))
    return get_warehouse(
        db, code=_coerce_text(data["code"], "code", WAREHOUSE_CODE_MAX_LENGTH)
    )


def list_warehouses(db: DbSession, params: Any = None) -> Sequence[Warehouse]:
    """List warehouses in primary-key order."""
    return WarehouseRepository(db).list(**_list_arguments(params))


# --------------------------------------------------------------------------
# Inventory (stock levels)
#
# The central safety invariant of the whole system is `quantity >= 0`. It is
# enforced three times over, deliberately:
#   1. here, by validating every incoming amount before any write;
#   2. by only ever *adding* through add_stock -- no subtraction path exists
#      in this step, so no operation can drive a row downwards;
#   3. by `ck_inventory_quantity_non_negative` in the schema, which rejects
#      the row even if a future caller bypasses this module entirely.
# Client-side validation is never part of that chain.
# --------------------------------------------------------------------------
INVENTORY_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {"quantity", "reorder_point", "safety_stock"}
)

_WAREHOUSE_KEYS: Final[frozenset[str]] = frozenset({"warehouse_id", "warehouse_code"})
_PRODUCT_KEYS: Final[frozenset[str]] = frozenset({"product_id", "sku"})


def _resolve_warehouse(db: DbSession, data: Mapping[str, Any]) -> Warehouse:
    """Find the warehouse named by 'warehouse_id' or 'warehouse_code'."""
    supplied = _WAREHOUSE_KEYS & set(data)
    if len(supplied) != 1:
        raise ValidationError(
            "Provide exactly one of 'warehouse_id' or 'warehouse_code'."
        )
    if "warehouse_id" in supplied:
        return get_warehouse(
            db, warehouse_id=_coerce_identifier(data["warehouse_id"], "warehouse_id")
        )
    return get_warehouse(
        db,
        code=_coerce_text(
            data["warehouse_code"], "warehouse_code", WAREHOUSE_CODE_MAX_LENGTH
        ),
    )


def _resolve_product(db: DbSession, data: Mapping[str, Any]) -> Product:
    """Find the product named by 'product_id' or 'sku'."""
    supplied = _PRODUCT_KEYS & set(data)
    if len(supplied) != 1:
        raise ValidationError("Provide exactly one of 'product_id' or 'sku'.")
    if "product_id" in supplied:
        return get_product(
            db, product_id=_coerce_identifier(data["product_id"], "product_id")
        )
    return get_product(db, sku=_coerce_text(data["sku"], "sku", SKU_MAX_LENGTH))


def is_low_stock(record: Inventory) -> bool:
    """Low stock is exactly ``quantity <= reorder_point``.

    The threshold is the schema's own per-row ``reorder_point``; nothing
    here forecasts or infers a different rule.
    """
    return record.quantity <= record.reorder_point


def serialize_inventory(record: Inventory) -> dict[str, Any]:
    """Project a stock row onto protobuf-Struct-safe primitives."""
    return {
        "id": record.id,
        "warehouse_id": record.warehouse_id,
        "warehouse_code": record.warehouse.code,
        "product_id": record.product_id,
        "sku": record.product.sku,
        "product_name": record.product.name,
        "quantity": record.quantity,
        "reorder_point": record.reorder_point,
        "safety_stock": record.safety_stock,
        "is_low_stock": is_low_stock(record),
        "updated_at": _isoformat(record.updated_at),
    }


def set_initial_stock(db: DbSession, payload: Any) -> Inventory:
    """Create the stock row for a (warehouse, product) pair.

    Refuses when a row already exists rather than overwriting it: silently
    replacing a quantity would destroy stock history and could mask a
    client bug as a successful write. Use add_stock to change a level.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(
        data, INVENTORY_LEVEL_FIELDS | _WAREHOUSE_KEYS | _PRODUCT_KEYS
    )

    if "quantity" not in data:
        raise ValidationError("'quantity' is required.")

    warehouse = _resolve_warehouse(db, data)
    product = _resolve_product(db, data)

    # Every level is non-negative; the schema asserts the same three rules.
    quantity = _coerce_int(data["quantity"], "quantity", minimum=0)
    reorder_point = (
        _coerce_int(data["reorder_point"], "reorder_point", minimum=0)
        if "reorder_point" in data
        else 0  # matches the column default
    )
    safety_stock = (
        _coerce_int(data["safety_stock"], "safety_stock", minimum=0)
        if "safety_stock" in data
        else 0  # matches the column default
    )

    repository = InventoryRepository(db)
    if repository.get_by_warehouse_and_product(warehouse.id, product.id) is not None:
        raise DuplicateError(
            f"Stock for product '{product.sku}' in warehouse "
            f"'{warehouse.code}' already exists; use add_stock to change it."
        )

    try:
        return repository.create(
            warehouse_id=warehouse.id,
            product_id=product.id,
            quantity=quantity,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
        )
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc,
            entity="stock record",
            conflict=f"warehouse '{warehouse.code}' and product '{product.sku}'",
        ) from exc


def add_stock(db: DbSession, payload: Any) -> Inventory:
    """Increase the stock of an existing (warehouse, product) row.

    Increment only -- there is no subtraction path in this step, so this
    operation cannot move a quantity towards zero. The write is a single
    database-side ``quantity = quantity + :amount`` (see
    ``InventoryRepository.increment_quantity``), so two concurrent callers
    cannot lose one another's increment the way a read-modify-write would.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(
        data, frozenset({"amount"}) | _WAREHOUSE_KEYS | _PRODUCT_KEYS
    )

    if "amount" not in data:
        raise ValidationError("'amount' is required.")
    amount = _coerce_int(data["amount"], "amount", minimum=1)

    warehouse = _resolve_warehouse(db, data)
    product = _resolve_product(db, data)

    repository = InventoryRepository(db)
    updated = repository.increment_quantity(
        warehouse_id=warehouse.id, product_id=product.id, amount=amount
    )
    if updated == 0:
        raise NotFoundError(
            f"No stock record for product '{product.sku}' in warehouse "
            f"'{warehouse.code}'."
        )

    record = repository.get_by_warehouse_and_product(warehouse.id, product.id)
    if record is None:  # pragma: no cover - the update just matched this row
        raise NotFoundError("Stock record disappeared during the update.")
    return record


def get_stock(db: DbSession, params: Any) -> Inventory:
    """Fetch the stock row for one (warehouse, product) pair."""
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, _WAREHOUSE_KEYS | _PRODUCT_KEYS)

    warehouse = _resolve_warehouse(db, data)
    product = _resolve_product(db, data)

    record = InventoryRepository(db).get_by_warehouse_and_product(
        warehouse.id, product.id
    )
    if record is None:
        raise NotFoundError(
            f"No stock record for product '{product.sku}' in warehouse "
            f"'{warehouse.code}'."
        )
    return record


def list_warehouse_inventory(db: DbSession, params: Any) -> Sequence[Inventory]:
    """Every stock row in one warehouse, ordered by product id.

    The warehouse is resolved first so an unknown warehouse is a NOT_FOUND
    rather than an empty list that looks like a warehouse with no stock.
    """
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, _WAREHOUSE_KEYS)

    warehouse = _resolve_warehouse(db, data)
    return InventoryRepository(db).list_for_warehouse(warehouse.id)


def list_low_stock(db: DbSession, params: Any = None) -> Sequence[Inventory]:
    """Stock rows at or below their reorder point.

    Optionally scoped to one warehouse; with no warehouse given it reports
    across every warehouse on this node.
    """
    data = _require_mapping(params or {}, "params")
    _reject_unknown_fields(data, _WAREHOUSE_KEYS)

    warehouse_id = None
    if _WAREHOUSE_KEYS & set(data):
        warehouse_id = _resolve_warehouse(db, data).id

    return InventoryRepository(db).list_low_stock(warehouse_id)


# --------------------------------------------------------------------------
# Suppliers
#
# Constraint note, because the two tables differ in strength:
#   * `supplier_products` carries UNIQUE(supplier_id, product_id) and three
#     CHECKs, so its rules are guaranteed by the database.
#   * `suppliers` carries NO unique constraint. The "one supplier per name"
#     rule below is therefore an *application* rule only. It gives a clean
#     ALREADY_EXISTS for the ordinary case, but unlike a product SKU it is
#     not race-proof: two concurrent creates could both pass the check.
#     Making it airtight would need a unique index, which is a schema
#     change and out of scope for this step.
# --------------------------------------------------------------------------
SUPPLIER_NAME_MAX_LENGTH: Final[int] = 255  # suppliers.name  VARCHAR(255)
SUPPLIER_EMAIL_MAX_LENGTH: Final[int] = 255  # suppliers.email VARCHAR(255)
SUPPLIER_PHONE_MAX_LENGTH: Final[int] = 64  # suppliers.phone VARCHAR(64)
SUPPLIER_SKU_MAX_LENGTH: Final[int] = 64  # supplier_products.supplier_sku

SUPPLIER_CREATE_FIELDS: Final[frozenset[str]] = frozenset(
    {"name", "email", "phone", "address", "is_active"}
)

#: Terms stored on the supplier/product link. `unit_cost` is the price and
#: `lead_time_days` the lead time; both column names come from the schema.
SUPPLIER_PRODUCT_TERM_FIELDS: Final[frozenset[str]] = frozenset(
    {"unit_cost", "lead_time_days", "supplier_sku", "minimum_order_qty", "is_preferred"}
)

_SUPPLIER_KEYS: Final[frozenset[str]] = frozenset({"supplier_id", "supplier_name"})


def _coerce_supplier_field(field: str, value: Any) -> Any:
    """Validate one supplier field by name."""
    if field == "name":
        return _coerce_text(value, "name", SUPPLIER_NAME_MAX_LENGTH)
    if field == "email":
        return _coerce_text(value, "email", SUPPLIER_EMAIL_MAX_LENGTH, allow_none=True)
    if field == "phone":
        return _coerce_text(value, "phone", SUPPLIER_PHONE_MAX_LENGTH, allow_none=True)
    if field == "address":
        return _coerce_text(value, "address", 10_000, allow_none=True)
    if field == "is_active":
        return _coerce_bool(value, "is_active")
    raise ValidationError(f"Unsupported field '{field}'.")


def serialize_supplier(supplier: Supplier) -> dict[str, Any]:
    """Project a Supplier row onto protobuf-Struct-safe primitives."""
    return {
        "id": supplier.id,
        "name": supplier.name,
        "email": supplier.email,
        "phone": supplier.phone,
        "address": supplier.address,
        "is_active": supplier.is_active,
        "created_at": _isoformat(supplier.created_at),
        "updated_at": _isoformat(supplier.updated_at),
    }


def serialize_supplier_product(link: SupplierProduct) -> dict[str, Any]:
    """Project a supplier/product link onto Struct-safe primitives.

    ``unit_cost`` becomes an exact decimal string for the same reason as a
    product price: a Struct stores numbers as doubles and rejects Decimal.
    """
    return {
        "id": link.id,
        "supplier_id": link.supplier_id,
        "supplier_name": link.supplier.name,
        "product_id": link.product_id,
        "sku": link.product.sku,
        "supplier_sku": link.supplier_sku,
        "unit_cost": str(link.unit_cost),
        "lead_time_days": link.lead_time_days,
        "minimum_order_qty": link.minimum_order_qty,
        "is_preferred": link.is_preferred,
        "created_at": _isoformat(link.created_at),
        "updated_at": _isoformat(link.updated_at),
    }


def create_supplier(db: DbSession, payload: Any) -> Supplier:
    """Create a supplier.

    ``name`` is the only required field; email, phone and address are
    nullable in the schema and stay optional here.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(data, SUPPLIER_CREATE_FIELDS)

    if "name" not in data:
        raise ValidationError("'name' is required.")

    fields = {field: _coerce_supplier_field(field, data[field]) for field in data}
    repository = SupplierRepository(db)

    # Application-level uniqueness; see the note at the top of this section.
    if repository.get_by_name(fields["name"]) is not None:
        raise DuplicateError(f"A supplier named '{fields['name']}' already exists.")

    try:
        return repository.create(
            name=fields["name"],
            email=fields.get("email"),
            phone=fields.get("phone"),
            address=fields.get("address"),
            is_active=fields.get("is_active", True),
        )
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc, entity="supplier", conflict=f"name '{fields['name']}'"
        ) from exc


def get_supplier(
    db: DbSession, *, supplier_id: int | None = None, name: str | None = None
) -> Supplier:
    """Fetch one supplier by id or name. Exactly one must be supplied."""
    if (supplier_id is None) == (name is None):
        raise ValidationError(
            "Provide exactly one of 'supplier_id' or 'supplier_name'."
        )

    repository = SupplierRepository(db)
    supplier = (
        repository.get(supplier_id)
        if supplier_id is not None
        else repository.get_by_name(name)
    )
    if supplier is None:
        label = f"id {supplier_id}" if supplier_id is not None else f"name '{name}'"
        raise NotFoundError(f"No supplier with {label}.")
    return supplier


def _resolve_supplier(db: DbSession, data: Mapping[str, Any]) -> Supplier:
    """Find the supplier named by 'supplier_id' or 'supplier_name'."""
    supplied = _SUPPLIER_KEYS & set(data)
    if len(supplied) != 1:
        raise ValidationError(
            "Provide exactly one of 'supplier_id' or 'supplier_name'."
        )
    if "supplier_id" in supplied:
        return get_supplier(
            db, supplier_id=_coerce_identifier(data["supplier_id"], "supplier_id")
        )
    return get_supplier(
        db,
        name=_coerce_text(
            data["supplier_name"], "supplier_name", SUPPLIER_NAME_MAX_LENGTH
        ),
    )


def list_suppliers(db: DbSession, params: Any = None) -> Sequence[Supplier]:
    """List suppliers in primary-key order."""
    return SupplierRepository(db).list(**_list_arguments(params))


def associate_supplier_product(db: DbSession, payload: Any) -> SupplierProduct:
    """Link a supplier to a product, recording price and lead time.

    Both sides must already exist. The pair is unique in the schema, so a
    second attempt is rejected rather than quietly updating the terms.
    """
    data = _require_mapping(payload, "payload")
    _reject_unknown_fields(
        data, SUPPLIER_PRODUCT_TERM_FIELDS | _SUPPLIER_KEYS | _PRODUCT_KEYS
    )

    for required in ("unit_cost", "lead_time_days"):
        if required not in data:
            raise ValidationError(f"'{required}' is required.")

    supplier = _resolve_supplier(db, data)
    product = _resolve_product(db, data)

    # unit_cost is NUMERIC(12, 2) with CHECK >= 0; the same exact-Decimal
    # conversion used for product prices keeps a wire double from becoming
    # a binary-float artefact.
    unit_cost = _coerce_price(data["unit_cost"], "unit_cost")
    # lead_time_days is INTEGER with CHECK >= 0 -- zero is a valid
    # same-day lead time, so the floor is 0 rather than 1.
    lead_time_days = _coerce_int(data["lead_time_days"], "lead_time_days", minimum=0)
    # minimum_order_qty is INTEGER with CHECK > 0.
    minimum_order_qty = (
        _coerce_int(data["minimum_order_qty"], "minimum_order_qty", minimum=1)
        if "minimum_order_qty" in data
        else 1  # matches the column default
    )
    supplier_sku = (
        _coerce_text(
            data["supplier_sku"],
            "supplier_sku",
            SUPPLIER_SKU_MAX_LENGTH,
            allow_none=True,
        )
        if "supplier_sku" in data
        else None
    )
    is_preferred = (
        _coerce_bool(data["is_preferred"], "is_preferred")
        if "is_preferred" in data
        else False  # matches the column default
    )

    repository = SupplierProductRepository(db)
    if repository.get_by_supplier_and_product(supplier.id, product.id) is not None:
        raise DuplicateError(
            f"Supplier '{supplier.name}' is already linked to product '{product.sku}'."
        )

    try:
        return repository.create(
            supplier_id=supplier.id,
            product_id=product.id,
            unit_cost=unit_cost,
            lead_time_days=lead_time_days,
            supplier_sku=supplier_sku,
            minimum_order_qty=minimum_order_qty,
            is_preferred=is_preferred,
        )
    except IntegrityError as exc:
        raise _translate_integrity_error(
            exc,
            entity="supplier/product link",
            conflict=f"supplier '{supplier.name}' and product '{product.sku}'",
        ) from exc


# --------------------------------------------------------------------------
# Shared with app.domain.orders
#
# Order placement reuses this module's validation and resolution primitives
# rather than growing a second, divergent copy. They are exported under
# public names so no other module has to reach for an underscore. If a
# third consumer appears, these belong in their own module.
# --------------------------------------------------------------------------
PRODUCT_KEYS = _PRODUCT_KEYS
WAREHOUSE_KEYS = _WAREHOUSE_KEYS
require_mapping = _require_mapping
reject_unknown_fields = _reject_unknown_fields
coerce_int = _coerce_int
coerce_text = _coerce_text
coerce_identifier = _coerce_identifier
resolve_warehouse = _resolve_warehouse
resolve_product = _resolve_product
translate_integrity_error = _translate_integrity_error
isoformat = _isoformat
