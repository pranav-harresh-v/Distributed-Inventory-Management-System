# Phase F — Business Logic Notes

Handoff record for Phase F (steps 23–30): products, warehouses, inventory,
suppliers, order placement, sales history. Written so the next session or
team member can continue without the conversation that produced it.

**Status:** complete. 619 tests pass, `ruff check` clean, `compileall` clean.

---

## 1. Where things live

```
app/api/client_service.py   gRPC servicer: auth, action routing, error mapping
app/domain/inventory.py     catalogue + stock: products, warehouses, inventory, suppliers
app/domain/orders.py        atomic order placement
app/storage/repositories.py data access; flush only, never commit
app/storage/models.py       UNCHANGED since Phase D — no schema change in all of Phase F
proto/client.proto          UNCHANGED since Phase E
```

Product/warehouse/inventory/supplier logic all sits in `inventory.py` because
the Phase D scaffold provided that module and step 23 chose it over a new
`products.py`. `orders.py` is separate because order placement is the
correctness-critical path and reuses `inventory.py`'s validators through
public aliases at the bottom of that file.

## 2. Action routing (the key design decision)

The Phase E protobuf contract is generic and was **not** changed. Operations
are selected by the `resource` string, with payloads as
`google.protobuf.Struct`. Handlers are registered in two tables in
`client_service.py`:

| RPC | resource | Notes |
|---|---|---|
| `post` | `product.create` | |
| `post` | `product.update` | identify with `id` or `sku_key` |
| `post` | `warehouse.create` | |
| `post` | `inventory.set_initial_stock` | rejects if a row already exists |
| `post` | `inventory.add_stock` | increment only |
| `post` | `supplier.create` | |
| `post` | `supplier.associate_product` | stores `unit_cost`, `lead_time_days` |
| `post` | `order.place` | atomic |
| `get` | `product`, `products` | |
| `get` | `warehouse`, `warehouses` | |
| `get` | `suppliers` | |
| `get` | `inventory.stock`, `inventory.warehouse`, `inventory.low_stock` | |

Adding an operation = write a domain function + register a handler. No proto
or codegen change.

**Authentication runs before action resolution.** This ordering is
deliberate and load-bearing: the reverse lets an unauthenticated caller
discover which resources exist by comparing `INVALID_ARGUMENT` against
`UNAUTHENTICATED`. It was briefly wrong during step 23 and caught by tests.

## 3. Transaction boundary

```
session_scope        <- the single commit, on clean exit; rollback + re-raise otherwise
 └─ domain function  <- flushes only, never commits
     └─ repository   <- flushes only, never commits
```

`session_scope` is entered by the servicer's `_authenticated` context
manager. No repository or domain function may call `.commit()`; tests assert
this by source inspection. This is what makes multi-table order placement
atomic.

## 4. Negative stock — four layers

1. Domain validation (`minimum=0` for stock, `minimum=1` for amounts).
2. **Guarded conditional UPDATE** — the important one:
   `SET quantity = quantity - :n WHERE ... AND quantity >= :n`, then check
   `rowcount`. The availability test and the write are one statement, so two
   concurrent orders cannot both pass a check and both write. Never
   read-modify-write.
3. `ck_inventory_quantity_non_negative` in the schema (backstop).
4. Transaction rollback.

`grep -rE "\.quantity\s*=[^=]" app/` returns zero hits: the only writes are
INSERT, guarded increment, guarded decrement. Keep it that way — if you add
an adjustment feature, give it a new guarded repository method rather than
mutating a loaded row and calling `save()`.

## 5. Money and timestamps (two recurring traps)

- **Money.** `Struct` stores numbers as doubles and *rejects* `Decimal`
  outright. Prices therefore cross the wire as **strings** and are rebuilt
  with `Decimal(str(value))`. `Money` in `models.py` raises `TypeError` on a
  float. Excess decimals round half-up to the column's 2 dp; this is pinned
  by tests rather than left silent.
- **Timestamps.** SQLite drops `tzinfo`, so stored values read back naive.
  Comparing one against an aware `utc_now()` raises `TypeError`. Always go
  through `ensure_utc()` on read. This bit the test suite twice.

## 6. Error mapping

| Domain error | gRPC status |
|---|---|
| `ValidationError` | `INVALID_ARGUMENT` |
| `NotFoundError` | `NOT_FOUND` |
| `DuplicateError` | `ALREADY_EXISTS` |
| `InsufficientStockError` | `FAILED_PRECONDITION` |
| auth failure | `UNAUTHENTICATED` |
| unexpected DB error | rollback + `INTERNAL` (logged server-side, opaque to client) |

No SQL, traceback, credential or hash ever reaches a client; tests assert it.

## 7. Known gaps / decisions to revisit

- **`suppliers.name` uniqueness is application-level only.** That table has
  no unique index, so unlike a product SKU it is **not race-proof**. A test
  documents this by inserting a duplicate directly. Making it airtight is a
  schema change.
- **Roles are declared but unenforced.** `USER_ROLES` exists and
  `UserInfo.role` is returned, but any authenticated user can write. Decide
  before the demo.
- **No read action for supplier–product links or for orders.** Terms and
  orders are write-only over gRPC today; tests verify them via repositories.
  `processBusinessRequest` is still `UNIMPLEMENTED` and is the natural home
  for structured order responses — and for Raft commands in Milestone 2.
- **No update/delete** for warehouses, suppliers or inventory thresholds; no
  order cancellation.
- **Order numbers are random** (`ORD-` + uuid4 hex). Fine now; for Milestone 2
  the leader must generate the id and put it *inside* the replicated command
  so every node applies the same value.
- **Suite takes ~6 minutes**, almost entirely bcrypt at cost 12. Lower the
  cost in tests only if it becomes a drag.

## 8. Milestone 2 readiness

Order placement is deliberately deterministic: lines are consolidated and
processed in ascending product-id order, so the same request touches the
same rows in the same sequence on every replica. `place_order` takes an
explicit session and never commits, so it can later run inside a
Raft-committed apply step unchanged.

## 9. Running things

```bash
python -m pytest -q                 # 619 tests
ruff check app tests scripts
python -m app.server                # 127.0.0.1:50051
python scripts/grpc_smoke_test.py   # developer smoke client
powershell -File scripts/generate_proto.ps1   # regenerate stubs (gitignored)
```

Tests use temporary SQLite databases and ephemeral ports; a session-scoped
guard fixture fails the run if anything touches `data/node1.db`.
