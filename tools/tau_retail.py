"""Python implementations of tau-bench synthetic business tools.

These are API/state tools, not neural inference tools. They implement action
names generated from the tau-bench retail and airline samples so workflows can
execute end-to-end before real phone services exist.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .abi import fail, ok, upstream_value


DEFAULT_DATASET_PATH = Path("data/tau_bench_diverse_10.json")
DEFAULT_STATE_PATH = Path(".wfgen_tool_state/retail_state.json")
TOOL_ALIASES = {
    "find_user_id_by_name_zip": "find_user_by_name",
    "find_user_id_by_email": "find_user_by_contact",
    "get_reservation_details": "get_order_details",
}


def run_retail_tool(
    entrypoint: str,
    request: dict[str, Any],
    *,
    dataset_path: str | Path | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    original_entrypoint = entrypoint
    entrypoint = TOOL_ALIASES.get(entrypoint, entrypoint)
    params = request.get("parameters") or {}
    tool_cfg = request.get("tool_config") or {}
    dataset = Path(dataset_path or tool_cfg.get("dataset_path") or DEFAULT_DATASET_PATH)
    output_state = Path(state_path or tool_cfg.get("state_path") or DEFAULT_STATE_PATH)
    state = load_state(dataset)
    overlay_state(state, output_state)
    before_state = deepcopy(state)

    try:
        if entrypoint == "find_user_by_contact":
            user = find_user_by_contact(state, params)
            return ok(artifacts=user_artifacts(user), raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "find_user_by_name":
            user = find_user_by_name(state, params)
            return ok(artifacts=user_artifacts(user), raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "get_user_details":
            user = get_user_details(state, params)
            return ok(artifacts=user_artifacts(user), raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "get_order_details":
            order = get_order_details(state, params)
            return ok(artifacts=order_artifacts(order), raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "get_product_details":
            product = get_product_details(state, params)
            return ok(artifacts={"product": product}, raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "calculate":
            result = calculate(params)
            return ok(artifacts={"calculation": result, "calculation_result": result}, raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "search_direct_flight":
            result = search_direct_flight(state, params)
            return ok(artifacts={"flight_options": result, "search_result": result}, raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint}, started_at=started_at)
        if entrypoint == "book_reservation":
            result = book_reservation(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"reservation_result": result, "booking_result": result}, raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "cancel_reservation":
            result = cancel_reservation(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"reservation_result": result, "cancellation_result": result}, raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "cancel_order":
            result = cancel_order(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(
                artifacts={"cancellation_result": result},
                raw={"entrypoint": entrypoint, "state_path": str(output_state)},
                started_at=started_at,
            )
        if entrypoint == "return_delivered_order_items":
            result = return_delivered_order_items(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"return_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "exchange_delivered_order_items":
            result = exchange_delivered_order_items(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"exchange_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "modify_pending_order_items":
            result = modify_pending_order_items(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"order_modification_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "modify_pending_order_address":
            result = modify_pending_order_address(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"address_modification_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "modify_pending_order_payment":
            result = modify_pending_order_payment(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"payment_modification_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "modify_user_address":
            result = modify_user_address(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"user_address_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "update_reservation_flights":
            result = update_reservation_flights(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"reservation_update_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        if entrypoint == "update_reservation_baggages":
            result = update_reservation_baggages(state, params, request)
            save_state_delta(output_state, before_state, state)
            return ok(artifacts={"baggage_update_result": result}, raw={"entrypoint": entrypoint, "state_path": str(output_state)}, started_at=started_at)
        return fail(f"Unsupported retail tool: {original_entrypoint}", started_at=started_at)
    except Exception as exc:
        return fail(str(exc), raw={"entrypoint": original_entrypoint, "resolved_entrypoint": entrypoint, "parameters": params}, started_at=started_at)


def user_artifacts(user: dict[str, Any]) -> dict[str, Any]:
    return {"user": user, "user_id": user.get("user_id", "")}


def order_artifacts(order: dict[str, Any]) -> dict[str, Any]:
    artifacts = {"order": order, "order_id": order.get("order_id", "")}
    if order.get("kind") == "reservation":
        artifacts["reservation"] = order
        artifacts["reservation_id"] = order.get("order_id", "")
    return artifacts


def find_user_by_contact(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    email = normalize_email(params.get("email"))
    if not email:
        raise ValueError("find_user_by_contact requires parameter: email")
    user_id = state["indexes"]["users_by_email"].get(email)
    if not user_id:
        raise LookupError(f"No user found for email {email}")
    return deepcopy(state["users"][user_id])


def find_user_by_name(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    first = normalize_text(params.get("first_name"))
    last = normalize_text(params.get("last_name"))
    zip_code = normalize_text(params.get("zip"))
    if not first or not last:
        raise ValueError("find_user_by_name requires first_name and last_name")
    key = name_key(first, last, zip_code)
    user_id = state["indexes"]["users_by_name_zip"].get(key)
    if not user_id and zip_code:
        user_id = state["indexes"]["users_by_name_zip"].get(name_key(first, last, ""))
    if not user_id:
        raise LookupError(f"No user found for {first} {last} zip={zip_code}")
    return deepcopy(state["users"][user_id])


def get_order_details(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    order_id = normalize_order_id(params.get("order_id"))
    if not order_id:
        raise ValueError("get_order_details requires parameter: order_id")
    order = state["orders"].get(order_id)
    if not order:
        raise LookupError(f"No order found for {order_id}")
    return deepcopy(order)


def get_user_details(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    user_id = normalize_text(params.get("user_id"))
    if not user_id:
        raise ValueError("get_user_details requires parameter: user_id")
    user = state["users"].get(user_id)
    if not user:
        raise LookupError(f"No user found for id {user_id}")
    return deepcopy(user)


def get_product_details(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    product_id = normalize_text(params.get("product_id"))
    if not product_id:
        raise ValueError("get_product_details requires parameter: product_id")
    product = state["products"].get(product_id)
    if not product:
        product = {"product_id": product_id, "name": f"Product {product_id}", "status": "available"}
    return deepcopy(product)


def cancel_order(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order_id = normalize_order_id(params.get("order_id"))
    reason = normalize_text(params.get("reason")) or "unspecified"
    if not order_id:
        raise ValueError("cancel_order requires parameter: order_id")

    order = state["orders"].get(order_id)
    if not order:
        raise LookupError(f"No order found for {order_id}")

    user_context = upstream_value(request, "user")
    if isinstance(user_context, dict) and user_context.get("user_id"):
        if order.get("user_id") and order["user_id"] != user_context["user_id"]:
            raise PermissionError("Resolved user does not own the target order")

    previous_status = order.get("status", "pending")
    idempotent = False
    if previous_status in {"cancelled", "canceled"}:
        status = "cancelled"
        idempotent = True
    elif previous_status not in {"pending", "processing", "confirmed"}:
        status = "not_cancellable"
    else:
        status = "cancelled"
        order["status"] = "cancelled"
        order["cancel_reason"] = reason

    return {
        "order_id": order_id,
        "status": status,
        "previous_status": previous_status,
        "reason": reason,
        "user_id": order.get("user_id"),
        "idempotent": idempotent,
    }


def return_delivered_order_items(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order = require_order_for_update(state, params, request, allowed_statuses={"delivered", "returned"})
    item_ids = list(params.get("item_ids") or [])
    payment_method_id = normalize_text(params.get("payment_method_id"))
    returned = order.setdefault("returned_item_ids", [])
    for item_id in item_ids:
        if item_id not in returned:
            returned.append(item_id)
    order["status"] = "returned" if set(returned) >= set(order.get("item_ids", item_ids)) else order.get("status", "delivered")
    return {
        "order_id": order["order_id"],
        "status": "return_started",
        "item_ids": item_ids,
        "payment_method_id": payment_method_id,
        "refund_status": "pending",
    }


def exchange_delivered_order_items(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order = require_order_for_update(state, params, request, allowed_statuses={"delivered", "exchanged"})
    item_ids = list(params.get("item_ids") or [])
    new_item_ids = list(params.get("new_item_ids") or [])
    order.setdefault("exchanges", []).append({"item_ids": item_ids, "new_item_ids": new_item_ids})
    order["status"] = "exchanged"
    return {
        "order_id": order["order_id"],
        "status": "exchange_started",
        "item_ids": item_ids,
        "new_item_ids": new_item_ids,
        "payment_method_id": normalize_text(params.get("payment_method_id")),
    }


def modify_pending_order_items(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order = require_order_for_update(state, params, request, allowed_statuses={"pending", "processing"})
    item_ids = list(params.get("item_ids") or [])
    new_item_ids = list(params.get("new_item_ids") or [])
    replacements = dict(zip(item_ids, new_item_ids, strict=False))
    existing = order.setdefault("item_ids", [])
    for old_item_id, new_item_id in replacements.items():
        if old_item_id in existing:
            existing[existing.index(old_item_id)] = new_item_id
        elif new_item_id not in existing:
            existing.append(new_item_id)
    order["payment_method_id"] = normalize_text(params.get("payment_method_id")) or order.get("payment_method_id")
    return {
        "order_id": order["order_id"],
        "status": "items_modified",
        "replacements": replacements,
        "payment_method_id": order.get("payment_method_id"),
    }


def modify_pending_order_address(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order = require_order_for_update(state, params, request, allowed_statuses={"pending", "processing"})
    address = extract_address(params)
    order["shipping_address"] = address
    return {"order_id": order["order_id"], "status": "address_modified", "shipping_address": address}


def modify_pending_order_payment(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    order = require_order_for_update(state, params, request, allowed_statuses={"pending", "processing"})
    payment_method_id = normalize_text(params.get("payment_method_id"))
    if not payment_method_id:
        raise ValueError("modify_pending_order_payment requires payment_method_id")
    order["payment_method_id"] = payment_method_id
    return {"order_id": order["order_id"], "status": "payment_modified", "payment_method_id": payment_method_id}


def modify_user_address(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    user_id = normalize_text(params.get("user_id"))
    if not user_id:
        user_context = upstream_value(request, "user")
        if isinstance(user_context, dict):
            user_id = normalize_text(user_context.get("user_id"))
    if not user_id:
        raise ValueError("modify_user_address requires user_id")
    user = state["users"].get(user_id)
    if not user:
        user = {"user_id": user_id, "domain": "retail"}
        state["users"][user_id] = user
    address = extract_address(params)
    user["address"] = address
    return {"user_id": user_id, "status": "user_address_modified", "address": address}


def update_reservation_flights(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    reservation = require_reservation_for_update(state, params, request)
    flights = list(params.get("flights") or [])
    reservation["flights"] = flights
    reservation["cabin"] = normalize_text(params.get("cabin")) or reservation.get("cabin")
    reservation["payment_id"] = normalize_text(params.get("payment_id")) or reservation.get("payment_id")
    return {
        "reservation_id": reservation["order_id"],
        "status": "flights_updated",
        "cabin": reservation.get("cabin"),
        "flights": flights,
        "payment_id": reservation.get("payment_id"),
    }


def update_reservation_baggages(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    reservation = require_reservation_for_update(state, params, request)
    reservation["total_baggages"] = int(params.get("total_baggages") or 0)
    reservation["nonfree_baggages"] = int(params.get("nonfree_baggages") or 0)
    reservation["payment_id"] = normalize_text(params.get("payment_id")) or reservation.get("payment_id")
    return {
        "reservation_id": reservation["order_id"],
        "status": "baggage_updated",
        "total_baggages": reservation["total_baggages"],
        "nonfree_baggages": reservation["nonfree_baggages"],
        "payment_id": reservation.get("payment_id"),
    }


def calculate(params: dict[str, Any]) -> dict[str, Any]:
    expression = normalize_text(params.get("expression") or params.get("formula") or params.get("question"))
    return {"expression": expression, "result": normalize_text(params.get("result")) or "", "status": "not_evaluated"}


def search_direct_flight(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    flights = list(params.get("flights") or params.get("candidates") or [])
    if not flights:
        flight_number = normalize_text(params.get("flight_number"))
        if flight_number:
            flights = [{"flight_number": flight_number, "date": normalize_text(params.get("date"))}]
    return {"matches": flights, "filters": params}


def book_reservation(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    reservation_id = normalize_order_id(params.get("reservation_id") or params.get("order_id"))
    if not reservation_id:
        seed = json.dumps(params, sort_keys=True)
        reservation_id = "R" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6].upper()
    user_context = upstream_value(request, "user")
    user_id = user_context.get("user_id", "") if isinstance(user_context, dict) else normalize_text(params.get("user_id"))
    reservation = {
        "order_id": reservation_id,
        "reservation_id": reservation_id,
        "user_id": user_id,
        "status": "confirmed",
        "kind": "reservation",
        "flights": list(params.get("flights") or []),
        "cabin": normalize_text(params.get("cabin")),
        "payment_id": normalize_text(params.get("payment_id") or params.get("payment_method_id")),
    }
    state["orders"][reservation_id] = reservation
    return {"reservation_id": reservation_id, "status": "booked", "reservation": reservation}


def cancel_reservation(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    reservation = require_reservation_for_update(state, params, request)
    previous_status = reservation.get("status", "confirmed")
    reservation["status"] = "cancelled"
    return {"reservation_id": reservation["order_id"], "status": "cancelled", "previous_status": previous_status}


def require_order_for_update(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
    *,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    order_id = normalize_order_id(params.get("order_id"))
    if not order_id:
        raise ValueError("order update requires order_id")
    order = state["orders"].get(order_id)
    if not order:
        raise LookupError(f"No order found for {order_id}")

    user_context = upstream_value(request, "user")
    if isinstance(user_context, dict) and user_context.get("user_id"):
        if order.get("user_id") and order["user_id"] != user_context["user_id"]:
            raise PermissionError("Resolved user does not own the target order")

    if order.get("status") not in allowed_statuses:
        raise ValueError(
            f"Order {order_id} has status {order.get('status')}; "
            f"expected one of {sorted(allowed_statuses)}"
        )
    return order


def require_reservation_for_update(
    state: dict[str, Any],
    params: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    reservation_id = normalize_order_id(params.get("reservation_id") or params.get("order_id"))
    if not reservation_id:
        raise ValueError("reservation update requires reservation_id")
    reservation = state["orders"].get(reservation_id)
    if not reservation:
        raise LookupError(f"No reservation found for {reservation_id}")
    if reservation.get("kind") != "reservation":
        raise ValueError(f"{reservation_id} is not a reservation")

    user_context = upstream_value(request, "user")
    if isinstance(user_context, dict) and user_context.get("user_id"):
        if reservation.get("user_id") and reservation["user_id"] != user_context["user_id"]:
            raise PermissionError("Resolved user does not own the target reservation")
    return reservation


def extract_address(params: dict[str, Any]) -> dict[str, str]:
    return {
        "address1": normalize_text(params.get("address1")),
        "address2": normalize_text(params.get("address2")),
        "city": normalize_text(params.get("city")),
        "state": normalize_text(params.get("state")),
        "country": normalize_text(params.get("country")),
        "zip": normalize_text(params.get("zip")),
    }


def load_state(dataset_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    state = {
        "users": {},
        "orders": {},
        "products": {},
        "indexes": {"users_by_email": {}, "users_by_name_zip": {}},
    }
    for sample in dataset.get("samples", []):
        add_sample_to_state(state, sample)
    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


def save_state_delta(state_path: Path, before_state: dict[str, Any], after_state: dict[str, Any]) -> None:
    delta = diff_state(before_state, after_state)
    if not any(delta.values()):
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        existing = read_state_file(state_path)
        for section, items in delta.items():
            existing.setdefault(section, {}).update(items)
        state_path.write_text(json.dumps(existing, indent=2, ensure_ascii=True), encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass


def diff_state(before_state: dict[str, Any], after_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    delta: dict[str, dict[str, Any]] = {"users": {}, "orders": {}, "products": {}}
    for section in delta:
        before_items = before_state.get(section, {})
        for key, value in after_state.get(section, {}).items():
            if before_items.get(key) != value:
                delta[section][key] = value
    return delta


def read_state_file(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"users": {}, "orders": {}, "products": {}}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"users": {}, "orders": {}, "products": {}}
    data.setdefault("users", {})
    data.setdefault("orders", {})
    data.setdefault("products", {})
    return data


def overlay_state(state: dict[str, Any], state_path: Path) -> None:
    existing = read_state_file(state_path)
    for order_id, order in existing.get("orders", {}).items():
        if order_id in state["orders"]:
            state["orders"][order_id].update(order)
    for user_id, user in existing.get("users", {}).items():
        if user_id in state["users"]:
            state["users"][user_id].update(user)
    for product_id, product in existing.get("products", {}).items():
        if product_id in state["products"]:
            state["products"][product_id].update(product)


def add_sample_to_state(state: dict[str, Any], sample: dict[str, Any]) -> None:
    actions = sample.get("ground_truth_actions") or []
    user_action = next(
        (
            a
            for a in actions
            if a.get("name", "").startswith("find_user_by_")
            or a.get("name") in {"find_user_id_by_name_zip", "find_user_id_by_email"}
        ),
        None,
    )
    user_details_action = next((a for a in actions if a.get("name") == "get_user_details"), None)
    user = None
    if user_action:
        user = build_user(sample, user_action.get("arguments") or {})
        add_user_to_state(state, user)
    elif user_details_action:
        user = build_user(sample, user_details_action.get("arguments") or {})
        add_user_to_state(state, user)

    for action in actions:
        if action.get("name") == "get_product_details":
            add_product_to_state(state, sample, action.get("arguments") or {})

    for action in actions:
        if action.get("name") in {"get_order_details", "get_reservation_details"}:
            add_order_to_state(state, sample, action.get("arguments") or {}, user, actions)


def build_user(sample: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    explicit_user_id = normalize_text(args.get("user_id"))
    email = normalize_email(args.get("email")) or extract_email_from_sample(sample)
    first = normalize_text(args.get("first_name"))
    last = normalize_text(args.get("last_name"))
    if not first or not last:
        inferred_first, inferred_last = extract_name_from_sample(sample)
        first = first or inferred_first
        last = last or inferred_last
    if email and (not first or not last):
        first, last = name_from_email(email)
    user_id_seed = json.dumps({"email": email, "first": first, "last": last, "zip": args.get("zip")}, sort_keys=True)
    user_id = explicit_user_id or "user_" + hashlib.sha1(user_id_seed.encode("utf-8")).hexdigest()[:10]
    return {
        "user_id": user_id,
        "first_name": first,
        "last_name": last,
        "email": email,
        "zip": normalize_text(args.get("zip")),
        "sample_task_id": sample.get("task_id"),
        "domain": sample.get("domain"),
    }


def add_user_to_state(state: dict[str, Any], user: dict[str, Any]) -> None:
    state["users"][user["user_id"]] = user
    if user.get("email"):
        state["indexes"]["users_by_email"][normalize_email(user["email"])] = user["user_id"]
    if user.get("first_name") or user.get("last_name"):
        state["indexes"]["users_by_name_zip"][name_key(user.get("first_name"), user.get("last_name"), user.get("zip"))] = user["user_id"]
        state["indexes"]["users_by_name_zip"][name_key(user.get("first_name"), user.get("last_name"), "")] = user["user_id"]


def add_product_to_state(state: dict[str, Any], sample: dict[str, Any], args: dict[str, Any]) -> None:
    product_id = normalize_text(args.get("product_id"))
    if not product_id:
        return
    state["products"][product_id] = {
        "product_id": product_id,
        "name": f"Product {product_id}",
        "description": extract_reason_for_call(sample),
        "status": "available",
        "sample_task_id": sample.get("task_id"),
        "domain": sample.get("domain"),
    }


def add_order_to_state(
    state: dict[str, Any],
    sample: dict[str, Any],
    args: dict[str, Any],
    user: dict[str, Any] | None,
    actions: list[dict[str, Any]],
) -> None:
    order_id = normalize_order_id(args.get("order_id"))
    if not order_id:
        return
    state["orders"][order_id] = {
        "order_id": order_id,
        "user_id": user.get("user_id") if user else "",
        "status": infer_order_status(order_id, sample, actions),
        "item_ids": infer_item_ids(order_id, actions),
        "items_text": extract_reason_for_call(sample),
        "sample_task_id": sample.get("task_id"),
        "domain": sample.get("domain"),
        "kind": "reservation" if sample.get("domain") == "airline" else "retail_order",
    }


def infer_order_status(order_id: str, sample: dict[str, Any], actions: list[dict[str, Any]]) -> str:
    names_for_order = [
        action.get("name")
        for action in actions
        if normalize_order_id((action.get("arguments") or {}).get("order_id") or (action.get("arguments") or {}).get("reservation_id")) == order_id
    ]
    if sample.get("domain") == "airline":
        return "confirmed"
    if any(name in names_for_order for name in {"return_delivered_order_items", "exchange_delivered_order_items"}):
        return "delivered"
    return "pending"


def infer_item_ids(order_id: str, actions: list[dict[str, Any]]) -> list[str]:
    item_ids: list[str] = []
    for action in actions:
        args = action.get("arguments") or {}
        action_order_id = normalize_order_id(args.get("order_id") or args.get("reservation_id"))
        if action_order_id != order_id:
            continue
        for key in ("item_ids", "new_item_ids"):
            for item_id in args.get(key) or []:
                if item_id not in item_ids:
                    item_ids.append(item_id)
    return item_ids


def extract_reason_for_call(sample: dict[str, Any]) -> str:
    instructions = ((sample.get("user_scenario") or {}).get("instructions") or {})
    return normalize_text(instructions.get("reason_for_call"))


def sample_instruction_text(sample: dict[str, Any]) -> str:
    instructions = ((sample.get("user_scenario") or {}).get("instructions") or {})
    parts = [
        instructions.get("known_info"),
        instructions.get("reason_for_call"),
        instructions.get("task_instructions"),
    ]
    return " ".join(normalize_text(part) for part in parts if normalize_text(part))


def extract_email_from_sample(sample: dict[str, Any]) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", sample_instruction_text(sample))
    return normalize_email(match.group(0)) if match else ""


def extract_name_from_sample(sample: dict[str, Any]) -> tuple[str, str]:
    text = sample_instruction_text(sample)
    match = re.search(r"\b(?:You are|I am)\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\b", text)
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def normalize_email(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_order_id(value: Any) -> str:
    text = normalize_text(value)
    if text and text.startswith("W"):
        return "#" + text
    return text


def name_from_email(email: str) -> tuple[str, str]:
    local = email.split("@", 1)[0]
    pieces = [piece for piece in re.split(r"[^A-Za-z]+", local) if piece]
    if len(pieces) >= 2:
        return pieces[0].title(), pieces[1].title()
    if pieces:
        return pieces[0].title(), ""
    return "", ""


def name_key(first: Any, last: Any, zip_code: Any) -> str:
    return "|".join([normalize_text(first).lower(), normalize_text(last).lower(), normalize_text(zip_code)])
