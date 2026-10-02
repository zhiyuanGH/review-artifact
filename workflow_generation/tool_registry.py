"""Available tool metadata for cloud workflow generation prompts."""

from __future__ import annotations

from typing import Any

from ..tools.cli import ALL_TOOLS, QNN_TOOLS, RETAIL_TOOLS, example_parameters
from ..tools.llm_business import LLM_BUSINESS_TOOLS
from ..tools.llm_tau import LLM_TAU_TOOLS


TOOL_DESCRIPTIONS = {
    "find_user_by_contact": "Find a retail user by email or phone-like contact evidence.",
    "find_user_by_name": "Find a retail user by name, optionally constrained by zip code.",
    "find_user_id_by_email": "Alias-style lookup for a user id by email.",
    "find_user_id_by_name_zip": "Alias-style lookup for a user id by name and zip code.",
    "get_user_details": "Fetch a user profile or airline customer profile by user_id.",
    "get_order_details": "Fetch retail order or airline reservation details by order/reservation id.",
    "get_reservation_details": "Fetch airline reservation details; alias-compatible with get_order_details.",
    "get_product_details": "Fetch product catalog details by product_id.",
    "calculate": "Perform a simple deterministic calculation needed by downstream tools.",
    "search_direct_flight": "Search candidate direct flights for a route/date/cabin constraint.",
    "book_reservation": "Create an airline reservation.",
    "cancel_reservation": "Cancel an airline reservation.",
    "cancel_order": "Cancel a pending retail order or airline reservation when business rules allow it.",
    "return_delivered_order_items": "Return delivered retail order items and route refund to a selected payment method.",
    "exchange_delivered_order_items": "Exchange delivered retail order items for replacement products.",
    "modify_pending_order_items": "Modify items in a pending retail order.",
    "modify_pending_order_address": "Modify the shipping address for a pending retail order.",
    "modify_pending_order_payment": "Modify the payment method for a pending retail order.",
    "modify_user_address": "Modify a user's default address for future orders.",
    "update_reservation_flights": "Update flights in an airline reservation.",
    "update_reservation_baggages": "Update checked baggage counts in an airline reservation.",
    "ocr_extract_text": "Extract text from an image using OCR/QNN-style tooling.",
    "extract_tau_slots": "Use an LLM to extract structured slots from a tau-bench user request.",
    "match_retail_product_variant": "Use an LLM to match a product variant requested by the user.",
    "rank_airline_flight_options": "Use an LLM to rank flight options against user preferences.",
    "detect_confirmation_or_revision": "Use an LLM to detect whether the user confirms or revises an action.",
    "resolve_payment_preference": "Use an LLM to resolve payment preference from user wording and context.",
}

MUTATING_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "cancel_order",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
    "update_reservation_flights",
    "update_reservation_baggages",
}

TOOL_OUTPUTS = {
    "find_user_by_contact": ["user", "user_id"],
    "find_user_by_name": ["user", "user_id"],
    "find_user_id_by_email": ["user", "user_id"],
    "find_user_id_by_name_zip": ["user", "user_id"],
    "get_user_details": ["user", "user_id"],
    "get_order_details": ["order", "order_id", "reservation", "reservation_id"],
    "get_reservation_details": ["order", "order_id", "reservation", "reservation_id"],
    "get_product_details": ["product"],
    "calculate": ["calculation", "calculation_result"],
    "search_direct_flight": ["flight_options", "search_result"],
    "book_reservation": ["reservation_result", "booking_result"],
    "cancel_reservation": ["reservation_result", "cancellation_result"],
    "cancel_order": ["cancellation_result"],
    "return_delivered_order_items": ["return_result"],
    "exchange_delivered_order_items": ["exchange_result"],
    "modify_pending_order_items": ["order_modification_result"],
    "modify_pending_order_address": ["address_modification_result"],
    "modify_pending_order_payment": ["payment_modification_result"],
    "modify_user_address": ["user_address_result"],
    "update_reservation_flights": ["reservation_update_result"],
    "update_reservation_baggages": ["baggage_update_result"],
    "ocr_extract_text": ["ocr_text"],
    "extract_tau_slots": ["tau_slots"],
    "match_retail_product_variant": ["product_match"],
    "rank_airline_flight_options": ["ranked_flight_options"],
    "detect_confirmation_or_revision": ["confirmation_signal"],
    "resolve_payment_preference": ["payment_preference"],
}


def available_tools() -> list[dict[str, Any]]:
    tools = []
    for name in sorted(ALL_TOOLS):
        tools.append(
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS.get(name, f"Execute the MOBench tool named {name}."),
                "kind": tool_kind(name),
                "parameters": parameter_schema(name),
                "outputs": TOOL_OUTPUTS.get(name, []),
                "side_effects": side_effects(name),
                "llm_implementable": name in LLM_BUSINESS_TOOLS or name in LLM_TAU_TOOLS,
            }
        )
    return tools


def tool_names() -> set[str]:
    return set(ALL_TOOLS)


def parameter_schema(name: str) -> list[dict[str, str]]:
    example = example_parameters(name)
    return [
        {
            "name": key,
            "example": "" if value is None else str(value),
            "description": f"Parameter {key} for {name}.",
        }
        for key, value in example.items()
    ]


def tool_kind(name: str) -> str:
    if name in QNN_TOOLS:
        return "dnn"
    if name in RETAIL_TOOLS:
        return "app"
    if name in LLM_TAU_TOOLS:
        return "app"
    return "app"


def side_effects(name: str) -> list[str]:
    if name in MUTATING_TOOLS:
        return ["mutates benchmark state"]
    return []
