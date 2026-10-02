"""Compact chain catalog for MOBench prompt compression."""

from __future__ import annotations

import copy
from typing import Any

from .tool_registry import available_tools


CHAIN_DEFINITIONS: list[dict[str, Any]] = [
    {
        "chain_id": "retail_order_context",
        "purpose": "Find a retail user and read the target order.",
        "tools": ["find_user_by_contact", "get_order_details"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "order_id"],
        "final_output": "order",
    },
    {
        "chain_id": "retail_cancel_order",
        "purpose": "Find user, read order, then cancel the order.",
        "tools": ["find_user_by_contact", "get_order_details", "cancel_order"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "order_id", "cancellation reason if present"],
        "final_output": "cancellation_result",
    },
    {
        "chain_id": "retail_return_items",
        "purpose": "Find user, read order, then return delivered items.",
        "tools": ["find_user_by_contact", "get_order_details", "return_delivered_order_items"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "order_id", "item_ids"],
        "final_output": "return_result",
    },
    {
        "chain_id": "retail_exchange_items",
        "purpose": "Find user, read order, read replacement products, then exchange delivered items.",
        "tools": ["find_user_by_contact", "get_order_details", "get_product_details*", "exchange_delivered_order_items"],
        "repeatable_tools": ["get_product_details*"],
        "required_inputs": ["phone/email/name/contact", "order_id", "item_ids", "new_item_ids"],
        "final_output": "exchange_result",
    },
    {
        "chain_id": "retail_modify_items",
        "purpose": "Find user, read pending order, read new products, then modify pending items.",
        "tools": ["find_user_by_contact", "get_order_details", "get_product_details*", "modify_pending_order_items"],
        "repeatable_tools": ["get_product_details*"],
        "required_inputs": ["phone/email/name/contact", "order_id", "item_ids", "new_item_ids"],
        "final_output": "order_modification_result",
    },
    {
        "chain_id": "retail_modify_payment",
        "purpose": "Find user, read pending order, then update payment method.",
        "tools": ["find_user_by_contact", "get_order_details", "modify_pending_order_payment"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "order_id", "payment_method_id"],
        "final_output": "payment_modification_result",
    },
    {
        "chain_id": "retail_modify_order_address",
        "purpose": "Find user, read pending order, then update order shipping address.",
        "tools": ["find_user_by_contact", "get_order_details", "modify_pending_order_address"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "order_id", "address"],
        "final_output": "address_modification_result",
    },
    {
        "chain_id": "retail_modify_user_address",
        "purpose": "Find a retail user and update their saved address.",
        "tools": ["find_user_by_contact", "modify_user_address"],
        "repeatable_tools": [],
        "required_inputs": ["phone/email/name/contact", "address"],
        "final_output": "user_address_result",
    },
    {
        "chain_id": "airline_cancel_reservation",
        "purpose": "Read airline user/reservation context, then cancel the reservation.",
        "tools": ["get_user_details", "get_reservation_details", "cancel_reservation"],
        "repeatable_tools": [],
        "required_inputs": ["user_id", "reservation_id"],
        "final_output": "cancellation_result",
    },
    {
        "chain_id": "airline_update_flights",
        "purpose": "Read airline context, search replacement flights, then update reservation flights.",
        "tools": ["get_user_details", "get_reservation_details", "search_direct_flight*", "update_reservation_flights"],
        "repeatable_tools": ["search_direct_flight*"],
        "required_inputs": ["user_id", "reservation_id", "flight search constraints", "new flights"],
        "final_output": "reservation_update_result",
    },
    {
        "chain_id": "airline_update_baggages",
        "purpose": "Read airline user/reservation context, then update baggage counts.",
        "tools": ["get_user_details", "get_reservation_details", "update_reservation_baggages"],
        "repeatable_tools": [],
        "required_inputs": ["user_id", "reservation_id", "baggage updates"],
        "final_output": "baggage_update_result",
    },
]


def chain_definitions() -> list[dict[str, Any]]:
    return copy.deepcopy(CHAIN_DEFINITIONS)


def chain_ids() -> set[str]:
    return {chain["chain_id"] for chain in CHAIN_DEFINITIONS}


def absorbed_tool_names() -> set[str]:
    names: set[str] = set()
    for chain in CHAIN_DEFINITIONS:
        for tool_name in chain.get("tools") or []:
            names.add(str(tool_name).rstrip("*"))
    return names


def compact_standalone_tools() -> list[dict[str, Any]]:
    absorbed = absorbed_tool_names()
    tools = []
    for tool in available_tools():
        name = str(tool.get("name") or "")
        if name in absorbed:
            continue
        tools.append(short_tool_record(tool))
    return tools


def compact_standalone_tool_names() -> set[str]:
    return {tool["name"] for tool in compact_standalone_tools()}


def compact_catalog() -> dict[str, Any]:
    chains = chain_definitions()
    standalone = compact_standalone_tools()
    original_tool_count = len(available_tools())
    absorbed_count = len(absorbed_tool_names())
    return {
        "chains": chains,
        "standalone_tools": standalone,
        "counts": {
            "original_standalone_tools": original_tool_count,
            "chains": len(chains),
            "remaining_standalone_tools": len(standalone),
            "compact_catalog_entries": len(chains) + len(standalone),
            "absorbed_primitive_tools": absorbed_count,
        },
    }


def short_tool_record(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters") or []
    outputs = tool.get("outputs") or []
    return {
        "name": tool.get("name"),
        "kind": tool.get("kind"),
        "purpose": trim_sentence(str(tool.get("description") or tool.get("name") or "")),
        "params": [
            {
                "name": param.get("name"),
                "type": param.get("type"),
                "required": bool(param.get("required", False)),
            }
            for param in parameters
        ],
        "outputs": [output.get("name") for output in outputs if isinstance(output, dict)],
        "side_effects": tool.get("side_effects") or [],
    }


def trim_sentence(text: str, *, max_chars: int = 96) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "."
