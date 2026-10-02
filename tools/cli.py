"""Command line adapter for MOBench JSON tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .abi import read_request, write_response
from .llm_business import LLM_BUSINESS_TOOLS, run_llm_business_tool
from .llm_tau import LLM_TAU_TOOLS, run_llm_tau_tool
from .qnn_ocr import run_ocr_extract_text
from .tau_retail import DEFAULT_DATASET_PATH, run_retail_tool


RETAIL_TOOLS = {
    "find_user_by_contact",
    "find_user_by_name",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_reservation_details",
    "get_product_details",
    "calculate",
    "search_direct_flight",
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
QNN_TOOLS = {"ocr_extract_text"}
ALL_TOOLS = sorted(RETAIL_TOOLS | QNN_TOOLS | LLM_TAU_TOOLS)


def run_tool(entrypoint: str, request: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    request = dict(request)
    request["entrypoint"] = entrypoint
    if args.params:
        request["parameters"] = json.loads(args.params)
    if entrypoint in RETAIL_TOOLS:
        if should_use_llm_business_tool(entrypoint, request):
            return run_llm_business_tool(
                entrypoint,
                request,
                dataset_path=args.dataset,
                state_path=args.state,
            )
        return run_retail_tool(
            entrypoint,
            request,
            dataset_path=args.dataset,
            state_path=args.state,
        )
    if entrypoint == "ocr_extract_text":
        return run_ocr_extract_text(request)
    if entrypoint in LLM_TAU_TOOLS:
        return run_llm_tau_tool(entrypoint, request)
    return {
        "success": False,
        "artifacts": {},
        "error": f"Unknown tool: {entrypoint}",
        "raw": {},
    }


def should_use_llm_business_tool(entrypoint: str, request: dict[str, Any]) -> bool:
    if entrypoint not in LLM_BUSINESS_TOOLS:
        return False
    cfg = request.get("tool_config") or {}
    implementation = str(cfg.get("implementation") or "").lower()
    runtime = str(cfg.get("tool_runtime") or cfg.get("llm_runtime") or "").lower()
    return implementation == "llm" or runtime in {"llama_cli", "llama_server", "mock"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MOBench tool via JSON ABI.")
    parser.add_argument("entrypoint", choices=ALL_TOOLS)
    parser.add_argument("--input", help="Request JSON path. Defaults to stdin.")
    parser.add_argument("--output", help="Response JSON path. Defaults to stdout.")
    parser.add_argument("--params", help="Inline JSON object used as request.parameters.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Tau-bench normalized sample JSON for retail tools.",
    )
    parser.add_argument("--state", help="State JSON path for state-mutating tools.")
    parser.add_argument(
        "--write-example-request",
        type=Path,
        help="Write a sample request JSON for this entrypoint and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_example_request:
        request = {
            "entrypoint": args.entrypoint,
            "parameters": example_parameters(args.entrypoint),
            "upstream": {},
        }
        args.write_example_request.parent.mkdir(parents=True, exist_ok=True)
        args.write_example_request.write_text(
            json.dumps(request, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return 0

    request = read_request(args.input)
    response = run_tool(args.entrypoint, request, args)
    write_response(response, args.output)
    return 0 if response.get("success") else 1


def example_parameters(entrypoint: str) -> dict[str, Any]:
    examples = {
        "find_user_by_contact": {"email": "juan.lopez7539@example.com"},
        "find_user_by_name": {"first_name": "Yusuf", "last_name": "Li", "zip": "91148"},
        "get_user_details": {"user_id": "ava_smith_9007"},
        "get_order_details": {"order_id": "#W4466964"},
        "get_product_details": {"product_id": "4760268021"},
        "cancel_order": {"order_id": "#W4466964", "reason": "ordered by mistake"},
        "return_delivered_order_items": {
            "order_id": "#W9183908",
            "item_ids": ["7407838442"],
            "payment_method_id": "gift_card_8070316",
        },
        "exchange_delivered_order_items": {
            "order_id": "#W6818211",
            "item_ids": ["2913673670"],
            "new_item_ids": ["1657832319"],
            "payment_method_id": "credit_card_7598260",
        },
        "modify_pending_order_items": {
            "order_id": "#W6002467",
            "item_ids": ["1151293680"],
            "new_item_ids": ["5886093635"],
            "payment_method_id": "paypal_1808675",
        },
        "modify_pending_order_address": {
            "order_id": "#W7752859",
            "address1": "740 Maple Drive",
            "city": "Charlotte",
            "state": "NC",
            "country": "USA",
            "zip": "28221",
        },
        "modify_pending_order_payment": {"order_id": "#W6207110", "payment_method_id": "paypal_5377635"},
        "modify_user_address": {
            "user_id": "yara_patel_8545",
            "address1": "258 Willow Lane",
            "city": "Fort Worth",
            "state": "TX",
            "country": "USA",
            "zip": "76130",
        },
        "update_reservation_flights": {
            "reservation_id": "ZO9C7T",
            "cabin": "business",
            "flights": [{"flight_number": "HAT021", "date": "2024-05-21"}],
            "payment_id": "gift_card_7397399",
        },
        "update_reservation_baggages": {
            "reservation_id": "JU7PH1",
            "total_baggages": 2,
            "nonfree_baggages": 0,
            "payment_id": "gift_card_8462123",
        },
        "ocr_extract_text": {"image_path": "data/example_image.jpg"},
        "extract_tau_slots": {"sample_id": "retail_exchange_keyboard_thermostat"},
        "match_retail_product_variant": {"sample_id": "retail_exchange_keyboard_thermostat"},
        "rank_airline_flight_options": {"sample_id": "airline_fastest_return_smallest_gift_card"},
        "detect_confirmation_or_revision": {"sample_id": "retail_exchange_keyboard_thermostat"},
        "resolve_payment_preference": {"sample_id": "airline_fastest_return_smallest_gift_card"},
    }
    return examples.get(entrypoint, {})


if __name__ == "__main__":
    raise SystemExit(main())
