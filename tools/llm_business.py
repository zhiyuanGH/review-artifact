"""LLM-backed implementations for read-only tau-bench business tools.

These remain logical MOBench tool nodes. The script adapter prepares exact
benchmark evidence, then a llama backend emits the tool artifact JSON.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .abi import fail, ok
from .llm_tau import (
    LlamaCliClient,
    LlamaServerClient,
    compact_response,
    extract_chat_text,
    parse_json_object,
    truthy,
)
from .tau_retail import DEFAULT_DATASET_PATH, run_retail_tool


LLM_BUSINESS_TOOLS = {
    "find_user_by_contact",
    "find_user_by_name",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
    "get_order_details",
    "get_reservation_details",
    "get_product_details",
    "search_direct_flight",
}


def run_llm_business_tool(
    entrypoint: str,
    request: dict[str, Any],
    *,
    dataset_path: str | Path | None = None,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    if entrypoint not in LLM_BUSINESS_TOOLS:
        return fail(f"Unsupported LLM-backed business tool: {entrypoint}", started_at=started_at)

    tool_cfg = request.get("tool_config") or {}
    reference_request = dict(request)
    reference_request["tool_config"] = strip_llm_runtime_config(tool_cfg)
    reference = run_retail_tool(
        entrypoint,
        reference_request,
        dataset_path=dataset_path or tool_cfg.get("dataset_path") or DEFAULT_DATASET_PATH,
        state_path=state_path or tool_cfg.get("state_path"),
    )
    if not reference.get("success"):
        return reference

    try:
        runtime = str(tool_cfg.get("tool_runtime") or tool_cfg.get("llm_runtime") or "llama_cli")
        if runtime == "mock" or truthy(
            tool_cfg.get("mock_llm") or tool_cfg.get("mock_business_llm") or os.environ.get("WFGEN_LLM_TOOL_MOCK")
        ):
            return ok(
                artifacts=reference.get("artifacts", {}),
                raw={
                    "mode": "mock_llm_business_tool",
                    "entrypoint": entrypoint,
                    "parameters": request.get("parameters") or {},
                    "reference_artifacts": reference.get("artifacts", {}),
                },
                started_at=started_at,
            )

        messages = build_business_messages(entrypoint, request, reference)
        if runtime == "llama_cli":
            client = LlamaCliClient.from_config(tool_cfg)
            response = client.chat(messages)
            text = response["text"]
            parsed = parse_json_object(text)
            raw = {
                "mode": "llama_cli_business_tool",
                "entrypoint": entrypoint,
                "model": client.model_path,
                "target": client.target,
                "response_text": text,
                "stdout": response.get("stdout", ""),
                "stderr": response.get("stderr", ""),
                "returncode": response.get("returncode"),
                "command": response.get("command"),
                "prompt_file": response.get("prompt_file"),
                "reference_artifacts": reference.get("artifacts", {}),
            }
        elif runtime == "llama_server":
            client = LlamaServerClient.from_config(tool_cfg)
            response = client.chat(messages)
            text = extract_chat_text(response)
            parsed = parse_json_object(text)
            raw = {
                "mode": "llama_server_business_tool",
                "entrypoint": entrypoint,
                "model": client.model,
                "url": client.url,
                "response_text": text,
                "response": compact_response(response),
                "reference_artifacts": reference.get("artifacts", {}),
            }
        else:
            raise ValueError(f"Unsupported business tool runtime: {runtime}")

        artifacts = normalize_business_artifacts(parsed, reference.get("artifacts", {}))
        return ok(artifacts=artifacts, raw=raw, started_at=started_at)
    except Exception as exc:
        return fail(
            str(exc),
            raw={
                "mode": "llm_business_tool",
                "entrypoint": entrypoint,
                "parameters": request.get("parameters") or {},
                "reference_artifacts": reference.get("artifacts", {}),
            },
            started_at=started_at,
        )


def strip_llm_runtime_config(tool_cfg: dict[str, Any]) -> dict[str, Any]:
    runtime_keys = {
        "implementation",
        "tool_runtime",
        "llm_runtime",
        "mock_llm",
        "mock_business_llm",
        "url",
        "model",
        "model_path",
        "llama_model_path",
        "llama_cli_target",
        "adb",
        "device_serial",
        "adb_serial",
        "remote_dir",
        "remote_prompt_dir",
        "binary",
        "llama_device",
        "ngl",
        "hexagon_env",
        "max_tokens",
        "temperature",
        "llama_timeout",
        "timeout_seconds",
        "json_schema",
        "extra_args",
    }
    return {key: value for key, value in tool_cfg.items() if key not in runtime_keys}


def build_business_messages(entrypoint: str, request: dict[str, Any], reference: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are implementing a MOBench tool node through local LLM inference. "
        "This is still a tool call, not a reasoning node. "
        "Return only one JSON object. Do not include markdown or commentary."
    )
    payload = {
        "tool": entrypoint,
        "parameters": request.get("parameters") or {},
        "upstream": request.get("upstream") or {},
        "candidate_artifacts": reference.get("artifacts", {}),
        "task": (
            "Return the exact tool output artifacts for this read-only lookup/detail tool. "
            "Use candidate_artifacts as the only evidence. Preserve JSON keys and values."
        ),
        "expected_shape": {"artifacts": reference.get("artifacts", {})},
    }
    user = "\n".join(
        [
            f"Tool: {entrypoint}",
            "Input/evidence JSON:",
            json.dumps(payload, ensure_ascii=True),
            "Answer JSON only:",
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_business_artifacts(parsed: dict[str, Any], reference_artifacts: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed.get("artifacts"), dict):
        artifacts = parsed["artifacts"]
    elif isinstance(parsed.get("output"), dict):
        artifacts = parsed["output"]
    else:
        artifacts = parsed
    if not isinstance(artifacts, dict):
        artifacts = {}
    repaired = dict(artifacts)
    for key, value in reference_artifacts.items():
        if key not in repaired or empty_value(repaired[key]):
            repaired[key] = value
    if reference_artifacts:
        return {key: repaired[key] for key in reference_artifacts if key in repaired}
    return repaired


def empty_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}
