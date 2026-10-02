"""Cloud JSON generation stage for MOBench workflow planning."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..workflow_planning.postprocess import hydrate_tool_parameters_from_samples, insert_llm_checker_nodes
from .artifacts import write_json
from .chain_expander import expand_chain_workflow_document
from .cloud_llm import CloudLLMConfig, chat_completion, extract_usage, safe_error
from .prompt_builder import PromptBundle
from .workflow_validator import validate_cloud_workflow


def generate_cloud_workflow_json(
    *,
    config: CloudLLMConfig,
    prompt: PromptBundle,
    selected_payload: dict[str, Any],
    output_dir: Path,
    insert_checker_nodes: bool = True,
    call_label: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    llm_dir = output_dir / "llm"
    started = time.time()
    raw_path = llm_dir / "raw_response.json"
    try:
        print(
            f"[cloud-llm] sending workflow-generation request {label_text(call_label)}model={config.model} "
            f"format=json_schema samples={len(selected_payload.get('samples') or [])} "
            f"prompt_chars={len(prompt.user_prompt_text)}",
            flush=True,
        )
        response = chat_completion(
            config,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt_text,
            response_schema=prompt.response_schema,
            prefer_json_schema=True,
        )
        raw = response.get("raw_response")
        usage = extract_usage(raw)
        print(
            f"[cloud-llm] response received {label_text(call_label)}id={raw.get('id', '') if isinstance(raw, dict) else ''} "
            f"format={response.get('response_format')} wall_ms={(time.time() - started) * 1000.0:.2f} "
            f"prompt_tokens={usage.get('prompt_tokens', '')} "
            f"completion_tokens={usage.get('completion_tokens', '')} "
            f"total_tokens={usage.get('total_tokens', '')}",
            flush=True,
        )
        workflow_doc = parse_workflow_json(response["content"])
        print(f"[cloud-llm] parsed {label_text(call_label)}workflow_count={len(workflow_doc.get('workflows') or [])}", flush=True)
        workflow_doc = hydrate_tool_parameters_from_samples(workflow_doc, selected_payload)
        if insert_checker_nodes:
            workflow_doc = insert_llm_checker_nodes(workflow_doc)
            print(f"[cloud-llm] inserted LLM checker nodes after tool nodes {label_text(call_label).strip()}", flush=True)
        workflow_doc["generation_metadata"] = {
            "generation_mode": "cloud",
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "response_format": response.get("response_format"),
            "generated_at_unix": started,
        }
        validate_cloud_workflow(workflow_doc, selected_payload)
        print(f"[cloud-llm] local validation passed {label_text(call_label).strip()}", flush=True)
        write_json(
            raw_path,
            {
                "status": "ok",
                "wall_ms": (time.time() - started) * 1000.0,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "response_format": response.get("response_format"),
                "fallback_errors": response.get("fallback_errors", []),
                "usage": usage,
                "response_id": raw.get("id") if isinstance(raw, dict) else "",
                "content": response.get("content"),
                "raw_response": raw,
            },
        )
        print(f"[cloud-llm] wrote raw response artifact {label_text(call_label)}{raw_path}", flush=True)
        return workflow_doc, {"raw_response": str(raw_path)}
    except Exception as exc:
        write_json(
            raw_path,
            {
                "status": "error",
                "wall_ms": (time.time() - started) * 1000.0,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "error": safe_error(exc),
            },
        )
        raise


def generate_compact_chain_workflow_json(
    *,
    config: CloudLLMConfig,
    prompt: PromptBundle,
    selected_payload: dict[str, Any],
    output_dir: Path,
    insert_checker_nodes: bool = True,
    call_label: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    llm_dir = output_dir / "llm"
    started = time.time()
    raw_path = llm_dir / "raw_response.json"
    chain_path = llm_dir / "chain_workflow.json"
    expanded_path = llm_dir / "expanded_workflow.json"
    try:
        print(
            f"[cloud-llm] sending compact-chain request {label_text(call_label)}model={config.model} "
            f"format=json_schema samples={len(selected_payload.get('samples') or [])} "
            f"prompt_chars={len(prompt.user_prompt_text)} catalog_entries={len(prompt.tool_list)}",
            flush=True,
        )
        response = chat_completion(
            config,
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt_text,
            response_schema=prompt.response_schema,
            prefer_json_schema=True,
        )
        raw = response.get("raw_response")
        usage = extract_usage(raw)
        print(
            f"[cloud-llm] compact response received {label_text(call_label)}id={raw.get('id', '') if isinstance(raw, dict) else ''} "
            f"format={response.get('response_format')} wall_ms={(time.time() - started) * 1000.0:.2f} "
            f"prompt_tokens={usage.get('prompt_tokens', '')} "
            f"completion_tokens={usage.get('completion_tokens', '')} "
            f"total_tokens={usage.get('total_tokens', '')}",
            flush=True,
        )
        chain_doc = parse_workflow_json(response["content"])
        print(
            f"[cloud-llm] parsed compact-chain {label_text(call_label)}"
            f"workflow_count={len(chain_doc.get('workflows') or [])}",
            flush=True,
        )
        write_json(chain_path, chain_doc, ensure_ascii=True)
        workflow_doc = expand_chain_workflow_document(
            chain_doc,
            selected_payload,
            provider=config.provider,
            model=config.model,
            base_url=config.base_url,
            insert_checker_nodes=insert_checker_nodes,
        )
        print(
            f"[cloud-llm] expanded compact-chain into wfgen.v1 {label_text(call_label)}"
            f"workflow_count={len(workflow_doc.get('workflows') or [])}",
            flush=True,
        )
        write_json(expanded_path, workflow_doc, ensure_ascii=True)
        write_json(
            raw_path,
            {
                "status": "ok",
                "wall_ms": (time.time() - started) * 1000.0,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "prompt_builder": "compact-chain",
                "response_format": response.get("response_format"),
                "fallback_errors": response.get("fallback_errors", []),
                "usage": usage,
                "response_id": raw.get("id") if isinstance(raw, dict) else "",
                "content": response.get("content"),
                "raw_response": raw,
            },
        )
        print(f"[cloud-llm] wrote compact raw response artifact {label_text(call_label)}{raw_path}", flush=True)
        return workflow_doc, {
            "raw_response": str(raw_path),
            "chain_workflow": str(chain_path),
            "expanded_workflow": str(expanded_path),
        }
    except Exception as exc:
        write_json(
            raw_path,
            {
                "status": "error",
                "wall_ms": (time.time() - started) * 1000.0,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "prompt_builder": "compact-chain",
                "error": safe_error(exc),
            },
        )
        raise


def merge_workflow_documents(
    docs: list[dict[str, Any]],
    selected_payload: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    workflows: list[dict[str, Any]] = []
    notes: list[str] = []
    assumptions: list[str] = []
    samples = selected_payload.get("samples") or []
    for doc_index, doc in enumerate(docs):
        sample = samples[doc_index] if doc_index < len(samples) else {}
        for workflow_index, workflow in enumerate(doc.get("workflows") or []):
            workflow = dict(workflow)
            original_id = workflow.get("workflow_id")
            workflow["workflow_id"] = merged_workflow_id(sample, doc_index, workflow_index)
            metadata = dict(workflow.get("metadata") or {})
            if original_id:
                metadata.setdefault("cloud_original_workflow_id", original_id)
            metadata.setdefault("selection_index", sample.get("selection_index", doc_index))
            metadata.setdefault("task_id", sample.get("task_id"))
            workflow["metadata"] = metadata
            workflows.append(workflow)
        notes.extend(str(note) for note in doc.get("generation_notes") or [])
        benchmark = doc.get("benchmark") or {}
        assumptions.extend(str(item) for item in benchmark.get("assumptions") or [])
    merged = {
        "schema_version": "wfgen.v1",
        "benchmark": {
            "name": "tau-bench-synthetic cloud generated",
            "description": "Cloud-LLM generated workflows from selected tau-bench synthetic samples.",
            "source_dataset": selected_payload.get("source_dataset", "fuvty/tau-bench-synthetic"),
            "task_count": len(workflows),
            "assumptions": dedupe(assumptions),
        },
        "workflows": workflows,
        "generation_notes": dedupe(notes + ["Merged from one cloud LLM call per selected sample."]),
        "generation_metadata": {
            "generation_mode": "cloud",
            "generation_granularity": "one_sample_per_prompt",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "sample_count": len(selected_payload.get("samples") or []),
        },
    }
    validate_cloud_workflow(merged, selected_payload)
    return merged


def merged_workflow_id(sample: dict[str, Any], doc_index: int, workflow_index: int) -> str:
    domain = safe_identifier(str(sample.get("domain") or "sample"))
    selection = sample.get("selection_index", doc_index)
    task_id = safe_identifier(str(sample.get("task_id") or f"sample_{doc_index}"))
    suffix = f"_{workflow_index}" if workflow_index else ""
    return f"tau_bench_cloud_{domain}_{selection}_{task_id}{suffix}"


def safe_identifier(value: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "workflow"


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def parse_workflow_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = strip_code_fence(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = scan_first_json_object(text)
    if not isinstance(payload, dict):
        raise ValueError("Cloud LLM output must be a JSON object.")
    return payload


def strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def scan_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("Cloud LLM output did not contain a parseable JSON object.")


def label_text(label: str) -> str:
    return f"[{label}] " if label else ""
