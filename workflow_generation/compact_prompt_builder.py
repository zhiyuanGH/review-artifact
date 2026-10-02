"""Compact chain prompt builder for cloud LLM workflow planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import write_json, write_text
from .chain_registry import compact_catalog
from .prompt_builder import PromptBundle


COMPACT_SYSTEM_PROMPT = (
    "Generate compact MOBench chain workflow JSON only; use known chain/tool ids; "
    "do not invent user/order/product/payment ids."
)

COMPACT_CHAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": ["wfgen.chain.v1"]},
        "workflows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "workflow_id": {"type": "string"},
                    "name": {"type": "string"},
                    "source_task_ids": {"type": "array", "items": {"type": "string"}},
                    "user_goal": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "step_id": {"type": "string"},
                                "kind": {"type": "string", "enum": ["chain", "tool"]},
                                "chain_id": {"type": "string"},
                                "tool_name": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "purpose": {"type": "string"},
                            },
                            "required": ["step_id", "kind", "depends_on", "purpose"],
                        },
                    },
                    "final_outputs": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["workflow_id", "name", "source_task_ids", "user_goal", "steps", "final_outputs", "notes"],
            },
        },
        "generation_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["schema_version", "workflows", "generation_notes"],
}


def build_compact_prompt_bundle(
    selected_payload: dict[str, Any],
    *,
    workflow_count: int,
    extra_instruction: str | None = None,
) -> PromptBundle:
    catalog = compact_catalog()
    payload = {
        "task": "Return one compact chain workflow per sample.",
        "workflow_count": workflow_count,
        "samples": compact_samples(selected_payload),
        "catalog": catalog,
        "schema_name": "wfgen_chain_workflow_spec",
        "response_schema": COMPACT_CHAIN_SCHEMA,
        "rules": compact_generation_rules(),
        "extra_instruction": extra_instruction or "",
    }
    user_prompt_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return PromptBundle(
        system_prompt=COMPACT_SYSTEM_PROMPT,
        user_prompt_text=user_prompt_text,
        user_prompt_payload=payload,
        tool_list=catalog["chains"] + catalog["standalone_tools"],
        response_schema=COMPACT_CHAIN_SCHEMA,
        artifacts={},
    )


def build_compact_sample_prompt_bundle(
    selected_payload: dict[str, Any],
    sample: dict[str, Any],
    *,
    extra_instruction: str | None = None,
) -> PromptBundle:
    sample_payload = {
        "source_dataset": selected_payload.get("source_dataset"),
        "config": selected_payload.get("config"),
        "split": selected_payload.get("split"),
        "selection": {
            "method": "single_sample_compact_chain_generation",
            "count": 1,
            "indices": [sample.get("selection_index")],
        },
        "samples": [sample],
    }
    return build_compact_prompt_bundle(
        sample_payload,
        workflow_count=1,
        extra_instruction=extra_instruction,
    )


def write_compact_prompt_artifacts(bundle: PromptBundle, output_dir: Path) -> PromptBundle:
    prompt_dir = output_dir / "prompts"
    catalog = bundle.user_prompt_payload.get("catalog") or {}
    stats = prompt_stats(bundle)
    artifacts = {
        "system_prompt": str(write_text(prompt_dir / "system_prompt.txt", bundle.system_prompt)),
        "user_prompt_json": str(write_json(prompt_dir / "user_prompt.json", bundle.user_prompt_payload)),
        "user_prompt_text": str(write_text(prompt_dir / "user_prompt.txt", bundle.user_prompt_text)),
        "tool_list": str(write_json(prompt_dir / "tool_list.json", bundle.tool_list)),
        "chain_registry": str(write_json(prompt_dir / "chain_registry.json", catalog)),
        "response_schema": str(write_json(prompt_dir / "response_schema.json", bundle.response_schema, ensure_ascii=True)),
        "prompt_stats": str(write_json(prompt_dir / "prompt_stats.json", stats)),
    }
    return PromptBundle(
        system_prompt=bundle.system_prompt,
        user_prompt_text=bundle.user_prompt_text,
        user_prompt_payload=bundle.user_prompt_payload,
        tool_list=bundle.tool_list,
        response_schema=bundle.response_schema,
        artifacts=artifacts,
    )


def compact_samples(selected_payload: dict[str, Any]) -> list[dict[str, Any]]:
    samples = []
    for sample in selected_payload.get("samples") or []:
        scenario = sample.get("user_scenario") or {}
        instructions = scenario.get("instructions") or {}
        samples.append(
            {
                "selection_index": sample.get("selection_index"),
                "task_id": sample.get("task_id"),
                "domain": sample.get("domain"),
                "request": instructions.get("reason_for_call") or "",
                "ground_truth_actions": compact_actions(sample.get("ground_truth_actions") or []),
            }
        )
    return samples


def compact_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": action.get("name"),
            "arguments": action.get("arguments") or {},
        }
        for action in actions
        if isinstance(action, dict)
    ]


def compact_generation_rules() -> list[str]:
    return [
        "Use schema_version wfgen.chain.v1.",
        "Return exactly workflow_count workflows.",
        "Use chain steps for absorbed primitive tools; use tool steps only for standalone catalog tools.",
        "Keep steps high-level; do not expand chain internals.",
        "Copy real ids only from the sample action arguments.",
        "MOBench expands chains into real tool nodes and inserts checker nodes after generation.",
    ]


def prompt_stats(bundle: PromptBundle) -> dict[str, Any]:
    catalog = bundle.user_prompt_payload.get("catalog") or {}
    counts = catalog.get("counts") or {}
    payload = bundle.user_prompt_payload
    return {
        "prompt_builder": "compact-chain",
        "system_chars": len(bundle.system_prompt),
        "user_prompt_chars": len(bundle.user_prompt_text),
        "schema_chars": len(json.dumps(bundle.response_schema, ensure_ascii=True, separators=(",", ":"))),
        "catalog_chars": len(json.dumps(catalog, ensure_ascii=False, separators=(",", ":"))),
        "sample_chars": len(json.dumps(payload.get("samples") or [], ensure_ascii=False, separators=(",", ":"))),
        "catalog_counts": counts,
    }
