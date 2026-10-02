"""Prompt construction for cloud LLM workflow generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..wfgen_core.schema import SYSTEM_PROMPT, WORKFLOW_SCHEMA
from .artifacts import write_json, write_text
from .tool_registry import available_tools


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt_text: str
    user_prompt_payload: dict[str, Any]
    tool_list: list[dict[str, Any]]
    response_schema: dict[str, Any]
    artifacts: dict[str, str]


def build_prompt_bundle(
    selected_payload: dict[str, Any],
    *,
    workflow_count: int,
    extra_instruction: str | None = None,
) -> PromptBundle:
    tools = available_tools()
    response_schema = WORKFLOW_SCHEMA
    payload = {
        "task": "Generate executable MOBench workflow JSON from selected tau-bench samples.",
        "workflow_count": workflow_count,
        "selected_samples": selected_payload,
        "user_requests": user_requests(selected_payload),
        "available_tools": tools,
        "output_schema_name": "wfgen_workflow_spec",
        "output_schema": response_schema,
        "generation_rules": generation_rules(),
        "extra_instruction": extra_instruction or "",
    }
    system_prompt = cloud_system_prompt()
    user_prompt_text = json.dumps(payload, indent=2, ensure_ascii=False)
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt_text=user_prompt_text,
        user_prompt_payload=payload,
        tool_list=tools,
        response_schema=response_schema,
        artifacts={},
    )


def build_sample_prompt_bundle(
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
            "method": "single_sample_cloud_generation",
            "count": 1,
            "indices": [sample.get("selection_index")],
        },
        "samples": [sample],
    }
    return build_prompt_bundle(
        sample_payload,
        workflow_count=1,
        extra_instruction=extra_instruction,
    )


def write_prompt_artifacts(bundle: PromptBundle, output_dir: Path) -> PromptBundle:
    prompt_dir = output_dir / "prompts"
    artifacts = {
        "system_prompt": str(write_text(prompt_dir / "system_prompt.txt", bundle.system_prompt)),
        "user_prompt_json": str(write_json(prompt_dir / "user_prompt.json", bundle.user_prompt_payload)),
        "user_prompt_text": str(write_text(prompt_dir / "user_prompt.txt", bundle.user_prompt_text)),
        "tool_list": str(write_json(prompt_dir / "tool_list.json", bundle.tool_list)),
        "response_schema": str(write_json(prompt_dir / "response_schema.json", bundle.response_schema, ensure_ascii=True)),
    }
    return PromptBundle(
        system_prompt=bundle.system_prompt,
        user_prompt_text=bundle.user_prompt_text,
        user_prompt_payload=bundle.user_prompt_payload,
        tool_list=bundle.tool_list,
        response_schema=bundle.response_schema,
        artifacts=artifacts,
    )


def cloud_system_prompt() -> str:
    return (
        SYSTEM_PROMPT
        + "\n\n"
        + "Cloud generation mode rules:\n"
        + "- Return only JSON matching the wfgen_workflow_spec schema.\n"
        + "- Generate exactly one workflow per selected sample.\n"
        + "- Use only entrypoints from available_tools for app/dnn tool nodes.\n"
        + "- Preserve tau-bench action intent and concrete arguments when they are provided.\n"
        + "- Do not invent ids, order numbers, payment ids, user ids, or product ids.\n"
        + "- Do not add LLM checker nodes after tools; MOBench post-processing will add them.\n"
        + "- Prefer parallel edges only when true data dependencies allow parallel execution.\n"
    )


def generation_rules() -> list[str]:
    return [
        "Use schema_version wfgen.v1.",
        "The top-level workflows array length must equal workflow_count.",
        "Each workflow source_task_ids must include the tau-bench task_id and selection_index.",
        "Each tool node kind must be app or dnn, with entrypoint from available_tools.",
        "Use llm nodes only for reasoning that is not already represented by a tool.",
        "App/tool parameters must be name/value/description triples.",
        "Edges must represent data dependencies between existing node ids.",
        "Execution hints can be approximate; local validation will recompute checker waves after post-processing.",
        "Final outputs should be the mutation result artifacts when the workflow mutates state.",
    ]


def user_requests(selected_payload: dict[str, Any]) -> list[dict[str, Any]]:
    requests = []
    for sample in selected_payload.get("samples") or []:
        scenario = sample.get("user_scenario") or {}
        instructions = scenario.get("instructions") or {}
        requests.append(
            {
                "selection_index": sample.get("selection_index"),
                "task_id": sample.get("task_id"),
                "domain": sample.get("domain"),
                "request": instructions.get("reason_for_call") or "",
                "ground_truth_actions": sample.get("ground_truth_actions") or [],
                "communicate_info": sample.get("ground_truth_communicate_info") or [],
                "nl_assertions": sample.get("ground_truth_nl_assertions") or [],
            }
        )
    return requests
