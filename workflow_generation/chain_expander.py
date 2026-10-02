"""Expand compact chain-level MOBench plans into standard workflow JSON."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from ..workflow_planning.tau_bench_builder import build_workflow
from ..workflow_planning.postprocess import hydrate_tool_parameters_from_samples, insert_llm_checker_nodes
from .chain_registry import chain_ids, compact_standalone_tool_names
from .workflow_validator import validate_cloud_workflow


def expand_chain_workflow_document(
    chain_doc: dict[str, Any],
    selected_payload: dict[str, Any],
    *,
    provider: str,
    model: str,
    base_url: str,
    insert_checker_nodes: bool = True,
) -> dict[str, Any]:
    validate_chain_workflow_document(chain_doc, selected_payload)

    samples = selected_payload.get("samples") or []
    compact_workflows = chain_doc.get("workflows") or []
    workflows = []
    for idx, sample in enumerate(samples):
        compact_workflow = compact_workflows[idx]
        workflow = build_workflow(sample)
        workflow = apply_compact_metadata(workflow, compact_workflow, sample, idx)
        workflows.append(workflow)

    workflow_doc = {
        "schema_version": "wfgen.v1",
        "benchmark": {
            "name": "tau-bench-synthetic compact chain cloud generated",
            "description": "Compact chain-level cloud responses expanded into executable MOBench tool workflows.",
            "source_dataset": selected_payload.get("source_dataset", "fuvty/tau-bench-synthetic"),
            "task_count": len(workflows),
            "assumptions": [
                "Cloud generation selects high-level chain/tool steps.",
                "MOBench expands chain steps into primitive executable tools using tau-bench ground-truth action arguments.",
                "Deployment and execution consume only normal wfgen.v1 tool/LLM nodes, never chain nodes.",
            ],
        },
        "workflows": workflows,
        "generation_notes": compact_notes(chain_doc),
        "generation_metadata": {
            "generation_mode": "cloud",
            "prompt_builder": "compact-chain",
            "intermediate_schema_version": chain_doc.get("schema_version"),
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "expanded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }
    workflow_doc = hydrate_tool_parameters_from_samples(workflow_doc, selected_payload)
    if insert_checker_nodes:
        workflow_doc = insert_llm_checker_nodes(workflow_doc)
    ensure_no_chain_nodes(workflow_doc)
    validate_cloud_workflow(workflow_doc, selected_payload)
    return workflow_doc


def validate_chain_workflow_document(chain_doc: dict[str, Any], selected_payload: dict[str, Any]) -> None:
    if chain_doc.get("schema_version") != "wfgen.chain.v1":
        raise ValueError("Compact chain response must use schema_version 'wfgen.chain.v1'.")
    workflows = chain_doc.get("workflows")
    if not isinstance(workflows, list):
        raise ValueError("Compact chain response must contain a workflows list.")
    expected = len(selected_payload.get("samples") or [])
    if expected and len(workflows) != expected:
        raise ValueError(f"Compact chain response returned {len(workflows)} workflows for {expected} selected samples.")
    for workflow_index, workflow in enumerate(workflows):
        validate_chain_workflow(workflow, workflow_index)


def validate_chain_workflow(workflow: dict[str, Any], workflow_index: int) -> None:
    if not isinstance(workflow, dict):
        raise ValueError(f"Compact workflow {workflow_index} is not an object.")
    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"Compact workflow {workflow_index} must contain at least one step.")
    known_chains = chain_ids()
    known_standalone_tools = compact_standalone_tool_names()
    seen_step_ids: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError(f"Compact workflow {workflow_index} contains a non-object step.")
        step_id = str(step.get("step_id") or "")
        if not step_id:
            raise ValueError(f"Compact workflow {workflow_index} contains a step without step_id.")
        if step_id in seen_step_ids:
            raise ValueError(f"Compact workflow {workflow_index} repeats step_id {step_id!r}.")
        seen_step_ids.add(step_id)
        kind = step.get("kind")
        if kind == "chain":
            chain_id = step.get("chain_id")
            if chain_id not in known_chains:
                raise ValueError(f"Compact workflow {workflow_index} uses unknown chain_id {chain_id!r}.")
        elif kind == "tool":
            tool_name = step.get("tool_name")
            if tool_name not in known_standalone_tools:
                raise ValueError(
                    f"Compact workflow {workflow_index} uses tool {tool_name!r}, "
                    "which is not a standalone compact-catalog tool."
                )
        else:
            raise ValueError(f"Compact workflow {workflow_index} has invalid step kind {kind!r}.")
        depends_on = step.get("depends_on")
        if not isinstance(depends_on, list):
            raise ValueError(f"Compact workflow {workflow_index} step {step_id!r} depends_on must be a list.")


def apply_compact_metadata(
    workflow: dict[str, Any],
    compact_workflow: dict[str, Any],
    sample: dict[str, Any],
    workflow_index: int,
) -> dict[str, Any]:
    result = copy.deepcopy(workflow)
    original_id = result.get("workflow_id")
    result["workflow_id"] = compact_expanded_workflow_id(sample, workflow_index)
    if compact_workflow.get("name"):
        result["name"] = compact_workflow["name"]
    if compact_workflow.get("user_goal"):
        result["user_goal"] = compact_workflow["user_goal"]
        result["description"] = compact_workflow["user_goal"]
    source_ids = compact_workflow.get("source_task_ids")
    if isinstance(source_ids, list) and source_ids:
        result["source_task_ids"] = [str(item) for item in source_ids]
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "prompt_builder": "compact-chain",
            "compact_workflow_id": compact_workflow.get("workflow_id"),
            "expanded_from_workflow_id": original_id,
            "selection_index": sample.get("selection_index", workflow_index),
            "task_id": sample.get("task_id"),
            "compact_steps": compact_workflow.get("steps") or [],
            "compact_final_outputs": compact_workflow.get("final_outputs") or [],
            "compact_notes": compact_workflow.get("notes") or [],
        }
    )
    result["metadata"] = metadata
    return result


def compact_notes(chain_doc: dict[str, Any]) -> list[str]:
    notes = [
        "Generated with compact-chain prompt builder.",
        "Chain-level response was expanded into standard MOBench primitive tool nodes before validation.",
    ]
    for note in chain_doc.get("generation_notes") or []:
        text = str(note)
        if text not in notes:
            notes.append(text)
    return notes


def compact_expanded_workflow_id(sample: dict[str, Any], workflow_index: int) -> str:
    domain = safe_identifier(str(sample.get("domain") or "sample"))
    selection = sample.get("selection_index", workflow_index)
    task_id = safe_identifier(str(sample.get("task_id") or f"sample_{workflow_index}"))
    return f"tau_bench_compact_{domain}_{selection}_{task_id}"


def safe_identifier(value: str) -> str:
    clean = "".join(char if char.isalnum() else "_" for char in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "workflow"


def ensure_no_chain_nodes(workflow_doc: dict[str, Any]) -> None:
    offenders = []
    for workflow in workflow_doc.get("workflows") or []:
        for node in workflow.get("nodes") or []:
            if node.get("kind") == "chain" or str(node.get("entrypoint") or "").startswith("chain:"):
                offenders.append(f"{workflow.get('workflow_id')}:{node.get('node_id')}")
    if offenders:
        raise ValueError("Expanded workflow still contains chain nodes: " + ", ".join(offenders))
