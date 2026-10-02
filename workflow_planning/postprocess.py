"""Post-processing helpers for generated workflow JSON."""

from __future__ import annotations

import copy
import re
from collections import deque
from typing import Any


TOOL_KINDS = {"app", "dnn"}
LLM_TAU_ENTRYPOINTS = {
    "extract_tau_slots",
    "match_retail_product_variant",
    "rank_airline_flight_options",
    "detect_confirmation_or_revision",
    "resolve_payment_preference",
}


def insert_llm_checker_nodes(workflow_doc: dict[str, Any]) -> dict[str, Any]:
    """Insert LLM checker nodes after every executable tool node.

    The workflow graph keeps tool nodes as tools. For each tool, an LLM checker
    consumes the tool output and emits a validation artifact. If the tool feeds
    downstream tool calls, those downstream tools also wait for the checker.
    The original tool-to-tool data edge is preserved so executable tools still
    receive the concrete artifact they already expect. This models:

    tool call -> LLM checks tool output -> following tool call
    """

    result = copy.deepcopy(workflow_doc)
    for workflow in result.get("workflows", []):
        _insert_for_workflow(workflow)
    notes = result.setdefault("generation_notes", [])
    note = (
        "Post-processed by wfgen: inserted LLM checker nodes after every tool output; "
        "dependent tool calls wait for the corresponding checker."
    )
    if note not in notes:
        notes.append(note)
    return result


def insert_llm_analysis_nodes(workflow_doc: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for inserting LLM checker nodes."""

    return insert_llm_checker_nodes(workflow_doc)


def hydrate_tool_parameters_from_samples(
    workflow_doc: dict[str, Any],
    dataset_data: Any,
) -> dict[str, Any]:
    """Fill missing executable tool parameters from normalized tau-bench samples."""

    result = copy.deepcopy(workflow_doc)
    samples = _samples_from_dataset_data(dataset_data)
    if not samples:
        return result

    for wf_index, workflow in enumerate(result.get("workflows", [])):
        sample = _match_sample(workflow, samples) or samples[min(wf_index, len(samples) - 1)]
        actions = list(sample.get("ground_truth_actions") or [])
        action_offsets: dict[str, int] = {}
        for node in workflow.get("nodes", []):
            if node.get("kind", "app") not in TOOL_KINDS:
                continue
            entrypoint = node.get("entrypoint") or node.get("node_id")
            if entrypoint in LLM_TAU_ENTRYPOINTS:
                _ensure_parameter(node, "sample_id", sample.get("experiment_id") or sample.get("task_id") or "")
                continue
            if node.get("parameters"):
                continue
            action = _next_action(actions, entrypoint, action_offsets)
            if action:
                node["parameters"] = _parameters_from_arguments(action.get("arguments") or {}, entrypoint)

    notes = result.setdefault("generation_notes", [])
    note = "Post-processed by wfgen: hydrated missing tool parameters from tau-bench ground-truth actions."
    if note not in notes:
        notes.append(note)
    return result


def _insert_for_workflow(workflow: dict[str, Any]) -> None:
    nodes = workflow.get("nodes") or []
    edges = workflow.get("edges") or []
    node_by_id = {node["node_id"]: node for node in nodes}
    existing_ids = set(node_by_id)

    tool_to_tool_edges = [
        edge
        for edge in edges
        if _is_tool(node_by_id.get(edge.get("from_node")))
        and _is_tool(node_by_id.get(edge.get("to_node")))
    ]
    tool_ids = [
        node["node_id"]
        for node in nodes
        if _is_tool(node)
    ]

    new_nodes: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    for src in tool_ids:
        downstream_edges = [edge for edge in tool_to_tool_edges if edge["from_node"] == src]
        checker_id = _unique_id(f"check_after_{src}", existing_ids)
        existing_ids.add(checker_id)
        src_node = node_by_id[src]
        src_artifact = _first_output_artifact(src_node, downstream_edges)
        checker_artifact = {
            "name": f"{checker_id}_validation",
            "artifact_type": "llm_output_check",
            "description": f"LLM validation of {src} tool output.",
            "codec": "json",
            "max_bytes": 65536,
        }
        new_nodes.append(
            {
                "node_id": checker_id,
                "kind": "llm",
                "label": f"Check {src}",
                "purpose": (
                    "Validate the previous tool output for schema, consistency, "
                    "missing fields, and downstream usability."
                ),
                "entrypoint": "llama_server_chat",
                "parameters": [
                    {
                        "name": "instruction",
                        "value": (
                            "Check the upstream tool output. Return compact JSON with: "
                            "verdict, issues, missing_fields, consistency_notes, confidence, "
                            "and recommended_next_tool_inputs. Do not invent IDs."
                        ),
                        "description": "Prompt instruction for the output checker node.",
                    }
                ],
                "inputs": [src_artifact],
                "outputs": [checker_artifact],
                "preferred_backend": "npu",
                "batch_size": 1,
                "side_effects": [],
            }
        )
        new_edges.append(
                {
                    "from_node": src,
                    "to_node": checker_id,
                    "artifact": src_artifact,
                    "reason": "The LLM checker validates this tool output.",
                }
            )
        for edge in downstream_edges:
            new_edges.append(
                {
                    "from_node": checker_id,
                    "to_node": edge["to_node"],
                    "artifact": checker_artifact,
                    "reason": "Downstream tool waits for LLM validation of upstream tool output.",
                }
            )

    if not new_nodes:
        _refresh_execution_hints(workflow)
        return

    workflow["nodes"] = nodes + new_nodes
    workflow["edges"] = edges + new_edges
    _refresh_execution_hints(workflow)


def _samples_from_dataset_data(dataset_data: Any) -> list[dict[str, Any]]:
    if isinstance(dataset_data, dict):
        samples = dataset_data.get("samples")
        if isinstance(samples, list):
            return [sample for sample in samples if isinstance(sample, dict)]
        if dataset_data.get("ground_truth_actions"):
            return [dataset_data]
    if isinstance(dataset_data, list):
        return [sample for sample in dataset_data if isinstance(sample, dict)]
    return []


def _match_sample(workflow: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    source_ids = {str(item) for item in workflow.get("source_task_ids") or []}
    workflow_text = " ".join(
        str(workflow.get(key, ""))
        for key in ("workflow_id", "name", "description", "user_goal")
    )
    for sample in samples:
        candidates = {
            str(sample.get("experiment_id", "")),
            str(sample.get("task_id", "")),
            str(sample.get("selection_index", "")),
        }
        if source_ids & candidates:
            return sample
        if sample.get("experiment_id") and str(sample["experiment_id"]) in workflow_text:
            return sample
    return None


def _next_action(
    actions: list[dict[str, Any]],
    entrypoint: str,
    offsets: dict[str, int],
) -> dict[str, Any] | None:
    normalized_entrypoint = _canonical_entrypoint(entrypoint)
    start = offsets.get(normalized_entrypoint, 0)
    for idx in range(start, len(actions)):
        action = actions[idx]
        if _canonical_entrypoint(action.get("name", "")) == normalized_entrypoint:
            offsets[normalized_entrypoint] = idx + 1
            return action
    return None


def _canonical_entrypoint(entrypoint: str) -> str:
    aliases = {
        "find_user_by_name": "find_user_id_by_name_zip",
        "find_user_by_contact": "find_user_id_by_email",
        "get_reservation_details": "get_order_details",
    }
    return aliases.get(entrypoint, entrypoint)


def _parameters_from_arguments(args: dict[str, Any], entrypoint: str) -> list[dict[str, str]]:
    return [
        {
            "name": key,
            "value": _stringify_parameter(value),
            "description": f"{entrypoint} argument {key} hydrated from tau-bench ground truth.",
        }
        for key, value in args.items()
    ]


def _stringify_parameter(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json_dumps(value)
    return "" if value is None else str(value)


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=True)


def _ensure_parameter(node: dict[str, Any], name: str, value: Any) -> None:
    params = node.setdefault("parameters", [])
    if any(param.get("name") == name for param in params):
        return
    params.append(
        {
            "name": name,
            "value": _stringify_parameter(value),
            "description": f"{name} supplied from selected tau-bench sample.",
        }
    )


def _is_tool(node: dict[str, Any] | None) -> bool:
    return bool(node and node.get("kind", "app") in TOOL_KINDS)


def _first_output_artifact(node: dict[str, Any], outgoing_edges: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = node.get("outputs") or []
    if outputs:
        return copy.deepcopy(outputs[0])
    if outgoing_edges:
        return copy.deepcopy(outgoing_edges[0].get("artifact") or _default_artifact(node["node_id"]))
    return _default_artifact(node["node_id"])


def _default_artifact(node_id: str) -> dict[str, Any]:
    return {
        "name": f"{node_id}_output",
        "artifact_type": "tool_output",
        "description": f"Output from {node_id}.",
        "codec": "json",
        "max_bytes": 65536,
    }


def _unique_id(base: str, existing: set[str]) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "checker_node"
    if clean not in existing:
        return clean
    idx = 2
    while f"{clean}_{idx}" in existing:
        idx += 1
    return f"{clean}_{idx}"


def _refresh_execution_hints(workflow: dict[str, Any]) -> None:
    hints = workflow.setdefault("execution_hints", {})
    hints["parallel_waves"] = _topological_waves(workflow)
    hints["critical_path"] = _critical_path(workflow)
    rationale = hints.get("offloading_rationale") or ""
    addition = "LLM checker nodes validate every tool output and gate dependent tool calls."
    hints["offloading_rationale"] = f"{rationale} {addition}".strip()


def _topological_waves(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    node_ids = [node["node_id"] for node in workflow.get("nodes", [])]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in workflow.get("edges", []):
        outgoing[edge["from_node"]].append(edge["to_node"])
        indegree[edge["to_node"]] += 1

    ready = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    waves: list[dict[str, Any]] = []
    while ready:
        wave_ids = list(ready)
        ready.clear()
        waves.append(
            {
                "wave_index": len(waves),
                "node_ids": wave_ids,
                "reason": "Derived after inserting LLM tool-output checker dependencies.",
            }
        )
        for node_id in wave_ids:
            for dst in outgoing[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)

    if sum(len(wave["node_ids"]) for wave in waves) != len(node_ids):
        raise ValueError(f"Workflow has a cycle after post-processing: {workflow.get('workflow_id')}")
    return waves


def _critical_path(workflow: dict[str, Any]) -> list[str]:
    node_ids = [node["node_id"] for node in workflow.get("nodes", [])]
    if not node_ids:
        return []
    incoming = {node_id: [] for node_id in node_ids}
    for edge in workflow.get("edges", []):
        incoming[edge["to_node"]].append(edge["from_node"])
    terminal_ids = [node_id for node_id in node_ids if not any(edge["from_node"] == node_id for edge in workflow.get("edges", []))]
    current = terminal_ids[-1] if terminal_ids else node_ids[-1]
    path = [current]
    while incoming.get(current):
        current = incoming[current][0]
        path.insert(0, current)
    return path
