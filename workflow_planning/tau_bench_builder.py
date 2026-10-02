"""Build executable workflow JSON from normalized tau-bench ground-truth actions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path
from typing import Any

try:
    from MOBench.workflow_planning.generate import validate_workflow_semantics
    from MOBench.workflow_planning.postprocess import insert_llm_checker_nodes
except ImportError:
    from workflow_planning.generate import validate_workflow_semantics
    from workflow_planning.postprocess import insert_llm_checker_nodes


USER_ACTIONS = {
    "find_user_by_contact",
    "find_user_by_name",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_user_details",
}
READ_ACTIONS = USER_ACTIONS | {"get_order_details", "get_reservation_details", "get_product_details", "search_direct_flight", "calculate"}
MUTATION_OUTPUTS = {
    "cancel_order": ("cancellation_result", "cancellation_result"),
    "return_delivered_order_items": ("return_result", "return_result"),
    "exchange_delivered_order_items": ("exchange_result", "exchange_result"),
    "modify_pending_order_items": ("order_modification_result", "order_modification_result"),
    "modify_pending_order_address": ("address_modification_result", "address_modification_result"),
    "modify_pending_order_payment": ("payment_modification_result", "payment_modification_result"),
    "modify_user_address": ("user_address_result", "user_address_result"),
    "update_reservation_flights": ("reservation_update_result", "reservation_update_result"),
    "update_reservation_baggages": ("baggage_update_result", "baggage_update_result"),
}


def build_workflow_document(selected_samples: dict[str, Any]) -> dict[str, Any]:
    samples = selected_samples.get("samples") or []
    workflows = [build_workflow(sample) for sample in samples]
    doc = {
        "schema_version": "wfgen.v1",
        "benchmark": {
            "name": "tau-bench-synthetic diverse 10",
            "description": "Ten diverse tau-bench synthetic scenarios converted from ground-truth action sequences.",
            "source_dataset": selected_samples.get("source_dataset", "fuvty/tau-bench-synthetic"),
            "task_count": len(workflows),
            "assumptions": [
                "Each workflow preserves the benchmark ground-truth action sequence as executable tool nodes.",
                "Read-only lookup/detail nodes can execute in parallel when there is no true data dependency.",
            ],
        },
        "workflows": workflows,
        "generation_notes": [
            "Built deterministically from tau-bench ground-truth actions for runtime/tool validation.",
            "This complements cloud-LLM workflow generation by providing a reference executable workflow set.",
        ],
    }
    doc = insert_llm_checker_nodes(doc)
    validate_workflow_semantics(doc)
    return doc


def build_workflow(sample: dict[str, Any]) -> dict[str, Any]:
    actions = sample.get("ground_truth_actions") or []
    template = template_name(sample)
    reason = reason_for_call(sample)
    action_counts = Counter(action.get("name") for action in actions)
    nodes: list[dict[str, Any]] = []
    action_nodes: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for idx, action in enumerate(actions):
        node = build_node(action, idx, action_counts)
        nodes.append(node)
        action_nodes.append((action, node))

    edges = build_edges(action_nodes)
    execution_hints = {
        "parallel_waves": topological_waves(nodes, edges),
        "critical_path": critical_path(nodes, edges),
        "offloading_rationale": "Business tools run as JSON CLI state/API tools; OCR/vision-style tools remain candidates for QNN/NPU.",
    }
    output_artifacts = [artifact for node in nodes for artifact in node.get("outputs", []) if node["entrypoint"] not in READ_ACTIONS]
    if not output_artifacts and nodes:
        output_artifacts = nodes[-1].get("outputs", [])

    return {
        "workflow_id": f"tau_bench_{sample.get('domain')}_{template}_{sample.get('selection_index')}",
        "name": sanitize_identifier(f"{sample.get('domain')}_{template}_{sample.get('selection_index')}"),
        "description": reason or f"{sample.get('domain')} workflow for template {template}.",
        "source_task_ids": [str(sample.get("task_id")), str(sample.get("selection_index"))],
        "user_goal": reason or "",
        "inputs": infer_workflow_inputs(actions),
        "nodes": nodes,
        "edges": edges,
        "outputs": output_artifacts,
        "execution_hints": execution_hints,
        "validation_checks": [
            {
                "check_id": "all_tool_nodes_succeed",
                "description": "Every executable tool node should return success=true.",
                "expected_signal": "all step.success values are true",
            }
        ],
    }


def build_node(action: dict[str, Any], idx: int, action_counts: Counter[str]) -> dict[str, Any]:
    name = action.get("name") or f"action_{idx}"
    args = action.get("arguments") or {}
    node_id = name if action_counts[name] == 1 else f"{name}_{idx}"
    output_name, output_type = output_for_action(name)
    return {
        "node_id": node_id,
        "kind": "app",
        "label": label_for_action(name),
        "purpose": purpose_for_action(name),
        "entrypoint": name,
        "parameters": [
            {"name": key, "value": json.dumps(value) if isinstance(value, (list, dict)) else str(value), "description": f"{name} argument {key}"}
            for key, value in args.items()
        ],
        "inputs": artifacts_from_args(args),
        "outputs": [
            {
                "name": output_name,
                "artifact_type": output_type,
                "codec": "json",
                "description": f"Output artifact from {name}.",
                "max_bytes": 65536,
            }
        ],
        "preferred_backend": "cloud",
        "batch_size": 1,
        "side_effects": side_effects_for_action(name),
    }


def build_edges(action_nodes: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    user_node = next((node for action, node in action_nodes if action.get("name") in USER_ACTIONS), None)
    order_nodes: dict[str, dict[str, Any]] = {}
    product_nodes: dict[str, list[dict[str, Any]]] = {}
    for action, node in action_nodes:
        args = action.get("arguments") or {}
        if action.get("name") == "get_order_details":
            order_nodes[normalize_order_id(args.get("order_id"))] = node
        if action.get("name") == "get_reservation_details":
            order_nodes[normalize_order_id(args.get("reservation_id"))] = node
        if action.get("name") == "get_product_details":
            product_nodes.setdefault(str(args.get("product_id")), []).append(node)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for action, node in action_nodes:
        name = action.get("name")
        if name in READ_ACTIONS:
            continue
        args = action.get("arguments") or {}
        if user_node and user_node["node_id"] != node["node_id"]:
            add_edge(edges, seen, user_node["node_id"], node["node_id"], "user", "user", "Provides user/account context.")
        order_id = normalize_order_id(args.get("order_id") or args.get("reservation_id"))
        if order_id and order_id in order_nodes:
            add_edge(edges, seen, order_nodes[order_id]["node_id"], node["node_id"], "order", "order", "Provides current order/reservation state.")
        for product_id in args.get("new_item_ids") or []:
            for product_node in product_nodes.get(str(product_id), []):
                add_edge(edges, seen, product_node["node_id"], node["node_id"], "product", "product", "Provides replacement product details.")
    return edges


def add_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    src: str,
    dst: str,
    artifact_name: str,
    artifact_type: str,
    reason: str,
) -> None:
    key = (src, dst, artifact_name)
    if key in seen:
        return
    seen.add(key)
    edges.append(
        {
            "from_node": src,
            "to_node": dst,
            "artifact": {
                "name": artifact_name,
                "artifact_type": artifact_type,
                "codec": "json",
                "description": reason,
                "max_bytes": 65536,
            },
            "reason": reason,
        }
    )


def topological_waves(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    node_ids = [node["node_id"] for node in nodes]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        outgoing[edge["from_node"]].append(edge["to_node"])
        indegree[edge["to_node"]] += 1
    ready = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    waves: list[dict[str, Any]] = []
    while ready:
        wave_ids = list(ready)
        ready.clear()
        waves.append({"wave_index": len(waves), "node_ids": wave_ids, "reason": "Derived from data dependencies."})
        for node_id in wave_ids:
            for dst in outgoing[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
    return waves


def critical_path(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return []
    incoming = {node["node_id"]: [] for node in nodes}
    for edge in edges:
        incoming[edge["to_node"]].append(edge["from_node"])
    last = nodes[-1]["node_id"]
    path = [last]
    while incoming.get(path[0]):
        path.insert(0, incoming[path[0]][0])
    return path


def infer_workflow_inputs(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for action in actions:
        for artifact in artifacts_from_args(action.get("arguments") or {}):
            if artifact["name"] not in seen:
                seen.add(artifact["name"])
                inputs.append(artifact)
    return inputs


def artifacts_from_args(args: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = []
    for key in args:
        artifact_type = "id" if key.endswith("_id") or key.endswith("_ids") else "value"
        artifacts.append(
            {
                "name": key,
                "artifact_type": artifact_type,
                "codec": "json" if key.endswith("_ids") or key in {"flights"} else "text/plain",
                "description": f"Input parameter {key}.",
                "max_bytes": 65536,
            }
        )
    return artifacts


def output_for_action(action_name: str) -> tuple[str, str]:
    if action_name in USER_ACTIONS:
        return "user", "user"
    if action_name in {"get_order_details", "get_reservation_details"}:
        return "order", "order"
    if action_name == "get_product_details":
        return "product", "product"
    return MUTATION_OUTPUTS.get(action_name, (f"{action_name}_result", "tool_result"))


def side_effects_for_action(action_name: str) -> list[str]:
    if action_name in READ_ACTIONS:
        return []
    return ["state_update", "audit_log"]


def purpose_for_action(action_name: str) -> str:
    return action_name.replace("_", " ")


def label_for_action(action_name: str) -> str:
    return action_name.replace("_", " ").title()


def template_name(sample: dict[str, Any]) -> str:
    desc = sample.get("description") or {}
    purpose = desc.get("purpose") if isinstance(desc, dict) else ""
    return sanitize_identifier((purpose or "").replace("Generated from template:", "").strip() or "unknown")


def reason_for_call(sample: dict[str, Any]) -> str:
    instructions = ((sample.get("user_scenario") or {}).get("instructions") or {})
    return str(instructions.get("reason_for_call") or "")


def normalize_order_id(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.startswith("W"):
        return f"#{text}"
    return text


def sanitize_identifier(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "workflow"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reference workflows from normalized tau-bench samples.")
    parser.add_argument("samples", type=Path, help="Normalized selected sample JSON.")
    parser.add_argument("-o", "--output", type=Path, default=Path("workflow.diverse10.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = json.loads(args.samples.read_text(encoding="utf-8"))
    doc = build_workflow_document(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wfgen: wrote {len(doc['workflows'])} workflows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
