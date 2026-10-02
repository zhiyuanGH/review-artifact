"""Generate standalone LLM codegen prompts from a MOBench workflow document."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROMPT_HEADER = """You are an expert Python engineer building ODABench executable workflow benchmarks.

Task:
Generate executable Python 3 code for exactly one on-device agent workflow. The purpose of this request is to measure the cost and latency of LLM-assisted executable-code generation from an already planned ODABench workflow.

Important constraints:
- Generate code for only the workflow in this prompt. Do not add other workflows or examples.
- Preserve every workflow node, edge, parameter, output artifact, side effect annotation, validation check, preferred backend, and batch size.
- Preserve the data dependencies from the workflow JSON. Nodes with no unsatisfied dependencies should be allowed to run concurrently.
- Return only one complete Python file in a single fenced ```python code block. Do not include prose outside the code block.

Implementation requirements:
1. Implement a self-contained Python 3 program using only the standard library.
2. Use asyncio to execute the DAG. A node may run after all predecessor nodes in the workflow edges have completed.
3. Define clear structures for nodes, edges, artifacts, node results, and timing records. It is acceptable to use dataclasses.
4. For app/tool nodes, create async functions matching each node entrypoint. If a real service is not available, implement deterministic local stub behavior that accepts the node parameters and returns JSON-like outputs matching the declared output artifact names and descriptions.
5. For LLM nodes using entrypoint `llama_server_chat`, implement an async wrapper that builds a prompt from the node instruction and upstream artifacts. It should optionally call an HTTP endpoint from environment variable `LLM_ENDPOINT` if present, and otherwise use a deterministic local fallback response. Use only the Python standard library for the optional HTTP call.
6. Implement artifact passing through an in-memory artifact store. Store each node's declared outputs by artifact name.
7. Include timing instrumentation for each node: node_id, label, kind, preferred_backend, batch_size, start_time, end_time, duration_seconds, status, and error if any.
8. Include total workflow latency and print a final JSON report with workflow_id, workflow_name, user_goal, final_outputs, validation_results, node_timings, and total_duration_seconds.
9. Implement the validation checks from the workflow JSON as executable best-effort checks. If exact fields are unavailable because a stub returned synthetic data, report the validation as `stubbed` with an explanation rather than silently skipping it.
10. Include a `main()` function and `if __name__ == "__main__":` entrypoint that runs the workflow once for the provided user prompt.
"""


def main() -> int:
    args = parse_args()
    workflow_doc = json.loads(args.workflow_json.read_text(encoding="utf-8"))
    selected_doc = json.loads(args.selected_samples.read_text(encoding="utf-8")) if args.selected_samples else {}
    samples = selected_doc.get("samples") or []
    sample_by_task = {sample.get("task_id"): sample for sample in samples}
    sample_by_index = {sample.get("selection_index"): sample for sample in samples}

    workflows = workflow_doc.get("workflows") or []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, workflow in enumerate(workflows):
        sample = find_sample(workflow, index, sample_by_task, sample_by_index)
        prompt = render_prompt(workflow, sample)
        filename = f"{index:03d}_{safe_stem(workflow.get('name') or workflow.get('workflow_id') or 'workflow')}.txt"
        path = args.output_dir / filename
        path.write_text(prompt, encoding="utf-8")
        action_names = [action.get("name", "") for action in (sample or {}).get("ground_truth_actions") or []]
        manifest_rows.append(
            {
                "file": filename,
                "workflow_id": workflow.get("workflow_id", ""),
                "name": workflow.get("name", ""),
                "task_ids": workflow.get("source_task_ids") or [],
                "selection_index": (sample or {}).get("selection_index", ""),
                "task_id": (sample or {}).get("task_id", ""),
                "domain": (sample or {}).get("domain", ""),
                "action_sequence": " -> ".join(action_names),
                "prompt_chars": len(prompt),
            }
        )

    readme = render_readme(manifest_rows)
    (args.output_dir / "README.txt").write_text(readme, encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest_rows)} prompts to {args.output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-json", type=Path, required=True, help="MOBench planning workflow.json.")
    parser.add_argument("--selected-samples", type=Path, default=None, help="Selected sample metadata JSON.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated .txt prompts.")
    return parser.parse_args()


def find_sample(
    workflow: dict[str, Any],
    index: int,
    sample_by_task: dict[Any, dict[str, Any]],
    sample_by_index: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    if index in sample_by_index:
        return sample_by_index[index]
    for task_id in workflow.get("source_task_ids") or []:
        if task_id in sample_by_task:
            return sample_by_task[task_id]
    return sample_by_index.get(index, {})


def render_prompt(workflow: dict[str, Any], sample: dict[str, Any]) -> str:
    instructions = ((sample.get("user_scenario") or {}).get("instructions") or {}) if sample else {}
    reason = instructions.get("reason_for_call") or workflow.get("user_goal") or ""
    action_names = [action.get("name", "") for action in sample.get("ground_truth_actions") or []] if sample else []
    metadata_lines = [
        f"- domain: {sample.get('domain', '')}" if sample else "- domain: ",
        f"- task_id: {sample.get('task_id', '')}" if sample else "- task_id: ",
        f"- selection_index: {sample.get('selection_index', '')}" if sample else "- selection_index: ",
    ]
    for key in ("task_instructions", "known_info", "unknown_info"):
        value = instructions.get(key)
        if value:
            metadata_lines.append(f"- {key}: {value}")

    workflow_json = json.dumps(workflow, indent=2, ensure_ascii=False)
    return (
        PROMPT_HEADER
        + "\n\nUser prompt for this workflow:\n"
        + (reason or "")
        + "\n\nAdditional user scenario metadata:\n"
        + "\n".join(metadata_lines)
        + "\n\nWorkflow action sequence:\n"
        + (" -> ".join(action_names) if action_names else infer_action_sequence(workflow))
        + "\n\nWorkflow JSON:\n```json\n"
        + workflow_json
        + "\n```\n"
    )


def infer_action_sequence(workflow: dict[str, Any]) -> str:
    names = []
    for node in workflow.get("nodes") or []:
        if node.get("kind") == "app":
            names.append(node.get("entrypoint") or node.get("label") or node.get("node_id", ""))
    return " -> ".join(name for name in names if name)


def render_readme(rows: list[dict[str, Any]]) -> str:
    lines = [
        "ODABench LLM code-generation prompts",
        "",
        "Each .txt file in this directory is a standalone prompt for exactly one workflow.",
        "Send one file at a time to the real-world LLM when measuring code-generation latency.",
        "",
        "Files:",
    ]
    for row in rows:
        lines.append(
            f"- {row['file']}: {row['workflow_id']} | {row['name']} | {row['action_sequence']}"
        )
    lines.append("")
    return "\n".join(lines)


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "workflow"


if __name__ == "__main__":
    raise SystemExit(main())
