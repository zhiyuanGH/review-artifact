"""Check whether a planned workflow can exercise CPU/GPU/NPU wave placement."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from MOBench.deployment_planning.core import (
        canonical_strategy,
        deploy_workflow_nodes,
        execution_waves_for_deployment,
    )
except ImportError:
    from deployment_planning.core import (
        canonical_strategy,
        deploy_workflow_nodes,
        execution_waves_for_deployment,
    )


REQUIRED_DEVICE_CLASSES = {"cpu", "gpu", "npu"}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workflow_path = resolve_workflow_path(args.run_dir, args.workflow_json)
    workflow_doc = json.loads(workflow_path.read_text(encoding="utf-8"))
    workflows = workflow_doc.get("workflows") or []
    indices = parse_indices(args.workflow_index, len(workflows))
    runtime_defaults = minimal_runtime_defaults(args)

    reports = []
    ok = True
    for index in indices:
        workflow = workflows[index]
        report = inspect_workflow(
            workflow,
            workflow_index=index,
            strategy=args.deployment_strategy,
            runtime_defaults=runtime_defaults,
        )
        reports.append(report)
        ok = ok and bool(report["has_required_wave"])
        print_workflow_report(report)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "workflow_json": str(workflow_path),
        "deployment_strategy": args.deployment_strategy,
        "canonical_strategy": canonical_strategy(args.deployment_strategy),
        "required_device_classes": sorted(REQUIRED_DEVICE_CLASSES),
        "workflow_indices": indices,
        "ok": ok,
        "workflows": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[coverage] wrote {args.output}")
    return 0 if ok else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a workflow has a wave whose LLM nodes cover CPU, GPU, and NPU placements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results" / "my_run")
    parser.add_argument("--workflow-json", type=Path, default=None)
    parser.add_argument("--workflow-index", default="all", help="Workflow index, comma list, or all.")
    parser.add_argument("--deployment-strategy", default="cli-balanced-wave-aware")
    parser.add_argument("--llama-model-path", default="/data/local/tmp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf")
    parser.add_argument("--max-parallel-nodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def resolve_workflow_path(run_dir: Path, workflow_json: Path | None) -> Path:
    path = workflow_json or run_dir / "planning" / "workflow.json"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Workflow JSON not found: {path}")
    return path


def parse_indices(raw: str, count: int) -> list[int]:
    if raw.lower() == "all":
        return list(range(count))
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    for index in indices:
        if index < 0 or index >= count:
            raise IndexError(f"Workflow index {index} is outside 0..{count - 1}")
    return indices


def minimal_runtime_defaults(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_parallel_nodes": args.max_parallel_nodes,
        "python_executable": sys.executable,
        "adb": "",
        "adb_serial": "",
        "llm_url": "http://127.0.0.1:8080/v1/chat/completions",
        "llm_model": "llama-3.2-1b-q4_0",
        "llm_max_tokens": 64,
        "llm_temperature": 0.0,
        "llm_timeout": 240,
        "llama_health_url": None,
        "llama_local_port": None,
        "llama_remote_port": 8080,
        "llama_remote_dir": "/data/local/tmp/llama.cpp",
        "llama_model_path": args.llama_model_path,
        "llama_server_device": "HTP0",
        "llama_parallel": 2,
        "llama_startup_timeout": 240,
        "llama_server_extra_args": [],
        "llama_cli_timeout": 240,
        "llama_cli_extra_args": [],
        "llama_cli_warmup_runs": 1,
        "llama_cli_warmup_max_tokens": 1,
        "llama_cli_warmup_prompt": "OK",
    }


def inspect_workflow(
    workflow: dict[str, Any],
    *,
    workflow_index: int,
    strategy: str,
    runtime_defaults: dict[str, Any],
) -> dict[str, Any]:
    deployments = deploy_workflow_nodes(workflow, strategy=strategy, runtime_defaults=runtime_defaults)
    deployment_by_node = {item["node_id"]: item for item in deployments}
    node_by_id = {node.get("node_id"): node for node in workflow.get("nodes") or []}
    wave_reports = []
    has_required_wave = False
    for wave in execution_waves_for_deployment(workflow):
        llm_nodes = [
            node_id
            for node_id in wave.get("node_ids") or []
            if node_by_id.get(node_id, {}).get("kind") == "llm"
        ]
        device_classes = sorted(
            {
                str(deployment_by_node[node_id].get("device_class"))
                for node_id in llm_nodes
                if node_id in deployment_by_node
            }
        )
        covers_required = REQUIRED_DEVICE_CLASSES.issubset(set(device_classes))
        has_required_wave = has_required_wave or covers_required
        wave_reports.append(
            {
                "wave_index": int(wave.get("wave_index", len(wave_reports))),
                "node_count": len(wave.get("node_ids") or []),
                "llm_node_count": len(llm_nodes),
                "device_classes": device_classes,
                "covers_required_device_classes": covers_required,
                "llm_deployments": [
                    {
                        "node_id": node_id,
                        "device_class": deployment_by_node.get(node_id, {}).get("device_class"),
                        "device": deployment_by_node.get(node_id, {}).get("device"),
                        "runtime": deployment_by_node.get(node_id, {}).get("selected_runtime"),
                    }
                    for node_id in llm_nodes
                ],
            }
        )
    return {
        "workflow_index": workflow_index,
        "workflow_id": workflow.get("workflow_id"),
        "name": workflow.get("name"),
        "has_required_wave": has_required_wave,
        "waves": wave_reports,
    }


def print_workflow_report(report: dict[str, Any]) -> None:
    status = "OK" if report["has_required_wave"] else "MISSING"
    print(
        f"[coverage] workflow {report['workflow_index']} {status}: "
        f"{report.get('workflow_id')} | {report.get('name')}"
    )
    for wave in report["waves"]:
        if not wave["llm_node_count"]:
            continue
        devices = ",".join(wave["device_classes"]) or "-"
        mark = "*" if wave["covers_required_device_classes"] else " "
        print(
            f"  {mark} wave {wave['wave_index']}: "
            f"llm={wave['llm_node_count']} devices={devices} "
            f"nodes={[(item['node_id'], item['device']) for item in wave['llm_deployments']]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
