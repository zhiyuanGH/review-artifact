"""MOBench stage 2: deployment planning and interpretation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from MOBench.deployment_planning import DEPLOYMENT_STAGE_DIR, DEPLOYMENT_STRATEGIES  # noqa: E402
from MOBench.deployment_planning.core import (  # noqa: E402
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_URL,
    DEFAULT_GENIE_BUNDLE_DIR,
    DEFAULT_ORT_GENAI_DIR,
    DEFAULT_ORT_GENAI_JAR,
    DEFAULT_ORT_GENAI_MODEL,
    DEFAULT_PHONE_LLAMA_DIR,
    DEFAULT_PHONE_MODEL,
    build_deployment_document,
    build_runtime_config,
    build_runtime_defaults,
    validate_deployment_document,
    workflow_deployment_for_index,
)
from MOBench.deployment_planning.visualization import write_deployment_visualizations  # noqa: E402
from MOBench.deployment_planning.interpreter import compile_workflow_document, sanitize_identifier  # noqa: E402


ENTRYPOINT_NAME = "deployment-planning-interpret"
ENTRYPOINT_STAGES = ["deployment_planning", "interpretation", "visualization"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    planning_dir = run_dir / "planning"
    deployment_dir = run_dir / DEPLOYMENT_STAGE_DIR
    deployment_dir.mkdir(parents=True, exist_ok=True)
    (deployment_dir / "runtime_configs").mkdir(exist_ok=True)
    (deployment_dir / "compiled").mkdir(exist_ok=True)

    planning_workflow_path = planning_dir / "workflow.json"
    if not planning_workflow_path.exists():
        raise FileNotFoundError(f"Planning workflow not found. Run `python wfgen_plan.py --run-dir {run_dir}` first: {planning_workflow_path}")
    workflow_doc = json.loads(planning_workflow_path.read_text(encoding="utf-8"))
    selected_samples_path = planning_dir / "selected_samples.json"
    selected_samples_path = selected_samples_path if selected_samples_path.exists() else None
    runtime_defaults = build_runtime_defaults(args, repo_root=REPO_ROOT)

    print(f"[deploy] run_dir={run_dir}", flush=True)
    print(f"[deploy] strategy={args.deployment_strategy}", flush=True)
    deployment_doc = build_deployment_document(
        workflow_doc,
        strategy=args.deployment_strategy,
        run_dir=run_dir,
        planning_workflow_path=planning_workflow_path,
        selected_samples_path=selected_samples_path,
        runtime_defaults=runtime_defaults,
    )
    validate_deployment_document(deployment_doc)

    deployment_plan_path = deployment_dir / "deployment_plan.json"
    workflow_deployment_path = deployment_dir / "workflow_deployment.json"
    write_json(deployment_plan_path, deployment_doc["deployment_plan"])
    write_json(workflow_deployment_path, deployment_doc)
    print(f"[deploy] wrote deployment plan: {deployment_plan_path}", flush=True)
    print(f"[deploy] wrote workflow deployment: {workflow_deployment_path}", flush=True)

    runtime_config_paths = write_runtime_configs(
        deployment_doc,
        deployment_dir / "runtime_configs",
        selected_samples_path=selected_samples_path,
        run_dir=run_dir,
    )
    compiled_paths = compile_workflows(planning_workflow_path, workflow_doc, deployment_dir / "compiled")
    visualization_paths = []
    if not args.no_visualize:
        visualization_paths = write_deployment_visualizations(deployment_doc, deployment_dir / "visualizations")

    summary_path = write_summary(
        deployment_dir,
        deployment_doc,
        runtime_config_paths,
        compiled_paths,
        visualization_paths,
    )
    manifest_path = write_manifest(
        deployment_dir,
        run_dir,
        args,
        planning_workflow_path,
        workflow_deployment_path,
        deployment_plan_path,
        runtime_config_paths,
        compiled_paths,
        visualization_paths,
        summary_path,
    )
    print(f"wfgen-deploy: wrote deployment artifacts to {deployment_dir}")
    print(f"wfgen-deploy: wrote manifest to {manifest_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOBench stage 2: assign node backends/devices, compile workflow runners, and visualize deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run root containing planning/workflow.json.")
    parser.add_argument("--deployment-strategy", choices=DEPLOYMENT_STRATEGIES, default="cli-balanced-wave-aware")
    parser.add_argument("--python", dest="python_executable", type=Path, default=None, help="Python executable for JSON CLI tool subprocesses.")
    parser.add_argument("--max-parallel-nodes", type=int, default=4)
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-max-tokens", type=int, default=64)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--llm-timeout", type=int, default=240)
    parser.add_argument("--llama-health-url", default=None)
    parser.add_argument("--adb", type=Path, default=None)
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--llama-local-port", type=int, default=None)
    parser.add_argument("--llama-remote-port", type=int, default=8080)
    parser.add_argument("--llama-remote-dir", default=DEFAULT_PHONE_LLAMA_DIR)
    parser.add_argument("--llama-model-path", default=DEFAULT_PHONE_MODEL)
    parser.add_argument("--llama-server-device", default="HTP0")
    parser.add_argument("--llama-parallel", type=int, default=2)
    parser.add_argument("--llama-startup-timeout", type=int, default=240)
    parser.add_argument("--llama-server-extra-arg", action="append", default=[])
    parser.add_argument("--llama-cli-timeout", type=int, default=240)
    parser.add_argument("--llama-cli-extra-arg", action="append", default=[])
    parser.add_argument("--llama-cli-warmup-runs", type=int, default=1, help="Warmup llama-cli calls per unique model/backend before measured execution; use 0 for cold-start timing.")
    parser.add_argument("--llama-cli-warmup-max-tokens", type=int, default=1, help="Generated tokens for llama-cli warmup calls.")
    parser.add_argument("--llama-cli-warmup-prompt", default="OK", help="Prompt text used for llama-cli warmup calls.")
    parser.add_argument("--ort-genai-dir", default=DEFAULT_ORT_GENAI_DIR)
    parser.add_argument("--ort-genai-model-path", default=DEFAULT_ORT_GENAI_MODEL)
    parser.add_argument("--ort-genai-jar", default=DEFAULT_ORT_GENAI_JAR)
    parser.add_argument("--ort-genai-timeout", type=int, default=240)
    parser.add_argument("--ort-genai-prompt-char-limit", type=int, default=512, help="Character budget for ORT GenAI prompts; 0 disables truncation.")
    parser.add_argument("--genie-bundle-dir", default=DEFAULT_GENIE_BUNDLE_DIR)
    parser.add_argument("--genie-config", default="auto", help="Genie config filename, absolute phone path, or auto for SoC-specific defaults.")
    parser.add_argument("--genie-backend", default="npu", choices=["cpu", "gpu", "npu"])
    parser.add_argument("--genie-log-level", default="warn", choices=["error", "warn", "info", "verbose"])
    parser.add_argument("--genie-timeout", type=int, default=240)
    parser.add_argument("--no-visualize", action="store_true", help="Skip deployment HTML visualizations.")
    return parser.parse_args(argv)


def resolve_run_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = []
    results_dir = REPO_ROOT / "results"
    if results_dir.exists():
        for path in results_dir.rglob("planning/workflow.json"):
            candidates.append(path.parent.parent)
    if not candidates:
        raise FileNotFoundError("No --run-dir provided and no results/**/planning/workflow.json found.")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1].resolve()


def write_runtime_configs(
    deployment_doc: dict[str, Any],
    output_dir: Path,
    *,
    selected_samples_path: Path | None,
    run_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    workflow_doc = deployment_doc["workflow_document"]
    deployment_plan = deployment_doc["deployment_plan"]
    runtime_defaults = deployment_plan["runtime_defaults"]
    for idx, workflow in enumerate(workflow_doc.get("workflows") or []):
        name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{idx}")
        workflow_deployment = workflow_deployment_for_index(deployment_plan, idx)
        state_path = run_dir / "execution" / "tool_state" / f"{idx:02d}_{name}.json"
        config = build_runtime_config(
            workflow,
            workflow_deployment,
            runtime_defaults=runtime_defaults,
            dataset_path=selected_samples_path,
            state_path=state_path,
        )
        path = output_dir / f"{idx:02d}_{name}.runtime.json"
        write_json(path, config)
        paths.append(path)
    return paths


def compile_workflows(workflow_path: Path, workflow_doc: dict[str, Any], output_dir: Path) -> list[Path]:
    paths = []
    for idx, workflow in enumerate(workflow_doc.get("workflows", [])):
        name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{idx}")
        output_path = output_dir / f"{idx:02d}_{name}.py"
        compile_workflow_document(
            workflow_path,
            output_path=output_path,
            workflow_index=idx,
            project_root=PACKAGE_PARENT,
        )
        paths.append(output_path)
    return paths


def write_summary(
    output_dir: Path,
    deployment_doc: dict[str, Any],
    runtime_config_paths: list[Path],
    compiled_paths: list[Path],
    visualization_paths: list[Path],
) -> Path:
    plan = deployment_doc["deployment_plan"]
    rows = []
    for workflow in plan.get("workflows") or []:
        runtimes = sorted({item.get("selected_runtime") for item in workflow.get("node_deployments") or []})
        devices = sorted({item.get("device_class") for item in workflow.get("node_deployments") or []})
        rows.append(
            "| "
            + " | ".join(
                [
                    str(workflow.get("workflow_index")),
                    str(workflow.get("workflow_id")),
                    str(len(workflow.get("node_deployments") or [])),
                    ", ".join(str(item) for item in runtimes),
                    ", ".join(str(item) for item in devices),
                ]
            )
            + " |"
        )
    lines = [
        "# MOBench Deployment Results",
        "",
        f"Entrypoint: `{ENTRYPOINT_NAME}`",
        f"Stages: `{' -> '.join(ENTRYPOINT_STAGES)}`",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`",
        f"Strategy: `{plan.get('strategy')}`",
        f"Workflow count: `{len(plan.get('workflows') or [])}`",
        f"Runtime configs: `{len(runtime_config_paths)}`",
        f"Compiled scripts: `{len(compiled_paths)}`",
        f"Visualizations: `{len(visualization_paths)}`",
        "",
        "| Index | Workflow | Nodes | Runtimes | Device classes |",
        "| ---: | --- | ---: | --- | --- |",
        *rows,
        "",
    ]
    path = output_dir / "DEPLOYMENT_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manifest(
    output_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    planning_workflow_path: Path,
    workflow_deployment_path: Path,
    deployment_plan_path: Path,
    runtime_config_paths: list[Path],
    compiled_paths: list[Path],
    visualization_paths: list[Path],
    summary_path: Path,
) -> Path:
    manifest = {
        "entrypoint": ENTRYPOINT_NAME,
        "pipeline_stages": ENTRYPOINT_STAGES,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "stage_dir": str(output_dir),
        "deployment_strategy": args.deployment_strategy,
        "artifacts": {
            "planning_workflow_json": str(planning_workflow_path),
            "workflow_deployment_json": str(workflow_deployment_path),
            "deployment_plan_json": str(deployment_plan_path),
            "runtime_configs": [str(path) for path in runtime_config_paths],
            "compiled_scripts": [str(path) for path in compiled_paths],
            "visualizations": [str(path) for path in visualization_paths],
            "summary": str(summary_path),
        },
    }
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
