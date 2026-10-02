"""MOBench entrance 2: execute-visualize.

This script is the user-facing entrypoint for executing selected workflows,
collecting phone/runtime results, and writing result-aware visualizations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .deployment_planning.core import deployment_requires_llama_server, workflow_deployment_for_index
    from .workflow_execution.runtime import MobileWorkflowRuntime
    from .resource_monitoring import PerfettoResourceMonitor, ResourceMonitorConfig
    from .tools.llm_business import LLM_BUSINESS_TOOLS
    from .tools.llm_tau import LLM_TAU_TOOLS
    from .visualizations.visualize_execution_performance import load_dashboard_payload, render_dashboard
    from .visualizations.visualize_workflow import render_html
    from .workflow_execution import EXECUTION_STAGE_DIR
    from .workflow_execution.deployed import deployment_plan as deployment_plan_from_doc
    from .workflow_execution.deployed import load_workflow_deployment, workflow_document
except ImportError:
    from deployment_planning.core import deployment_requires_llama_server, workflow_deployment_for_index
    from workflow_execution.runtime import MobileWorkflowRuntime
    from resource_monitoring import PerfettoResourceMonitor, ResourceMonitorConfig
    from tools.llm_business import LLM_BUSINESS_TOOLS
    from tools.llm_tau import LLM_TAU_TOOLS
    from visualizations.visualize_execution_performance import load_dashboard_payload, render_dashboard
    from visualizations.visualize_workflow import render_html
    from workflow_execution import EXECUTION_STAGE_DIR
    from workflow_execution.deployed import deployment_plan as deployment_plan_from_doc
    from workflow_execution.deployed import load_workflow_deployment, workflow_document


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = REPO_ROOT.parent
DEFAULT_LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_LLM_MODEL = "llama-3.2-1b-q4_0"
DEFAULT_PHONE_MODEL = "/data/local/tmp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf"
DEFAULT_PHONE_LLAMA_DIR = "/data/local/tmp/llama.cpp"
ENTRYPOINT_NAME = "real-execution"
ENTRYPOINT_STAGES = ["execution", "visualization"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reject_old_execution_api(args)
    deployment_path, run_dir = resolve_deployment_path(args)
    deployment_doc = load_workflow_deployment(deployment_path)
    workflow_doc = workflow_document(deployment_doc)
    deployment_plan = deployment_plan_from_doc(deployment_doc)
    workflows = workflow_doc.get("workflows") or []
    if not workflows:
        raise ValueError(f"No workflows found in deployed workflow: {deployment_path}")

    selected_indices = select_workflow_indices(workflows, args)
    if args.list:
        print_workflow_list(workflows, selected_indices)
        return 0

    output_dir = run_dir / EXECUTION_STAGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "tool_state").mkdir(exist_ok=True)

    dataset_path = resolve_dataset_path(args, deployment_path, deployment_doc)
    apply_deployment_execution_defaults(args, deployment_doc, selected_indices)
    llama_server_info = maybe_ensure_llama_server(args, output_dir)
    warmup_info = warmup_llama_cli_runtime_configs(args, run_dir, workflows, selected_indices, output_dir)
    run_records = []
    status_code = 0
    error_log_path = output_dir / "ERROR_LOG.jsonl"
    resource_monitor, resource_info = maybe_start_resource_monitor(args, output_dir)

    try:
        for sequence, workflow_index in enumerate(selected_indices):
            workflow = workflows[workflow_index]
            name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{workflow_index}")
            state_path = output_dir / "tool_state" / f"{workflow_index:02d}_{name}.json"
            workflow_deployment = workflow_deployment_for_index(deployment_plan, workflow_index)
            config_path = runtime_config_path(run_dir, workflow_index, workflow)
            config = load_deployed_runtime_config(config_path)
            hydrate_runtime_config_for_execution(config, args, state_path)

            print(f"[run] {workflow_index:02d} {workflow.get('workflow_id')} -> {name}")
            workflow_started_epoch = time.time()
            runtime = MobileWorkflowRuntime(config, dry_run=args.dry_run)
            try:
                result = runtime.execute_workflow(workflow)
            except Exception as exc:
                result = error_result(workflow, exc, workflow_started_epoch, args)
            workflow_finished_epoch = time.time()
            result["workflow_document_metadata"] = {
                "schema_version": workflow_doc.get("schema_version"),
                "benchmark": workflow_doc.get("benchmark"),
                "generation_metadata": workflow_doc.get("generation_metadata", {}),
            }
            result["execution_control"] = {
                "workflow_index": workflow_index,
                "sequence_index": sequence,
                "runtime_config": str(config_path),
                "dataset": str(dataset_path) if dataset_path else None,
                "state": str(state_path),
                "dry_run": args.dry_run,
                "tool_mode": args.tool_mode,
                "deployment_json": str(deployment_path),
                "deployment_strategy": deployment_plan.get("strategy"),
                "node_deployments": workflow_deployment.get("node_deployments", []),
                "llama_server": llama_server_info,
                "llama_cli_warmup": compact_warmup_info(warmup_info),
                "resource_monitor": compact_resource_monitor_info(resource_info),
                "started_epoch_s": workflow_started_epoch,
                "finished_epoch_s": workflow_finished_epoch,
            }

            result_path = output_dir / f"{workflow_index:02d}_{name}.result.json"
            result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

            visualization_path = None
            if not args.no_visualize:
                visualization_path = output_dir / "visualizations" / f"{workflow_index:02d}_{name}.html"
                visualization_path.write_text(render_html(workflow, workflow_doc, result), encoding="utf-8")

            record = {
                "workflow_index": workflow_index,
                "workflow_id": workflow.get("workflow_id"),
                "name": workflow.get("name"),
                "status": result.get("status"),
                "dry_run": result.get("dry_run"),
                "total_wall_ms": result.get("total_wall_ms"),
                "started_epoch_s": workflow_started_epoch,
                "finished_epoch_s": workflow_finished_epoch,
                "result_json": str(result_path),
                "runtime_config": str(config_path),
                "visualization": str(visualization_path) if visualization_path else None,
            }
            run_records.append(record)
            print(
                f"[done] {workflow_index:02d} status={record['status']} "
                f"wall_ms={record['total_wall_ms']:.2f} result={result_path}"
            )
            if result.get("status") != "completed":
                write_error_log(error_log_path, record, result)
                if args.fail_on_error:
                    status_code = 1
                if args.stop_on_failure:
                    break
    finally:
        if resource_monitor is not None:
            resource_info = finish_resource_monitor(resource_monitor, run_records, args.perfetto_required)

    apply_resource_results(output_dir, workflows, workflow_doc, run_records, resource_info, args.no_visualize)
    write_manifest(output_dir, args, deployment_path, dataset_path, run_records, llama_server_info, warmup_info, resource_info)
    write_summary(output_dir, run_records, resource_info)
    write_performance_overview(output_dir)
    print(f"[summary] wrote execution results to {output_dir}")
    return status_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MOBench stage 3: real execution. Execute deployed workflows from "
            "<run-dir>/deployment/workflow_deployment.json, collect runtime artifacts, "
            "and render result-aware HTML under <run-dir>/execution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("workflow_json", nargs="?", type=Path, default=None, help="Deployment JSON document. Plain planning workflow.json is rejected.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run root containing deployment/workflow_deployment.json.")
    parser.add_argument("--plan-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dataset", type=Path, default=None, help="Normalized tau-bench sample JSON for json_cli tools.")
    parser.add_argument("--config", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workflow-index", action="append", default=[], help="Workflow index, comma list, or range, for example 0,2,5-7.")
    parser.add_argument("--workflow-id", action="append", default=[], help="Workflow id to execute. Can be repeated.")
    parser.add_argument("--list", action="store_true", help="List matching workflows without executing them.")
    parser.add_argument("--dry-run", action="store_true", help="Mock all node outputs without calling tools or llama-server.")
    parser.add_argument("--tool-mode", choices=["json_cli", "mock"], default="json_cli", help="How app/dnn tool nodes should run when not in dry-run.")
    parser.add_argument("--read-tool-runtime", choices=["script", "llama_cli", "llama_server", "mock"], default="script", help="Implementation backend for read-only business tool nodes; logical nodes remain tools.")
    parser.add_argument("--llm-tool-runtime", choices=["llama_cli", "llama_server", "mock"], default="llama_cli", help="Runtime for semantic LLM tools such as extract_tau_slots.")
    parser.add_argument("--llm-node-runtime", choices=["llama_server", "llama_cli", "ort_genai_cli", "genie_cli"], default="llama_server", help="Runtime for workflow nodes whose kind is llm.")
    parser.add_argument("--python", dest="python_executable", type=Path, default=None, help="Python executable for json_cli tool subprocesses.")
    parser.add_argument("--max-parallel-nodes", type=int, default=4, help="Maximum number of workflow nodes to run in parallel per wave.")
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL, help="OpenAI-compatible llama-server chat completions URL.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Model name recorded for llama-server calls.")
    parser.add_argument("--llm-max-tokens", type=int, default=64, help="Maximum generated tokens per LLM node/tool call.")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--llm-timeout", type=int, default=240, help="Timeout in seconds for llama-server LLM calls.")
    parser.add_argument("--no-ensure-llama-server", action="store_true", help="Do not auto-start phone llama-server when the LLM endpoint is down.")
    parser.add_argument("--llama-health-url", default=None, help="Health endpoint used before launching llama-server. If omitted, /health on --llm-url host is used.")
    parser.add_argument("--adb", type=Path, default=None, help="adb executable. If omitted, ADB env var, bundled platform-tools, or PATH is used.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb device serial.")
    parser.add_argument("--llama-local-port", type=int, default=None, help="Local adb-forward port. If omitted, the port in --llm-url is used.")
    parser.add_argument("--llama-remote-port", type=int, default=8080, help="Phone-side llama-server port.")
    parser.add_argument("--llama-remote-dir", default=DEFAULT_PHONE_LLAMA_DIR, help="Phone directory containing llama.cpp bin/ and lib/.")
    parser.add_argument("--llama-model-path", default=DEFAULT_PHONE_MODEL, help="Phone-side GGUF model path for llama-server and llama-cli tools.")
    parser.add_argument("--llama-device", default="HTP0", help="llama-server accelerator device.")
    parser.add_argument("--llama-parallel", type=int, default=2, help="llama-server parallel sequence count.")
    parser.add_argument("--llama-startup-timeout", type=int, default=240, help="Seconds to wait for phone llama-server startup.")
    parser.add_argument("--llama-server-extra-arg", action="append", default=[], help="Extra llama-server argument. Can be repeated.")
    parser.add_argument("--llama-cli-device", default="none", help="Device for semantic llama-cli tools. Defaults to CPU so it does not contend with llama-server HTP.")
    parser.add_argument("--llama-cli-ngl", type=int, default=None, help="GPU/NPU layers for semantic llama-cli tools. If omitted, 0 is used for device none, otherwise 99.")
    parser.add_argument("--llama-cli-timeout", type=int, default=240, help="Timeout in seconds for phone llama-cli tool calls.")
    parser.add_argument("--llama-cli-extra-arg", action="append", default=[], help="Extra llama-cli argument for semantic LLM tools. Can be repeated.")
    parser.add_argument("--llm-node-cli-device", default=None, help="Device for kind=llm nodes when --llm-node-runtime llama_cli. Defaults to --llama-cli-device.")
    parser.add_argument("--llm-node-cli-ngl", type=int, default=None, help="GPU/NPU layers for kind=llm llama-cli nodes. Defaults like --llama-cli-ngl.")
    parser.add_argument("--llm-node-cli-timeout", type=int, default=None, help="Timeout seconds for kind=llm llama-cli nodes. Defaults to --llama-cli-timeout.")
    parser.add_argument("--llm-node-cli-extra-arg", action="append", default=[], help="Extra llama-cli arg for kind=llm nodes. Can be repeated.")
    parser.add_argument("--llm-node-cli-device-map", action="append", default=[], help="Per-node llama-cli device override, e.g. check_after_n0=GPUOpenCL or check_after_n1=HTP0.")
    parser.add_argument("--resource-monitor", choices=["perfetto", "off"], default="perfetto", help="Phone resource monitor used around the selected workflow batch.")
    parser.add_argument("--resource-sample-interval", type=float, default=1.0, help="Seconds between lightweight ADB resource samples.")
    parser.add_argument("--perfetto-buffer-mb", type=int, default=128, help="Perfetto ring buffer size in MB.")
    parser.add_argument("--perfetto-tools-dir", type=Path, default=REPO_ROOT / ".wfgen_perfetto", help="Local directory for PC-side Perfetto trace_processor tooling.")
    parser.add_argument("--perfetto-required", action="store_true", help="Fail execution if Perfetto monitoring cannot start or stop cleanly.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip result-aware HTML visualizations.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop the batch after the first failed workflow.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero if any workflow fails. By default failures are logged and the batch continues.")
    return parser.parse_args(argv)


def reject_old_execution_api(args: argparse.Namespace) -> None:
    if args.plan_dir:
        raise ValueError(
            "`wfgen_execute.py --plan-dir ...` is no longer supported. "
            "Run `python wfgen_deploy.py --run-dir <run-dir>` first, then "
            "`python wfgen_execute.py --run-dir <run-dir>`."
        )
    if args.output_dir:
        raise ValueError("Execution output is fixed at <run-dir>/execution. Remove --output-dir.")
    if args.config:
        raise ValueError("Runtime configs are generated by wfgen_deploy.py. Remove --config.")


def resolve_deployment_path(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir:
        run_dir = args.run_dir.resolve()
        path = run_dir / "deployment" / "workflow_deployment.json"
        if not path.exists():
            raise FileNotFoundError(f"Deployment artifact not found. Run `python wfgen_deploy.py --run-dir {run_dir}` first: {path}")
        return path, run_dir
    if args.workflow_json:
        path = args.workflow_json.resolve()
        if not path.exists():
            raise FileNotFoundError(f"deployment_json not found: {path}")
        run_dir = infer_run_dir_from_deployment_path(path)
        return path, run_dir
    candidates = latest_deployment_candidates()
    if not candidates:
        raise FileNotFoundError("No deployment JSON provided and no results/**/deployment/workflow_deployment.json found.")
    path = candidates[-1].resolve()
    print(f"[deployment] using latest workflow deployment: {path}")
    return path, infer_run_dir_from_deployment_path(path)


def infer_run_dir_from_deployment_path(path: Path) -> Path:
    if path.parent.name == "deployment":
        return path.parent.parent.resolve()
    return path.parent.resolve()


def latest_deployment_candidates() -> list[Path]:
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        return []
    return sorted(
        [path for path in results_dir.rglob("deployment/workflow_deployment.json") if path.is_file()],
        key=lambda path: (path.stat().st_mtime, str(path)),
    )


def apply_deployment_execution_defaults(
    args: argparse.Namespace,
    deployment_doc: dict[str, Any],
    selected_indices: list[int],
) -> None:
    plan = deployment_plan_from_doc(deployment_doc)
    defaults = plan.get("runtime_defaults") or {}
    args.tool_mode = "json_cli"
    args.read_tool_runtime = "deployment"
    args.llm_tool_runtime = "deployment"
    args.llm_node_runtime = "deployment"
    args._deployment_requires_llama_server = deployment_requires_llama_server(plan, selected_indices)
    args.llm_url = defaults.get("llm_url", DEFAULT_LLM_URL)
    args.llm_model = defaults.get("llm_model", DEFAULT_LLM_MODEL)
    args.llm_max_tokens = int(defaults.get("llm_max_tokens", args.llm_max_tokens))
    args.llm_temperature = float(defaults.get("llm_temperature", args.llm_temperature))
    args.llm_timeout = int(defaults.get("llm_timeout", args.llm_timeout))
    args.llama_health_url = defaults.get("llama_health_url")
    args.llama_local_port = defaults.get("llama_local_port")
    args.llama_remote_port = int(defaults.get("llama_remote_port", args.llama_remote_port))
    args.llama_remote_dir = defaults.get("llama_remote_dir", args.llama_remote_dir)
    args.llama_model_path = defaults.get("llama_model_path", args.llama_model_path)
    args.llama_device = defaults.get("llama_server_device", args.llama_device)
    args.llama_parallel = int(defaults.get("llama_parallel", args.llama_parallel))
    args.llama_startup_timeout = int(defaults.get("llama_startup_timeout", args.llama_startup_timeout))
    args.llama_server_extra_arg = list(defaults.get("llama_server_extra_args", args.llama_server_extra_arg) or [])
    if defaults.get("adb") and not args.adb:
        args.adb = Path(defaults["adb"])
    if defaults.get("adb_serial") and not args.adb_serial:
        args.adb_serial = defaults["adb_serial"]


def runtime_config_path(run_dir: Path, workflow_index: int, workflow: dict[str, Any]) -> Path:
    name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{workflow_index}")
    path = run_dir / "deployment" / "runtime_configs" / f"{workflow_index:02d}_{name}.runtime.json"
    if not path.exists():
        raise FileNotFoundError(f"Deployment runtime config not found. Re-run wfgen_deploy.py: {path}")
    return path


def load_deployed_runtime_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hydrate_runtime_config_for_execution(config: dict[str, Any], args: argparse.Namespace, state_path: Path) -> None:
    adb = str(args.adb.resolve()) if args.adb else ""
    serial = args.adb_serial or ""
    llm_cfg = config.get("llm") or {}
    for cli_cfg in [llm_cfg.get("llama_cli"), *(llm_cfg.get("llama_cli_overrides") or {}).values()]:
        if not isinstance(cli_cfg, dict):
            continue
        if adb:
            cli_cfg["adb"] = adb
        if serial:
            cli_cfg["device_serial"] = serial
    for cli_cfg in [llm_cfg.get("ort_genai"), llm_cfg.get("genie")]:
        if not isinstance(cli_cfg, dict):
            continue
        if adb:
            cli_cfg["adb"] = adb
        if serial:
            cli_cfg["device_serial"] = serial
    for tool_cfg in (config.get("qnn_tools") or {}).values():
        command = tool_cfg.get("command")
        if isinstance(command, list):
            for idx, item in enumerate(command):
                if str(item) == "--state" and idx + 1 < len(command):
                    command[idx + 1] = str(state_path)
        if adb:
            tool_cfg["adb"] = adb
        if serial:
            tool_cfg["device_serial"] = serial
        for key in ("adb",):
            if key in tool_cfg and adb:
                tool_cfg[key] = adb


def warmup_llama_cli_runtime_configs(
    args: argparse.Namespace,
    run_dir: Path,
    workflows: list[dict[str, Any]],
    selected_indices: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    started = time.time()
    artifact_path = output_dir / "LLAMA_CLI_WARMUP.json"
    if args.dry_run:
        info = {
            "status": "skipped",
            "reason": "dry_run",
            "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
            "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "wall_ms": 0.0,
            "runs": [],
        }
        artifact_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        return info
    print("[warmup] preparing llama-cli backends before measured execution", flush=True)
    seen: set[str] = set()
    runs = []
    for workflow_index in selected_indices:
        workflow = workflows[workflow_index]
        name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{workflow_index}")
        state_path = output_dir / "tool_state" / f"{workflow_index:02d}_{name}.json"
        config_path = runtime_config_path(run_dir, workflow_index, workflow)
        config = load_deployed_runtime_config(config_path)
        hydrate_runtime_config_for_execution(config, args, state_path)
        runtime = MobileWorkflowRuntime(config, dry_run=args.dry_run)
        result = runtime.warmup_llama_cli(seen)
        result["workflow_index"] = workflow_index
        result["runtime_config"] = str(config_path)
        runs.append(result)
        warmed = sum(1 for item in result.get("runs") or [] if item.get("status") == "ok")
        skipped = sum(1 for item in result.get("runs") or [] if item.get("status") == "skipped")
        failed = sum(1 for item in result.get("runs") or [] if item.get("status") == "error")
        print(f"[warmup] workflow {workflow_index:02d}: warmed={warmed} skipped={skipped} failed={failed}", flush=True)
    status = "ok" if all(item.get("status") in {"ok", "skipped"} for run in runs for item in (run.get("runs") or [])) else "error"
    info = {
        "status": status,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_ms": (time.time() - started) * 1000.0,
        "unique_config_count": len(seen),
        "runs": runs,
        "artifact": str(artifact_path),
    }
    artifact_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[warmup] status={status} unique_configs={len(seen)} artifact={artifact_path}", flush=True)
    return info


def compact_warmup_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": info.get("status"),
        "wall_ms": info.get("wall_ms"),
        "unique_config_count": info.get("unique_config_count", 0),
        "artifact": info.get("artifact"),
    }


def resolve_workflow_path(args: argparse.Namespace) -> Path:
    if args.plan_dir:
        path = (args.plan_dir / "workflow.json").resolve()
        if not path.exists():
            raise FileNotFoundError(f"--plan-dir does not contain workflow.json: {path}")
        return path
    if args.workflow_json:
        path = args.workflow_json.resolve()
        if not path.exists():
            raise FileNotFoundError(f"workflow_json not found: {path}")
        return path
    candidates = latest_workflow_candidates()
    if not candidates:
        raise FileNotFoundError("No workflow_json provided and no results/**/workflow.json found.")
    path = candidates[-1].resolve()
    print(f"[workflow] using latest workflow JSON: {path}")
    return path


def latest_workflow_candidates() -> list[Path]:
    results_dir = REPO_ROOT / "results"
    if not results_dir.exists():
        return []
    candidates = []
    for path in results_dir.rglob("workflow.json"):
        if not path.is_file():
            continue
        if path.parent == results_dir:
            continue
        if is_planned_workflow_json(path):
            candidates.append(path)
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def is_planned_workflow_json(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    workflows = payload.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        return False
    manifest_path = path.parent / "manifest.json"
    selected_path = path.parent / "selected_samples.json"
    if selected_path.exists():
        return True
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("entrypoint") == "planning-generation-interpret-visualize"
    return False


def resolve_dataset_path(args: argparse.Namespace, deployment_path: Path, deployment_doc: dict[str, Any]) -> Path | None:
    if args.dataset:
        return args.dataset.resolve()
    selected = deployment_doc.get("selected_samples")
    if selected and Path(selected).exists():
        return Path(selected).resolve()
    run_dir = infer_run_dir_from_deployment_path(deployment_path)
    candidate = run_dir / "planning" / "selected_samples.json"
    return candidate.resolve() if candidate.exists() else None


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "results" / f"execute_{stamp}").resolve()


def load_base_config(config_path: Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def maybe_ensure_llama_server(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    info = {
        "enabled": should_ensure_llama_server(args),
        "health_url": health_url(args),
        "started_by_runner": False,
        "reachable_before_launch": False,
        "reachable_after_launch": False,
        "log_path": None,
        "error": None,
    }
    if args.dry_run:
        info["reason"] = "dry_run"
        return info
    if not should_ensure_llama_server(args):
        info["reason"] = "not_required"
        return info
    if args.no_ensure_llama_server:
        info["reason"] = "disabled"
        return info

    health = info["health_url"]
    if endpoint_ready(health, timeout=2.0):
        info["reachable_before_launch"] = True
        info["reachable_after_launch"] = True
        print(f"[llama-server] endpoint already reachable: {health}")
        return info

    try:
        adb_path = discover_adb_path(args.adb)
        serial_args = ["-s", args.adb_serial] if args.adb_serial else []
        info["adb"] = str(adb_path)
        info["adb_serial"] = args.adb_serial
        devices = adb_devices(adb_path)
        info["adb_devices"] = devices
        online_devices = [device for device in devices["devices"] if device.get("state") == "device"]
        if args.adb_serial:
            online_devices = [device for device in online_devices if device.get("serial") == args.adb_serial]
        if not online_devices:
            raise RuntimeError(adb_unavailable_message(devices, args.adb_serial))

        local_port = llama_local_port(args)
        forward_cmd = [str(adb_path), *serial_args, "forward", f"tcp:{local_port}", f"tcp:{args.llama_remote_port}"]
        subprocess.run(forward_cmd, capture_output=True, text=True, timeout=30, check=True)
        info["forward"] = {"local_port": local_port, "remote_port": args.llama_remote_port}

        if endpoint_ready(health, timeout=2.0):
            info["reachable_after_launch"] = True
            info["reason"] = "adb_forward_restored_existing_server"
            print(f"[llama-server] restored adb forward and endpoint is reachable: {health}")
            return info

        log_path = output_dir / "llama_server.log"
        remote_cmd = build_llama_server_command(args)
        info["log_path"] = str(log_path)
        info["remote_command"] = remote_cmd
        log_file = log_path.open("ab")
        subprocess.Popen(
            [str(adb_path), *serial_args, "shell", "sh", "-c", remote_cmd],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        info["started_by_runner"] = True
        print(f"[llama-server] launching phone server; log={log_path}")
        if wait_for_endpoint(health, args.llama_startup_timeout):
            info["reachable_after_launch"] = True
            print(f"[llama-server] ready: {health}")
        else:
            info["error"] = f"Timed out waiting {args.llama_startup_timeout}s for {health}"
            print(f"[llama-server] {info['error']}")
    except subprocess.CalledProcessError as exc:
        info["error"] = called_process_error_message(exc)
        write_startup_error(output_dir / "LLAMA_SERVER_ERROR.txt", exc)
        print(f"[llama-server] startup failed; continuing batch and logging errors: {info['error']}")
    except Exception as exc:
        info["error"] = str(exc)
        write_startup_error(output_dir / "LLAMA_SERVER_ERROR.txt", exc)
        print(f"[llama-server] startup failed; continuing batch and logging errors: {exc}")
    return info


def should_ensure_llama_server(args: argparse.Namespace) -> bool:
    if args.dry_run or args.no_ensure_llama_server:
        return False
    if hasattr(args, "_deployment_requires_llama_server"):
        return bool(args._deployment_requires_llama_server)
    return (
        args.llm_node_runtime == "llama_server"
        or args.read_tool_runtime == "llama_server"
        or args.llm_tool_runtime == "llama_server"
    )


def health_url(args: argparse.Namespace) -> str:
    if args.llama_health_url:
        return args.llama_health_url
    parsed = urllib.parse.urlparse(args.llm_url)
    return urllib.parse.urlunparse((parsed.scheme or "http", parsed.netloc, "/health", "", "", ""))


def endpoint_ready(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def wait_for_endpoint(url: str, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if endpoint_ready(url, timeout=2.0):
            return True
        time.sleep(2)
    return False


def discover_adb_path(explicit_path: Path | None) -> Path:
    if explicit_path:
        path = explicit_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"adb not found: {path}")
        return path
    env_adb = os.environ.get("ADB")
    if env_adb and Path(env_adb).exists():
        return Path(env_adb).resolve()
    found = shutil.which("adb")
    if found:
        return Path(found).resolve()
    raise FileNotFoundError("adb not found. Pass --adb or set ADB.")


def adb_devices(adb_path: Path) -> dict[str, Any]:
    cmd = [str(adb_path), "devices", "-l"]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
    devices = []
    for line in completed.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        serial = parts[0]
        state = parts[1] if len(parts) > 1 else "unknown"
        details = {}
        for token in parts[2:]:
            if ":" in token:
                key, value = token.split(":", 1)
                details[key] = value
        devices.append({"serial": serial, "state": state, "details": details})
    return {
        "command": cmd,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "devices": devices,
    }


def adb_unavailable_message(devices: dict[str, Any], serial: str | None) -> str:
    target = f"serial {serial!r}" if serial else "any device"
    states = ", ".join(f"{device['serial']}:{device['state']}" for device in devices["devices"]) or "none"
    return f"No online adb device found for {target}; adb devices state: {states}"


def called_process_error_message(exc: subprocess.CalledProcessError) -> str:
    stderr = (exc.stderr or "").strip()
    stdout = (exc.stdout or "").strip()
    pieces = [f"Command {exc.cmd!r} returned exit status {exc.returncode}"]
    if stdout:
        pieces.append(f"stdout: {stdout}")
    if stderr:
        pieces.append(f"stderr: {stderr}")
    return "; ".join(pieces)


def llama_local_port(args: argparse.Namespace) -> int:
    if args.llama_local_port:
        return args.llama_local_port
    parsed = urllib.parse.urlparse(args.llm_url)
    if parsed.port:
        return parsed.port
    return 8080


def build_llama_server_command(args: argparse.Namespace) -> str:
    remote_dir = shell_quote(args.llama_remote_dir)
    lib_dir = shell_quote(f"{args.llama_remote_dir}/lib")
    bin_path = shell_quote(f"{args.llama_remote_dir}/bin/llama-server")
    parts = [
        f"cd {remote_dir}",
        (
            f"LD_LIBRARY_PATH={lib_dir} "
            f"ADSP_LIBRARY_PATH={lib_dir} "
            f"GGML_HEXAGON_NDEV=1 "
            f"{bin_path}"
        ),
        "--no-mmap",
        "-m",
        shell_quote(args.llama_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(args.llama_remote_port),
        "--device",
        shell_quote(args.llama_device),
        "-ngl",
        "99",
        "--ctx-size",
        "8192",
        "--ubatch-size",
        "1024",
        "-fa",
        "on",
        "--poll",
        "1000",
        "-t",
        "6",
        "--cpu-mask",
        "0xfc",
        "--cpu-strict",
        "1",
        "--parallel",
        str(args.llama_parallel),
        "--cont-batching",
        *[shell_quote(arg) for arg in args.llama_server_extra_arg],
    ]
    return " && ".join(parts[:1]) + " && " + " ".join(parts[1:])


def shell_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def select_workflow_indices(workflows: list[dict[str, Any]], args: argparse.Namespace) -> list[int]:
    selected: list[int] = []
    if not args.workflow_index and not args.workflow_id:
        return list(range(len(workflows)))

    for spec in args.workflow_index:
        for index in parse_index_spec(spec, len(workflows)):
            if index not in selected:
                selected.append(index)

    ids = {workflow.get("workflow_id"): idx for idx, workflow in enumerate(workflows)}
    for workflow_id in args.workflow_id:
        if workflow_id not in ids:
            raise ValueError(f"workflow_id not found: {workflow_id}")
        index = ids[workflow_id]
        if index not in selected:
            selected.append(index)
    return selected


def parse_index_spec(spec: str, workflow_count: int) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    for index in indices:
        if index < 0 or index >= workflow_count:
            raise IndexError(f"workflow index {index} out of range for {workflow_count} workflows")
    return indices


def build_runtime_config(
    workflow: dict[str, Any],
    *,
    base_config: dict[str, Any],
    args: argparse.Namespace,
    dataset_path: Path | None,
    state_path: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["max_parallel_nodes"] = args.max_parallel_nodes
    llm_config = dict(config.get("llm") or {})
    llm_config.update(
        {
            "runtime": args.llm_node_runtime,
            "url": args.llm_url,
            "model": args.llm_model,
            "max_tokens": args.llm_max_tokens,
            "temperature": args.llm_temperature,
            "timeout": args.llm_timeout,
        }
    )
    if args.llm_node_runtime == "llama_cli":
        llm_config["llama_cli"] = llama_cli_node_runtime_config(args)
        llm_config["llama_cli_overrides"] = llama_cli_node_overrides(args)
    config["llm"] = llm_config
    config["qnn_tools"] = build_tool_configs(workflow, args, dataset_path, state_path)
    return config


def build_tool_configs(
    workflow: dict[str, Any],
    args: argparse.Namespace,
    dataset_path: Path | None,
    state_path: Path,
) -> dict[str, Any]:
    entrypoints = sorted(
        {
            str(node.get("entrypoint") or node.get("node_id"))
            for node in workflow.get("nodes", [])
            if node.get("kind", "app") in {"app", "dnn"}
        }
    )
    if args.tool_mode == "mock":
        return {entrypoint: {"mode": "mock"} for entrypoint in entrypoints}
    if dataset_path is None:
        raise FileNotFoundError("json_cli tool mode needs --dataset or selected_samples.json next to workflow.json.")

    python_executable = args.python_executable or default_python_executable()
    tool_configs = {}
    for entrypoint in entrypoints:
        extra_tool_config = {}
        extra_tool_config.update(llm_business_tool_config(entrypoint, args))
        extra_tool_config.update(llm_tau_tool_config(entrypoint, args))
        tool_timeout = max(120, args.llama_cli_timeout + 60) if extra_tool_config else 120
        tool_configs[entrypoint] = {
            "mode": "json_cli",
            "command": [
                str(python_executable),
                "-m",
                "MOBench.tools.cli",
                entrypoint,
                "--dataset",
                str(dataset_path),
                "--state",
                str(state_path),
            ],
            "cwd": str(PACKAGE_PARENT),
            "timeout": tool_timeout,
            **extra_tool_config,
        }
    return tool_configs


def llm_business_tool_config(entrypoint: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run or entrypoint not in LLM_BUSINESS_TOOLS or args.read_tool_runtime == "script":
        return {}
    if args.read_tool_runtime == "mock":
        return {"implementation": "llm", "tool_runtime": "mock", "mock_business_llm": True}
    if args.read_tool_runtime == "llama_server":
        return {
            "implementation": "llm",
            "tool_runtime": "llama_server",
            "llm_runtime": "llama_server",
            "url": args.llm_url,
            "model": args.llm_model,
            "max_tokens": args.llm_max_tokens,
            "temperature": args.llm_temperature,
            "timeout_seconds": args.llm_timeout,
        }
    return {
        "implementation": "llm",
        "tool_runtime": "llama_cli",
        **llama_cli_tool_runtime_config(args),
    }


def llm_tau_tool_config(entrypoint: str, args: argparse.Namespace) -> dict[str, Any]:
    if entrypoint not in LLM_TAU_TOOLS:
        return {}
    if args.dry_run:
        return {}
    if args.llm_tool_runtime == "mock":
        return {"mock_llm": True}
    if args.llm_tool_runtime == "llama_server":
        return {
            "llm_runtime": "llama_server",
            "url": args.llm_url,
            "model": args.llm_model,
            "max_tokens": args.llm_max_tokens,
            "temperature": args.llm_temperature,
            "timeout_seconds": args.llm_timeout,
        }
    return {
        "llm_runtime": "llama_cli",
        **llama_cli_tool_runtime_config(args),
    }


def llama_cli_tool_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "llama_cli_target": "phone",
        "adb": str(discover_adb_path(args.adb)),
        "device_serial": args.adb_serial or "",
        "remote_dir": args.llama_remote_dir,
        "model_path": args.llama_model_path,
        "llama_device": args.llama_cli_device,
        "ngl": args.llama_cli_ngl if args.llama_cli_ngl is not None else (0 if args.llama_cli_device == "none" else 99),
        "hexagon_env": args.llama_cli_device.startswith("HTP"),
        "max_tokens": args.llm_max_tokens,
        "temperature": args.llm_temperature,
        "llama_timeout": args.llama_cli_timeout,
        "extra_args": args.llama_cli_extra_arg,
        "json_schema": "{}",
        "backend": "llama_cli_phone",
    }


def llama_cli_node_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    device = args.llm_node_cli_device or args.llama_cli_device
    ngl = args.llm_node_cli_ngl
    if ngl is None:
        ngl = args.llama_cli_ngl if args.llama_cli_ngl is not None else (0 if device == "none" else 99)
    return {
        "llama_cli_target": "phone",
        "adb": str(discover_adb_path(args.adb)),
        "device_serial": args.adb_serial or "",
        "remote_dir": args.llama_remote_dir,
        "model_path": args.llama_model_path,
        "llama_device": device,
        "ngl": ngl,
        "hexagon_env": device.startswith("HTP"),
        "max_tokens": args.llm_max_tokens,
        "temperature": args.llm_temperature,
        "llama_timeout": args.llm_node_cli_timeout or args.llama_cli_timeout,
        "extra_args": [*args.llama_cli_extra_arg, *args.llm_node_cli_extra_arg],
        "json_schema": "",
        "backend": "llama_cli_node_phone",
    }


def llama_cli_node_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for item in args.llm_node_cli_device_map:
        if "=" not in item:
            raise ValueError(f"Invalid --llm-node-cli-device-map value {item!r}; expected node_id=device")
        node_id, device = item.split("=", 1)
        node_id = node_id.strip()
        device = device.strip()
        if not node_id or not device:
            raise ValueError(f"Invalid --llm-node-cli-device-map value {item!r}; expected node_id=device")
        overrides[node_id] = {
            "llama_device": device,
            "ngl": 0 if device == "none" else 99,
            "hexagon_env": device.startswith("HTP"),
        }
    return overrides


def default_python_executable() -> Path:
    for venv_python in (REPO_ROOT / ".venv" / "bin" / "python", REPO_ROOT / ".venv" / "Scripts" / "python.exe"):
        if venv_python.exists():
            return venv_python
    return Path(sys.executable)


def print_workflow_list(workflows: list[dict[str, Any]], selected_indices: list[int]) -> None:
    selected = set(selected_indices)
    for idx, workflow in enumerate(workflows):
        mark = "*" if idx in selected else " "
        waves = workflow.get("execution_hints", {}).get("parallel_waves") or []
        print(
            f"{mark} {idx:02d} {workflow.get('workflow_id')} "
            f"nodes={len(workflow.get('nodes', []))} edges={len(workflow.get('edges', []))} waves={len(waves)}"
        )


def error_result(
    workflow: dict[str, Any],
    exc: Exception,
    started: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "workflow_id": workflow.get("workflow_id"),
        "name": workflow.get("name"),
        "status": "error",
        "dry_run": args.dry_run,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_wall_ms": (time.time() - started) * 1000.0,
        "waves": workflow.get("execution_hints", {}).get("parallel_waves") or [],
        "steps": [],
        "artifacts": {},
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def write_error_log(error_log_path: Path, record: dict[str, Any], result: dict[str, Any]) -> None:
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "workflow_index": record.get("workflow_index"),
        "workflow_id": record.get("workflow_id"),
        "status": record.get("status"),
        "result_json": record.get("result_json"),
        "error": result.get("error"),
        "failed_steps": [
            {
                "node_id": step.get("node_id"),
                "success": step.get("success"),
                "raw": step.get("raw"),
            }
            for step in result.get("steps", [])
            if not step.get("success", False)
        ],
    }
    with error_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_startup_error(path: Path, exc: Exception) -> None:
    path.write_text(
        "\n".join(
            [
                f"logged_at_utc: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
                f"type: {type(exc).__name__}",
                f"message: {exc}",
                "",
                traceback.format_exc(),
            ]
        ),
        encoding="utf-8",
    )


def maybe_start_resource_monitor(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[PerfettoResourceMonitor | None, dict[str, Any]]:
    if args.resource_monitor == "off":
        return None, {"enabled": False, "mode": "off", "status": "disabled"}
    resources_dir = output_dir / "resources"
    monitor = PerfettoResourceMonitor(
        ResourceMonitorConfig(
            adb_path=args.adb,
            device_serial=args.adb_serial,
            output_dir=resources_dir,
            tools_dir=args.perfetto_tools_dir,
            sample_interval_s=max(0.1, float(args.resource_sample_interval)),
            buffer_mb=max(1, int(args.perfetto_buffer_mb)),
            required=bool(args.perfetto_required),
            dry_run=bool(args.dry_run),
            python_executable=str(args.python_executable or default_python_executable()),
        )
    )
    try:
        return monitor, monitor.start()
    except Exception as exc:
        if args.perfetto_required:
            raise
        info = {
            "enabled": True,
            "mode": "perfetto",
            "status": "failed_to_start",
            "resource_dir": str(resources_dir),
            "error": str(exc),
        }
        return None, info


def finish_resource_monitor(
    monitor: PerfettoResourceMonitor | None,
    run_records: list[dict[str, Any]],
    required: bool,
) -> dict[str, Any]:
    if monitor is None:
        return {"enabled": False, "mode": "off", "status": "disabled"}
    try:
        return monitor.stop(run_records)
    except Exception as exc:
        if required:
            raise
        info = dict(monitor.info)
        info.update({"status": "failed_to_stop", "error": str(exc)})
        print(f"[resource] monitor stop failed: {exc}", flush=True)
        return info


def apply_resource_results(
    output_dir: Path,
    workflows: list[dict[str, Any]],
    workflow_doc: dict[str, Any],
    run_records: list[dict[str, Any]],
    resource_info: dict[str, Any],
    no_visualize: bool,
) -> None:
    by_workflow = resource_info.get("resource_by_workflow") or load_resource_by_workflow(output_dir)
    compact_monitor = compact_resource_monitor_info(resource_info)
    for record in run_records:
        workflow_index = record["workflow_index"]
        workflow_resource = by_workflow.get(str(workflow_index), {})
        record["resource_summary"] = compact_workflow_resource_summary(workflow_resource)
        result_path = Path(record["result_json"])
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["resource_monitor"] = compact_monitor
        result["resource_usage"] = workflow_resource
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if no_visualize or not record.get("visualization"):
            continue
        visualization_path = Path(record["visualization"])
        visualization_path.write_text(render_html(workflows[workflow_index], workflow_doc, result), encoding="utf-8")


def load_resource_by_workflow(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "resources" / "resource_by_workflow.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def compact_resource_monitor_info(resource_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": resource_info.get("enabled", True),
        "mode": "perfetto" if resource_info.get("enabled", True) else resource_info.get("mode", "off"),
        "status": resource_info.get("status"),
        "resource_dir": resource_info.get("resource_dir"),
        "sample_count": resource_info.get("sample_count"),
        "artifacts": resource_info.get("artifacts", {}),
        "capabilities": resource_info.get("capabilities", {}),
        "errors": resource_info.get("errors", []),
    }


def compact_workflow_resource_summary(resource: dict[str, Any]) -> dict[str, Any]:
    metrics = resource.get("metrics") or {}
    return {
        "sample_count": resource.get("sample_count", 0),
        "cpu_avg_percent": metric_stat(metrics, "cpu_util_percent", "avg"),
        "cpu_max_percent": metric_stat(metrics, "cpu_util_percent", "max"),
        "memory_max_mb": metric_stat(metrics, "mem_used_mb", "max"),
        "gpu_busy_avg_percent": metric_stat(metrics, "gpu_busy_percent", "avg"),
        "gpu_busy_max_percent": metric_stat(metrics, "gpu_busy_percent", "max"),
        "thermal_max_c": metric_stat(metrics, "thermal_max_c", "max"),
        "battery_current_avg_ma": metric_stat(metrics, "battery_current_ma", "avg"),
    }


def metric_stat(metrics: dict[str, Any], metric: str, stat: str) -> Any:
    value = (metrics.get(metric) or {}).get(stat)
    return "" if value is None else value


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    deployment_path: Path,
    dataset_path: Path | None,
    run_records: list[dict[str, Any]],
    llama_server_info: dict[str, Any],
    warmup_info: dict[str, Any],
    resource_info: dict[str, Any],
) -> Path:
    manifest = {
        "entrypoint": ENTRYPOINT_NAME,
        "pipeline_stages": ENTRYPOINT_STAGES,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "workflow_deployment_json": str(deployment_path),
        "dataset": str(dataset_path) if dataset_path else None,
        "control": {
            "workflow_index": args.workflow_index,
            "workflow_id": args.workflow_id,
            "dry_run": args.dry_run,
            "tool_mode": args.tool_mode,
            "read_tool_runtime": args.read_tool_runtime,
            "max_parallel_nodes": args.max_parallel_nodes,
            "llm_node_runtime": args.llm_node_runtime,
            "llm_url": args.llm_url,
            "llm_model": args.llm_model,
            "llm_max_tokens": args.llm_max_tokens,
            "llm_timeout": args.llm_timeout,
            "llm_tool_runtime": args.llm_tool_runtime,
            "llama_cli_device": args.llama_cli_device,
            "llama_cli_ngl": args.llama_cli_ngl,
            "llama_cli_timeout": args.llama_cli_timeout,
            "llm_node_cli_device": args.llm_node_cli_device,
            "llm_node_cli_ngl": args.llm_node_cli_ngl,
            "llm_node_cli_timeout": args.llm_node_cli_timeout,
            "llm_node_cli_device_map": args.llm_node_cli_device_map,
            "resource_monitor": args.resource_monitor,
            "resource_sample_interval": args.resource_sample_interval,
            "perfetto_buffer_mb": args.perfetto_buffer_mb,
            "perfetto_tools_dir": str(args.perfetto_tools_dir),
            "perfetto_required": args.perfetto_required,
        },
        "llama_server": llama_server_info,
        "llama_cli_warmup": compact_warmup_info(warmup_info),
        "resource_monitor": compact_resource_monitor_info(resource_info),
        "runs": run_records,
    }
    path = output_dir / "EXECUTION_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_summary(output_dir: Path, run_records: list[dict[str, Any]], resource_info: dict[str, Any]) -> Path:
    lines = [
        "# MOBench Execution Results",
        "",
        f"Entrypoint: `{ENTRYPOINT_NAME}`",
        f"Stages: `{' -> '.join(ENTRYPOINT_STAGES)}`",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`",
        f"Workflow runs: `{len(run_records)}`",
        "",
        "| Index | Workflow | Status | Dry run | Wall ms | Result | Visualization |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in run_records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["workflow_index"]),
                    str(record["workflow_id"]),
                    str(record["status"]),
                    str(record["dry_run"]),
                    f"{record['total_wall_ms']:.2f}" if record.get("total_wall_ms") is not None else "",
                    str(record["result_json"]),
                    str(record.get("visualization") or ""),
                ]
            )
            + " |"
        )
    resource_monitor = compact_resource_monitor_info(resource_info)
    lines.extend(
        [
            "",
            "## Resource Monitoring",
            "",
            f"Status: `{resource_monitor.get('status')}`",
            f"Samples: `{resource_monitor.get('sample_count') if resource_monitor.get('sample_count') is not None else ''}`",
            f"Resource directory: `{resource_monitor.get('resource_dir') or ''}`",
            "",
            "| Index | Samples | CPU avg % | CPU max % | Memory max MB | GPU busy avg % | GPU busy max % | Thermal max C | Battery current avg mA |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in run_records:
        summary = record.get("resource_summary") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["workflow_index"]),
                    str(summary.get("sample_count", "")),
                    str(summary.get("cpu_avg_percent", "")),
                    str(summary.get("cpu_max_percent", "")),
                    str(summary.get("memory_max_mb", "")),
                    str(summary.get("gpu_busy_avg_percent", "")),
                    str(summary.get("gpu_busy_max_percent", "")),
                    str(summary.get("thermal_max_c", "")),
                    str(summary.get("battery_current_avg_ma", "")),
                ]
            )
            + " |"
        )
    artifacts = resource_monitor.get("artifacts") or {}
    if artifacts:
        lines.extend(
            [
                "",
                "Resource artifacts:",
                "",
                f"- Perfetto trace: `{artifacts.get('perfetto_trace', '')}`",
                f"- Samples CSV: `{artifacts.get('resource_samples', '')}`",
                f"- Summary JSON: `{artifacts.get('resource_summary', '')}`",
                f"- Per-workflow JSON: `{artifacts.get('resource_by_workflow', '')}`",
            ]
        )
    path = output_dir / "EXECUTION_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def sanitize_identifier(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "workflow"
    if value[0].isdigit():
        value = f"workflow_{value}"
    return value


def write_performance_overview(output_dir: Path) -> Path | None:
    try:
        output = output_dir / "performance_overview.html"
        payload = load_dashboard_payload(output_dir)
        output.write_text(render_dashboard(output_dir, payload), encoding="utf-8")
        return output
    except Exception as exc:
        print(f"[visualize] performance overview failed: {exc}", flush=True)
        return None


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"wfgen-execute: error: {exc}", file=sys.stderr)
        raise SystemExit(1)
