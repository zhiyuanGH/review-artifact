"""Benchmark MOBench deployment algorithms and LLM models against one planned workflow set.

This entrypoint intentionally skips workflow planning. It copies an existing
run's planning/ directory into one child run per deployment/model combination,
then runs only wfgen_deploy.py and wfgen_execute.py for each child run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from MOBench.deployment_planning import DEPLOYMENT_STRATEGIES, FAIR_CLI_DEPLOYMENT_STRATEGIES
    from MOBench.deployment_planning.core import best_effort_adb_path
except ImportError:
    from deployment_planning import DEPLOYMENT_STRATEGIES, FAIR_CLI_DEPLOYMENT_STRATEGIES
    from deployment_planning.core import best_effort_adb_path

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_LLAMA_MODEL_PATH = "/data/local/tmp/gguf/Llama-3.2-1B-Instruct-Q4_0.gguf"
ENTRYPOINT_NAME = "deployment-benchmark"
ENTRYPOINT_STAGES = ["deployment", "execution", "comparison_visualization"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.retry_interval is None:
        args.retry_interval = args.inter_run_interval
    context = build_benchmark_context(args)
    source_run_dir = context["source_run_dir"]
    source_planning_dir = context["source_planning_dir"]
    benchmark_dir = context["benchmark_dir"]
    algorithms = context["algorithms"]
    models = context["models"]
    planned_runs = context["planned_runs"]
    multi_model = context["multi_model"]

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (args.progress_file.resolve() if args.progress_file else benchmark_dir / "deployment_benchmark_progress.json")
    python = str(args.python.resolve()) if args.python else sys.executable

    record_by_key = load_existing_records(planned_runs) if args.resume and not args.overwrite else {}
    manifest = {
        "entrypoint": ENTRYPOINT_NAME,
        "pipeline_stages": ENTRYPOINT_STAGES,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "source_run_dir": str(source_run_dir),
        "source_planning_dir": str(source_planning_dir),
        "benchmark_dir": str(benchmark_dir),
        "algorithms": algorithms,
        "models": models,
        "layout": "model/algorithm" if multi_model else "algorithm",
        "resume": args.resume,
        "retry_results_dir": str(args.retry_results_dir.resolve()) if args.retry_results_dir else None,
        "retry_failed": args.retry_failed,
        "retry_non_retryable": args.retry_non_retryable,
        "retry_attempts": args.retry_attempts,
        "retry_max_parallel_nodes": args.retry_max_parallel_nodes,
        "retry_interval_s": args.retry_interval,
        "inter_run_interval_s": args.inter_run_interval,
        "llama_cli_warmup": {
            "runs": args.llama_cli_warmup_runs,
            "max_tokens": args.llama_cli_warmup_max_tokens,
            "prompt": args.llama_cli_warmup_prompt,
        },
        "progress_file": str(progress_path),
        "records": records_for_manifest(planned_runs, record_by_key),
    }
    manifest_path = benchmark_dir / "deployment_benchmark_manifest.json"
    write_state(manifest_path, progress_path, manifest, planned_runs, record_by_key)

    print(f"[bench] source planning: {source_planning_dir}", flush=True)
    print(f"[bench] benchmark dir:   {benchmark_dir}", flush=True)
    print(f"[bench] algorithms:      {', '.join(algorithms)}", flush=True)
    print(f"[bench] models:          {', '.join(model['model_name'] for model in models)}", flush=True)
    print(f"[bench] progress:        {progress_path}", flush=True)
    print(f"[bench] resume:          {args.resume}", flush=True)
    print(f"[bench] retry failed:    {args.retry_failed} (attempts={args.retry_attempts}, retry parallel={args.retry_max_parallel_nodes})", flush=True)
    print(f"[bench] interval:        {args.inter_run_interval:.1f}s", flush=True)

    status_code = 0
    stop_requested = False
    for sequence_index, planned in enumerate(planned_runs, start=1):
        if stop_requested:
            break
        model = planned["model"]
        algorithm = planned["algorithm"]
        run_dir = planned["run_dir"]
        combo_key = planned["combo_key"]
        print("", flush=True)
        print(
            f"[bench] === {sequence_index}/{len(planned_runs)} "
            f"{model['model_name']} :: {algorithm} ===",
            flush=True,
        )
        existing_record = record_by_key.get(combo_key)
        if should_skip_existing(existing_record, args):
            status = str((existing_record or {}).get("status") or "completed")
            print(
                f"[bench] skip terminal combination from previous run "
                f"({status}): {model['model_name']} :: {algorithm}",
                flush=True,
            )
            write_state(
                manifest_path,
                progress_path,
                manifest,
                planned_runs,
                record_by_key,
                current={"combo_key": combo_key, "status": f"skipped_{status}"},
            )
            continue

        record: dict[str, Any] = {
            "combo_key": combo_key,
            "algorithm": algorithm,
            "model_index": planned["model_index"],
            "model_name": model["model_name"],
            "model_path": model["model_path"],
            "model_id": model["model_id"],
            "run_dir": str(run_dir),
            "started_at_utc": utc_now(),
            "status": "started",
            "commands": [],
            "cleanup": [],
            "errors": [],
            "attempts": [],
        }
        if existing_record:
            record["previous_status"] = existing_record.get("status")
            record["previous_finished_at_utc"] = existing_record.get("finished_at_utc")
        record_by_key[combo_key] = record
        write_state(
            manifest_path,
            progress_path,
            manifest,
            planned_runs,
            record_by_key,
            current={"combo_key": combo_key, "status": "started"},
        )

        source_for_run = Path(planned.get("source_planning_dir") or source_planning_dir)
        if not (source_for_run / "workflow.json").exists():
            source_for_run = source_planning_dir
        if not (source_for_run / "workflow.json").exists():
            raise FileNotFoundError(f"Source planning directory does not contain workflow.json: {source_for_run}")

        existing_status = str((existing_record or {}).get("status") or "")
        retrying_existing = bool(
            args.retry_failed
            and existing_record
            and existing_status != "completed"
            and (existing_status != "execution_partial_non_retryable" or args.retry_non_retryable)
        )
        max_attempts = max(1, args.retry_attempts if retrying_existing else 1 + (args.retry_attempts if args.retry_failed else 0))
        for attempt_index in range(max_attempts):
            max_parallel_nodes = (
                args.retry_max_parallel_nodes
                if retrying_existing or attempt_index > 0
                else args.max_parallel_nodes
            )
            attempt_status = "retrying" if retrying_existing or attempt_index > 0 else "started"
            record["status"] = attempt_status
            write_state(
                manifest_path,
                progress_path,
                manifest,
                planned_runs,
                record_by_key,
                current={
                    "combo_key": combo_key,
                    "status": attempt_status,
                    "attempt_index": attempt_index,
                    "max_parallel_nodes": max_parallel_nodes,
                },
            )
            if attempt_index > 0:
                cooldown_between_runs(
                    args.retry_interval,
                    manifest_path,
                    progress_path,
                    manifest,
                    planned_runs,
                    record_by_key,
                    sequence_index - 1,
                    label=f"retry {attempt_index}",
                )
            is_retry = retrying_existing or attempt_index > 0
            if is_retry and can_merge_retry(Path(planned["run_dir"])):
                attempt = run_merge_retry_attempt(
                    python=python,
                    planned=planned,
                    source_planning_dir=source_for_run,
                    benchmark_dir=benchmark_dir,
                    args=args,
                    max_parallel_nodes=max_parallel_nodes,
                    attempt_index=attempt_index,
                )
            else:
                attempt = run_benchmark_attempt(
                    python=python,
                    planned=planned,
                    source_planning_dir=source_for_run,
                    benchmark_dir=benchmark_dir,
                    args=args,
                    max_parallel_nodes=max_parallel_nodes,
                    attempt_index=attempt_index,
                    is_retry=is_retry,
                )
            record["attempts"].append(attempt)
            apply_attempt_to_record(record, attempt)
            record_by_key[combo_key] = record
            write_json(run_dir / "benchmark_run_record.json", record)
            write_state(
                manifest_path,
                progress_path,
                manifest,
                planned_runs,
                record_by_key,
                current={
                    "combo_key": combo_key,
                    "status": record["status"],
                    "attempt_index": attempt_index,
                    "max_parallel_nodes": max_parallel_nodes,
                },
            )

            if record["status"] == "completed":
                break
            if not args.retry_failed:
                break
            if not attempt.get("retryable", False):
                if record["status"] == "execution_partial":
                    record["status"] = "execution_partial_non_retryable"
                    record["retryability"] = attempt.get("retryability")
                break
            if attempt_index >= args.retry_attempts:
                record["status"] = "retry_exhausted"
                record["retryability"] = attempt.get("retryability")
                break
            print(
                f"[bench] retrying {model['model_name']} :: {algorithm} "
                f"with max_parallel_nodes={args.retry_max_parallel_nodes}",
                flush=True,
            )

        record["finished_at_utc"] = utc_now()
        write_json(run_dir / "benchmark_run_record.json", record)
        record_by_key[combo_key] = record
        write_state(
            manifest_path,
            progress_path,
            manifest,
            planned_runs,
            record_by_key,
            current={"combo_key": combo_key, "status": record["status"]},
        )
        if record["status"] in {"failed", "retry_exhausted"}:
            status_code = 1
            if args.fail_fast:
                stop_requested = True
        if stop_requested:
            break
        if record.get("executed"):
            cooldown_between_runs(
                args.inter_run_interval,
                manifest_path,
                progress_path,
                manifest,
                planned_runs,
                record_by_key,
                sequence_index,
            )

    try:
        from visualizations.visualize_deployment_benchmarks import write_benchmark_visualizations

        paths = write_benchmark_visualizations(benchmark_dir)
        manifest["comparison_artifacts"] = {key: str(value) for key, value in paths.items()}
        write_state(manifest_path, progress_path, manifest, planned_runs, record_by_key)
        print(f"[bench] comparison html: {paths.get('html')}", flush=True)
    except Exception as exc:
        status_code = 1
        manifest["comparison_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_state(manifest_path, progress_path, manifest, planned_runs, record_by_key)
        print(f"[bench] comparison visualization failed: {exc}", flush=True)

    print("", flush=True)
    print(f"[bench] finished: {benchmark_dir}", flush=True)
    return status_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark deployment algorithms by reusing an existing MOBench planning/ directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-run-dir", type=Path, default=None, help="Run root containing planning/workflow.json. Defaults to the latest planned run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Benchmark root. Defaults to results/deployment_benchmark_<timestamp>.")
    parser.add_argument("--retry-results-dir", type=Path, default=None, help="Existing deployment benchmark root or child run directory. Retries only non-terminal cases unless --retry-non-retryable is set.")
    parser.add_argument(
        "--algorithms",
        default=None,
        help="Comma-separated deployment algorithms to benchmark.",
    )
    parser.add_argument("--python", type=Path, default=None, help="Python executable used to run MOBench scripts.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing per-algorithm child directories under the benchmark root.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip completed combinations in an existing benchmark output directory and rerun failed/incomplete combinations.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume behavior; existing child directories require --overwrite.")
    parser.add_argument("--progress-file", type=Path, default=None, help="Progress JSON path. Defaults to <benchmark-dir>/deployment_benchmark_progress.json.")
    parser.add_argument("--inter-run-interval", type=float, default=60.0, help="Cooldown seconds between executed model/algorithm combinations. Use 0 to disable.")
    parser.add_argument("--retry-failed", dest="retry_failed", action="store_true", default=True, help="Retry failed or partial combinations with lower parallelism.")
    parser.add_argument("--no-retry-failed", dest="retry_failed", action="store_false", help="Disable retry attempts after a failed or partial combination.")
    parser.add_argument("--retry-non-retryable", action="store_true", help="Also retry combinations previously marked execution_partial_non_retryable, useful after fixing deterministic tool/data code.")
    parser.add_argument("--retry-attempts", type=int, default=1, help="Retry attempts after the initial attempt.")
    parser.add_argument("--retry-max-parallel-nodes", type=int, default=3, help="Max parallel workflow nodes used for retry attempts.")
    parser.add_argument("--retry-interval", type=float, default=None, help="Cooldown seconds before retry attempts. Defaults to --inter-run-interval.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed algorithm. By default failures are logged and the benchmark continues.")
    parser.add_argument("--no-stop-between-runs", action="store_true", help="Do not stop stale llama processes or remove ADB forwards between algorithms.")
    parser.add_argument("--dry-run", action="store_true", help="Execute with mocked node outputs.")
    parser.add_argument("--workflow-index", action="append", default=[], help="Workflow index, comma list, or range to pass to wfgen_execute.py.")
    parser.add_argument("--workflow-id", action="append", default=[], help="Workflow id to pass to wfgen_execute.py.")
    parser.add_argument("--max-parallel-nodes", type=int, default=4, help="Maximum workflow nodes to run in parallel per wave.")
    parser.add_argument("--llm-max-tokens", type=int, default=64, help="Maximum generated tokens per LLM node/tool call.")
    parser.add_argument("--llama-cli-timeout", type=int, default=240, help="Timeout in seconds for phone llama-cli calls.")
    parser.add_argument("--llama-cli-warmup-runs", type=int, default=1, help="Warmup llama-cli calls per unique model/backend before measured execution; use 0 for cold-start timing.")
    parser.add_argument("--llama-cli-warmup-max-tokens", type=int, default=1, help="Generated tokens for llama-cli warmup calls.")
    parser.add_argument("--llama-cli-warmup-prompt", default="OK", help="Prompt text used for llama-cli warmup calls.")
    parser.add_argument("--llama-remote-dir", default="/data/local/tmp/llama.cpp", help="Phone directory containing llama.cpp bin/ and lib/.")
    parser.add_argument("--llama-model-path", default=DEFAULT_LLAMA_MODEL_PATH, help="Phone-side GGUF model path used when --llama-models is omitted.")
    parser.add_argument("--llama-models", default="", help="Comma-separated phone-side GGUF model paths or basenames to benchmark across every algorithm.")
    parser.add_argument("--llama-cli-extra-arg", action="append", default=[], help="Extra llama-cli argument. Can be repeated.")
    parser.add_argument("--adb", type=Path, default=None, help="adb executable.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb device serial.")
    parser.add_argument("--resource-monitor", choices=["perfetto", "off"], default="perfetto", help="Phone resource monitor used around execution.")
    parser.add_argument("--resource-sample-interval", type=float, default=1.0, help="Seconds between lightweight ADB resource samples.")
    parser.add_argument("--perfetto-buffer-mb", type=int, default=128, help="Perfetto ring buffer size in MB.")
    parser.add_argument("--perfetto-tools-dir", type=Path, default=REPO_ROOT / ".wfgen_perfetto", help="Local directory for PC-side Perfetto tooling.")
    parser.add_argument("--perfetto-required", action="store_true", help="Fail execution if Perfetto monitoring cannot start or stop cleanly.")
    parser.add_argument("--no-deploy-visualize", action="store_true", help="Skip deployment visualizations.")
    parser.add_argument("--no-execute-visualize", action="store_true", help="Skip execution visualizations.")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop each workflow batch after the first failed workflow.")
    parser.add_argument("--fail-on-error", action="store_true", help="Return non-zero from wfgen_execute.py if any workflow fails.")
    return parser.parse_args(argv)


def build_benchmark_context(args: argparse.Namespace) -> dict[str, Any]:
    if args.retry_results_dir:
        return build_retry_context(args)

    source_run_dir = resolve_source_run_dir(args.source_run_dir)
    source_planning_dir = source_run_dir / "planning"
    if not (source_planning_dir / "workflow.json").exists():
        raise FileNotFoundError(f"Source run does not contain planning/workflow.json: {source_run_dir}")

    benchmark_dir = resolve_benchmark_dir(args.output_dir)
    algorithms = parse_algorithms(args.algorithms)
    models = parse_models(args)
    multi_model = len(models) > 1
    planned_runs = build_planned_runs(benchmark_dir, models, algorithms, multi_model)
    return {
        "source_run_dir": source_run_dir,
        "source_planning_dir": source_planning_dir,
        "benchmark_dir": benchmark_dir,
        "algorithms": algorithms,
        "models": models,
        "multi_model": multi_model,
        "planned_runs": planned_runs,
    }


def build_retry_context(args: argparse.Namespace) -> dict[str, Any]:
    retry_dir = args.retry_results_dir.resolve()
    if not retry_dir.exists():
        raise FileNotFoundError(f"--retry-results-dir does not exist: {retry_dir}")
    if args.output_dir and args.output_dir.resolve() != retry_dir:
        raise ValueError("--retry-results-dir selects the output directory; do not pass a different --output-dir.")

    if (retry_dir / "deployment_benchmark_manifest.json").exists():
        manifest = load_json(retry_dir / "deployment_benchmark_manifest.json", {})
        benchmark_dir = retry_dir
        planned_runs = planned_runs_from_manifest(benchmark_dir, manifest, args)
    elif (retry_dir / "benchmark_run_record.json").exists():
        benchmark_dir = find_benchmark_root_for_child(retry_dir)
        manifest = load_json(benchmark_dir / "deployment_benchmark_manifest.json", {})
        planned_runs = [planned_run_from_child(retry_dir, manifest)]
    else:
        raise FileNotFoundError(
            "--retry-results-dir must point to a benchmark root with deployment_benchmark_manifest.json "
            f"or a child run with benchmark_run_record.json: {retry_dir}"
        )

    if not planned_runs:
        raise ValueError(f"No benchmark child runs matched retry filters under {retry_dir}")

    source_planning_dir = resolve_retry_source_planning_dir(manifest, planned_runs)
    source_run_dir = source_planning_dir.parent if source_planning_dir else retry_dir
    algorithms = ordered_unique([planned["algorithm"] for planned in planned_runs])
    models = ordered_models_from_planned(planned_runs)
    multi_model = benchmark_uses_model_layout(planned_runs)
    args.output_dir = benchmark_dir
    args.resume = True
    return {
        "source_run_dir": source_run_dir,
        "source_planning_dir": source_planning_dir,
        "benchmark_dir": benchmark_dir,
        "algorithms": algorithms,
        "models": models,
        "multi_model": multi_model,
        "planned_runs": planned_runs,
    }


def parse_algorithms(value: str | None) -> list[str]:
    raw_value = value or ",".join(FAIR_CLI_DEPLOYMENT_STRATEGIES)
    algorithms = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not algorithms:
        raise ValueError("At least one deployment algorithm is required.")
    unknown = [item for item in algorithms if item not in DEPLOYMENT_STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown deployment algorithm(s): {', '.join(unknown)}")
    return algorithms


def parse_models(args: argparse.Namespace) -> list[dict[str, str]]:
    raw_models = args.llama_models.strip() or args.llama_model_path
    paths = [item.strip() for item in raw_models.split(",") if item.strip()]
    if not paths:
        paths = [DEFAULT_LLAMA_MODEL_PATH]
    models = []
    seen_ids: set[str] = set()
    for index, path in enumerate(paths):
        model_path = normalize_model_path(path)
        model_name = model_path.rstrip("/").rsplit("/", 1)[-1] or f"model_{index}"
        model_id = unique_model_id(model_name, seen_ids)
        models.append({"model_path": model_path, "model_name": model_name, "model_id": model_id})
    return models


def normalize_model_path(path: str) -> str:
    if "/" in path:
        return path
    return f"/data/local/tmp/gguf/{path}"


def unique_model_id(model_name: str, seen: set[str]) -> str:
    base = sanitize_name(model_name)
    if base.lower().endswith("_gguf"):
        base = base[:-5] or "model"
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}_{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def sanitize_name(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "model"


def resolve_source_run_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    candidates = []
    if DEFAULT_RESULTS_DIR.exists():
        for workflow_path in DEFAULT_RESULTS_DIR.rglob("planning/workflow.json"):
            if "deployment_benchmark_" in str(workflow_path):
                continue
            candidates.append(workflow_path.parent.parent)
    if not candidates:
        raise FileNotFoundError("No --source-run-dir provided and no results/**/planning/workflow.json found.")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1].resolve()


def resolve_benchmark_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    return DEFAULT_RESULTS_DIR / f"deployment_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def combination_run_dir(benchmark_dir: Path, model: dict[str, str], algorithm: str, multi_model: bool) -> Path:
    if multi_model:
        return benchmark_dir / model["model_id"] / algorithm
    return benchmark_dir / algorithm


def build_planned_runs(
    benchmark_dir: Path,
    models: list[dict[str, str]],
    algorithms: list[str],
    multi_model: bool,
) -> list[dict[str, Any]]:
    planned = []
    for model_index, model in enumerate(models):
        for algorithm in algorithms:
            combo_key = f"{model['model_id']}::{algorithm}"
            run_dir = combination_run_dir(benchmark_dir, model, algorithm, multi_model)
            planned.append(
                {
                    "combo_key": combo_key,
                    "model_index": model_index,
                    "model": model,
                    "algorithm": algorithm,
                    "run_dir": run_dir,
                }
            )
    return planned


def planned_runs_from_manifest(
    benchmark_dir: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    requested_algorithms = set(parse_algorithms(args.algorithms)) if args.algorithms else None
    requested_models = requested_model_filters(args)
    records = list(manifest.get("records") or [])
    planned = []
    if records:
        for sequence, record in enumerate(records):
            raw_run_dir = str(record.get("run_dir") or "")
            if not raw_run_dir:
                continue
            run_dir = Path(raw_run_dir).resolve()
            algorithm = str(record.get("algorithm") or run_dir.name)
            if requested_algorithms and algorithm not in requested_algorithms:
                continue
            model = model_from_record(record, sequence)
            if requested_models and model["model_path"] not in requested_models and model["model_name"] not in requested_models:
                continue
            combo_key = str(record.get("combo_key") or f"{model['model_id']}::{algorithm}")
            planned.append(
                {
                    "combo_key": combo_key,
                    "model_index": int(record.get("model_index") or sequence),
                    "model": model,
                    "algorithm": algorithm,
                    "run_dir": run_dir,
                    "source_planning_dir": str(run_dir / "planning"),
                }
            )
        return planned

    algorithms = parse_algorithms(args.algorithms or ",".join(manifest.get("algorithms") or FAIR_CLI_DEPLOYMENT_STRATEGIES))
    models = models_from_manifest(manifest) if not args.llama_models.strip() else parse_models(args)
    if requested_algorithms:
        algorithms = [item for item in algorithms if item in requested_algorithms]
    multi_model = len(models) > 1
    return build_planned_runs(benchmark_dir, models, algorithms, multi_model)


def planned_run_from_child(child_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    record = load_json(child_dir / "benchmark_run_record.json", {})
    algorithm = str(record.get("algorithm") or (load_json(child_dir / "deployment" / "deployment_plan.json", {}) or {}).get("strategy") or child_dir.name)
    model = model_from_record(record, 0)
    combo_key = str(record.get("combo_key") or f"{model['model_id']}::{algorithm}")
    parent_records = manifest.get("records") or []
    for parent_record in parent_records:
        if Path(str(parent_record.get("run_dir") or "")).resolve() == child_dir.resolve():
            combo_key = str(parent_record.get("combo_key") or combo_key)
            model = model_from_record(parent_record, 0)
            algorithm = str(parent_record.get("algorithm") or algorithm)
            break
    return {
        "combo_key": combo_key,
        "model_index": int(record.get("model_index") or 0),
        "model": model,
        "algorithm": algorithm,
        "run_dir": child_dir.resolve(),
        "source_planning_dir": str(child_dir.resolve() / "planning"),
    }


def requested_model_filters(args: argparse.Namespace) -> set[str]:
    if not args.llama_models.strip():
        return set()
    return {normalize_model_path(item.strip()) for item in args.llama_models.split(",") if item.strip()}


def model_from_record(record: dict[str, Any], index: int) -> dict[str, str]:
    model_path = str(record.get("model_path") or DEFAULT_LLAMA_MODEL_PATH)
    model_name = str(record.get("model_name") or model_path.rstrip("/").rsplit("/", 1)[-1] or f"model_{index}")
    model_id = str(record.get("model_id") or sanitize_name(model_name))
    return {"model_path": model_path, "model_name": model_name, "model_id": model_id}


def models_from_manifest(manifest: dict[str, Any]) -> list[dict[str, str]]:
    models = []
    seen: set[str] = set()
    for index, model in enumerate(manifest.get("models") or []):
        model_path = str(model.get("model_path") or DEFAULT_LLAMA_MODEL_PATH)
        model_name = str(model.get("model_name") or model_path.rstrip("/").rsplit("/", 1)[-1] or f"model_{index}")
        model_id = str(model.get("model_id") or unique_model_id(model_name, seen))
        if model_id in seen:
            model_id = unique_model_id(model_id, seen)
        else:
            seen.add(model_id)
        models.append({"model_path": model_path, "model_name": model_name, "model_id": model_id})
    return models or [{"model_path": DEFAULT_LLAMA_MODEL_PATH, "model_name": DEFAULT_LLAMA_MODEL_PATH.rsplit("/", 1)[-1], "model_id": "Llama-3_2-1B-Instruct-Q4_0"}]


def ordered_models_from_planned(planned_runs: list[dict[str, Any]]) -> list[dict[str, str]]:
    models = []
    seen: set[str] = set()
    for planned in planned_runs:
        model = planned["model"]
        key = model["model_id"]
        if key in seen:
            continue
        seen.add(key)
        models.append(model)
    return models


def ordered_unique(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def benchmark_uses_model_layout(planned_runs: list[dict[str, Any]]) -> bool:
    return len({planned["model"]["model_id"] for planned in planned_runs}) > 1


def find_benchmark_root_for_child(child_dir: Path) -> Path:
    for parent in [child_dir.parent, child_dir.parent.parent, *child_dir.parents]:
        if (parent / "deployment_benchmark_manifest.json").exists():
            return parent.resolve()
    return child_dir.parent.resolve()


def resolve_retry_source_planning_dir(manifest: dict[str, Any], planned_runs: list[dict[str, Any]]) -> Path:
    for key in ("source_planning_dir",):
        raw = manifest.get(key)
        if raw and (Path(str(raw)) / "workflow.json").exists():
            return Path(str(raw)).resolve()
    source_run = manifest.get("source_run_dir")
    if source_run and (Path(str(source_run)) / "planning" / "workflow.json").exists():
        return (Path(str(source_run)) / "planning").resolve()
    for planned in planned_runs:
        candidate = Path(str(planned.get("source_planning_dir") or "")) if planned.get("source_planning_dir") else Path(planned["run_dir"]) / "planning"
        if (candidate / "workflow.json").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find planning/workflow.json for retry.")


def load_existing_records(planned_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for planned in planned_runs:
        path = Path(planned["run_dir"]) / "benchmark_run_record.json"
        record = load_json(path, {})
        if not isinstance(record, dict) or not record:
            continue
        combo_key = str(record.get("combo_key") or planned["combo_key"])
        record.setdefault("combo_key", combo_key)
        record.setdefault("algorithm", planned["algorithm"])
        record.setdefault("model_index", planned["model_index"])
        record.setdefault("model_name", planned["model"]["model_name"])
        record.setdefault("model_path", planned["model"]["model_path"])
        record.setdefault("model_id", planned["model"]["model_id"])
        record.setdefault("run_dir", str(planned["run_dir"]))
        records[combo_key] = record
    return records


def should_skip_existing(record: dict[str, Any] | None, args: argparse.Namespace) -> bool:
    if not args.resume or args.overwrite or not record:
        return False
    status = str(record.get("status") or "")
    if status == "completed":
        return True
    if status == "execution_partial_non_retryable":
        return not args.retry_non_retryable
    return False


def prepare_run_dir(
    run_dir: Path,
    source_planning_dir: Path,
    benchmark_dir: Path,
    *,
    overwrite: bool,
    replace_existing: bool,
) -> None:
    if run_dir.exists():
        if not (overwrite or replace_existing):
            raise FileExistsError(f"Benchmark child directory already exists. Pass --overwrite or --resume to replace it: {run_dir}")
        assert_child_path(run_dir, benchmark_dir)
        if path_is_relative_to(source_planning_dir.resolve(), run_dir.resolve()):
            clean_retry_run_dir(run_dir)
        else:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_planning_dir, run_dir / "planning")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_planning_dir, run_dir / "planning")
    print(f"[bench] copied planning -> {run_dir / 'planning'}", flush=True)


def assert_child_path(child: Path, parent: Path) -> None:
    child.resolve().relative_to(parent.resolve())


def path_is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def clean_retry_run_dir(run_dir: Path) -> None:
    for name in [
        "deployment",
        "execution",
        "benchmark_run_record.json",
    ]:
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def deploy_command(
    python: str,
    run_dir: Path,
    algorithm: str,
    model_path: str,
    args: argparse.Namespace,
    max_parallel_nodes: int,
) -> list[str]:
    cmd = [
        python,
        str(REPO_ROOT / "wfgen_deploy.py"),
        "--run-dir",
        str(run_dir),
        "--deployment-strategy",
        algorithm,
        "--max-parallel-nodes",
        str(max_parallel_nodes),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--llama-cli-timeout",
        str(args.llama_cli_timeout),
        "--llama-cli-warmup-runs",
        str(args.llama_cli_warmup_runs),
        "--llama-cli-warmup-max-tokens",
        str(args.llama_cli_warmup_max_tokens),
        "--llama-cli-warmup-prompt",
        args.llama_cli_warmup_prompt,
        "--llama-remote-dir",
        args.llama_remote_dir,
        "--llama-model-path",
        model_path,
    ]
    for extra in args.llama_cli_extra_arg:
        cmd.extend(["--llama-cli-extra-arg", extra])
    if args.adb:
        cmd.extend(["--adb", str(args.adb)])
    if args.adb_serial:
        cmd.extend(["--adb-serial", args.adb_serial])
    if args.no_deploy_visualize:
        cmd.append("--no-visualize")
    return cmd


def execute_command(
    python: str,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    workflow_indices: list[str] | None = None,
) -> list[str]:
    cmd = [
        python,
        str(REPO_ROOT / "wfgen_execute.py"),
        "--run-dir",
        str(run_dir),
        "--resource-monitor",
        args.resource_monitor,
        "--resource-sample-interval",
        str(args.resource_sample_interval),
        "--perfetto-buffer-mb",
        str(args.perfetto_buffer_mb),
        "--perfetto-tools-dir",
        str(args.perfetto_tools_dir),
    ]
    selected_workflow_indices = workflow_indices if workflow_indices is not None else args.workflow_index
    for workflow_index in selected_workflow_indices:
        cmd.extend(["--workflow-index", workflow_index])
    for workflow_id in args.workflow_id:
        cmd.extend(["--workflow-id", workflow_id])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.adb:
        cmd.extend(["--adb", str(args.adb)])
    if args.adb_serial:
        cmd.extend(["--adb-serial", args.adb_serial])
    if args.perfetto_required:
        cmd.append("--perfetto-required")
    if args.no_execute_visualize:
        cmd.append("--no-visualize")
    if args.stop_on_failure:
        cmd.append("--stop-on-failure")
    if args.fail_on_error:
        cmd.append("--fail-on-error")
    return cmd


def run_benchmark_attempt(
    *,
    python: str,
    planned: dict[str, Any],
    source_planning_dir: Path,
    benchmark_dir: Path,
    args: argparse.Namespace,
    max_parallel_nodes: int,
    attempt_index: int,
    is_retry: bool,
) -> dict[str, Any]:
    model = planned["model"]
    algorithm = planned["algorithm"]
    run_dir = Path(planned["run_dir"])
    attempt: dict[str, Any] = {
        "attempt_index": attempt_index,
        "started_at_utc": utc_now(),
        "status": "started",
        "is_retry": is_retry,
        "max_parallel_nodes": max_parallel_nodes,
        "commands": [],
        "cleanup": [],
        "errors": [],
        "retryable": False,
    }
    executed = False
    try:
        prepare_run_dir(
            run_dir,
            source_planning_dir,
            benchmark_dir,
            overwrite=args.overwrite,
            replace_existing=args.resume or args.retry_results_dir is not None or attempt_index > 0,
        )
        executed = True
        if not args.no_stop_between_runs:
            attempt["cleanup"].append(stop_stale_phone_runtimes(args))

        deploy_cmd = deploy_command(python, run_dir, algorithm, model["model_path"], args, max_parallel_nodes)
        deploy_record = run_command(deploy_cmd)
        attempt["commands"].append(deploy_record)
        if deploy_record["returncode"] != 0:
            attempt["status"] = "failed"
            attempt["retryable"] = True
            attempt["retryability"] = {
                "retryable": True,
                "retryable_reasons": [f"deployment_returncode_{deploy_record['returncode']}"],
                "non_retryable_reasons": [],
            }
            attempt["errors"].append(
                {
                    "type": "deployment_failed",
                    "message": (
                        f"Deployment failed for {algorithm} / {model['model_name']} "
                        f"with exit code {deploy_record['returncode']}"
                    ),
                }
            )
            return attempt

        execute_cmd = execute_command(python, run_dir, args)
        execute_record = run_command(execute_cmd)
        attempt["commands"].append(execute_record)
        if execute_record["returncode"] != 0:
            attempt["status"] = "failed"
            attempt["retryable"] = True
            attempt["retryability"] = {
                "retryable": True,
                "retryable_reasons": [f"execution_returncode_{execute_record['returncode']}"],
                "non_retryable_reasons": [],
            }
            attempt["errors"].append(
                {
                    "type": "execution_failed",
                    "message": (
                        f"Execution failed for {algorithm} / {model['model_name']} "
                        f"with exit code {execute_record['returncode']}"
                    ),
                }
            )
            return attempt

        execution_failures = execution_failure_summary(run_dir)
        attempt["execution_failures"] = execution_failures
        attempt["retryability"] = execution_failures.get("retryability", {})
        attempt["retryable"] = bool((execution_failures.get("retryability") or {}).get("retryable"))
        if execution_failures["failed_workflows"]:
            attempt["status"] = "execution_partial"
            attempt["errors"].append(
                {
                    "type": "workflow_failures",
                    "message": (
                        f"{execution_failures['failed_workflows']} of "
                        f"{execution_failures['workflow_count']} workflows failed."
                    ),
                }
            )
        else:
            attempt["status"] = "completed"
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["retryable"] = True
        attempt["retryability"] = {
            "retryable": True,
            "retryable_reasons": [type(exc).__name__],
            "non_retryable_reasons": [],
        }
        attempt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        print(f"[bench] {algorithm} / {model['model_name']} failed: {exc}", flush=True)
    finally:
        if executed and not args.no_stop_between_runs:
            attempt.setdefault("cleanup", []).append(stop_stale_phone_runtimes(args))
        attempt["finished_at_utc"] = utc_now()
    return attempt


def can_merge_retry(run_dir: Path) -> bool:
    summary = execution_failure_summary(run_dir)
    return bool(summary.get("failed_workflow_indices"))


def run_merge_retry_attempt(
    *,
    python: str,
    planned: dict[str, Any],
    source_planning_dir: Path,
    benchmark_dir: Path,
    args: argparse.Namespace,
    max_parallel_nodes: int,
    attempt_index: int,
) -> dict[str, Any]:
    model = planned["model"]
    algorithm = planned["algorithm"]
    run_dir = Path(planned["run_dir"])
    original_summary = execution_failure_summary(run_dir)
    failed_indices = [
        int(index)
        for index in original_summary.get("failed_workflow_indices") or []
        if index is not None
    ]
    requested_indices = requested_workflow_index_filter(args)
    if requested_indices:
        failed_indices = [index for index in failed_indices if index in requested_indices]
    attempt: dict[str, Any] = {
        "attempt_index": attempt_index,
        "started_at_utc": utc_now(),
        "status": "retrying",
        "is_retry": True,
        "merge_retry": True,
        "max_parallel_nodes": max_parallel_nodes,
        "commands": [],
        "cleanup": [],
        "errors": [],
        "retryable": bool((original_summary.get("retryability") or {}).get("retryable")),
        "retryability": original_summary.get("retryability") or {},
        "target_failed_workflow_indices": failed_indices,
    }
    if not failed_indices:
        attempt["status"] = "completed" if original_summary.get("failed_workflows") == 0 else "execution_partial"
        attempt["finished_at_utc"] = utc_now()
        return attempt

    retry_dir = retry_attempt_dir(benchmark_dir, planned, attempt_index)
    attempt["retry_dir"] = str(retry_dir)
    executed = False
    try:
        if retry_dir.exists():
            assert_child_path(retry_dir, benchmark_dir)
            shutil.rmtree(retry_dir)
        retry_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_planning_dir, retry_dir / "planning")
        print(
            f"[bench] retrying failed workflow indices only: "
            f"{','.join(str(index) for index in failed_indices)}",
            flush=True,
        )
        if not args.no_stop_between_runs:
            attempt["cleanup"].append(stop_stale_phone_runtimes(args))

        deploy_cmd = deploy_command(python, retry_dir, algorithm, model["model_path"], args, max_parallel_nodes)
        deploy_record = run_command(deploy_cmd)
        attempt["commands"].append(deploy_record)
        if deploy_record["returncode"] != 0:
            attempt["status"] = "failed"
            attempt["retryable"] = True
            attempt["errors"].append({"type": "deployment_failed", "message": f"Retry deployment failed with exit code {deploy_record['returncode']}"})
            return attempt

        execute_cmd = execute_command(
            python,
            retry_dir,
            args,
            workflow_indices=[str(index) for index in failed_indices],
        )
        execute_record = run_command(execute_cmd)
        attempt["commands"].append(execute_record)
        executed = True
        if execute_record["returncode"] != 0:
            attempt["status"] = "failed"
            attempt["retryable"] = True
            attempt["errors"].append({"type": "execution_failed", "message": f"Retry execution failed with exit code {execute_record['returncode']}"})
            return attempt

        merge_result = merge_successful_retry_results(run_dir, retry_dir, max_parallel_nodes)
        attempt["merge_result"] = merge_result
        updated_summary = execution_failure_summary(run_dir)
        attempt["execution_failures"] = updated_summary
        attempt["retryability"] = updated_summary.get("retryability") or {}
        attempt["retryable"] = bool((updated_summary.get("retryability") or {}).get("retryable"))
        if updated_summary["failed_workflows"]:
            attempt["status"] = "execution_partial"
            attempt["errors"].append(
                {
                    "type": "workflow_failures",
                    "message": (
                        f"{updated_summary['failed_workflows']} of "
                        f"{updated_summary['workflow_count']} workflows still failed after retry."
                    ),
                }
            )
        else:
            attempt["status"] = "completed"
    except Exception as exc:
        attempt["status"] = "failed"
        attempt["retryable"] = True
        attempt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        print(f"[bench] retry merge failed for {algorithm} / {model['model_name']}: {exc}", flush=True)
    finally:
        if executed and not args.no_stop_between_runs:
            attempt.setdefault("cleanup", []).append(stop_stale_phone_runtimes(args))
        attempt["finished_at_utc"] = utc_now()
    return attempt


def requested_workflow_index_filter(args: argparse.Namespace) -> set[int]:
    indices: set[int] = set()
    for spec in args.workflow_index or []:
        for chunk in str(spec).split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_text, end_text = chunk.split("-", 1)
                if start_text.strip().isdigit() and end_text.strip().isdigit():
                    start = int(start_text)
                    end = int(end_text)
                    indices.update(range(min(start, end), max(start, end) + 1))
                continue
            if chunk.isdigit():
                indices.add(int(chunk))
    return indices


def retry_attempt_dir(benchmark_dir: Path, planned: dict[str, Any], attempt_index: int) -> Path:
    combo_key = str(planned.get("combo_key") or planned.get("algorithm") or "retry")
    digest = hashlib.sha1(combo_key.encode("utf-8")).hexdigest()[:10]
    return benchmark_dir / "_retry_tmp" / f"a{attempt_index:02d}_{digest}"


def merge_successful_retry_results(run_dir: Path, retry_dir: Path, max_parallel_nodes: int) -> dict[str, Any]:
    output_dir = run_dir / "execution"
    retry_output_dir = retry_dir / "execution"
    manifest_path = output_dir / "EXECUTION_MANIFEST.json"
    retry_manifest_path = retry_output_dir / "EXECUTION_MANIFEST.json"
    manifest = load_json(manifest_path, {})
    retry_manifest = load_json(retry_manifest_path, {})
    runs = manifest.get("runs") or []
    retry_runs = retry_manifest.get("runs") or []
    run_by_index = {int(record.get("workflow_index")): record for record in runs if record.get("workflow_index") is not None}
    replaced = []
    retry_log_entries = []

    for retry_record in retry_runs:
        workflow_index = retry_record.get("workflow_index")
        if workflow_index is None or retry_record.get("status") != "completed":
            continue
        workflow_index = int(workflow_index)
        original_record = run_by_index.get(workflow_index)
        if not original_record:
            continue
        copied = replace_single_workflow_result(
            run_dir,
            retry_dir,
            original_record,
            retry_record,
            max_parallel_nodes,
        )
        original_record.update(copied["record_updates"])
        replaced.append(workflow_index)
        retry_log_entries.append(
            {
                "logged_at_utc": utc_now(),
                "workflow_index": workflow_index,
                "workflow_id": original_record.get("workflow_id"),
                "status": "retry_replaced_success",
                "retry_dir": str(retry_dir),
                "max_parallel_nodes": max_parallel_nodes,
                "result_json": original_record.get("result_json"),
            }
        )

    if replaced:
        update_resource_by_workflow(output_dir, retry_output_dir, replaced)
        refresh_resource_summary_from_workflows(output_dir)
        manifest["runs"] = runs
        manifest["retry_replacements"] = list(manifest.get("retry_replacements") or []) + retry_log_entries
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        append_jsonl(output_dir / "RETRY_LOG.jsonl", retry_log_entries)
        regenerate_execution_dashboard(output_dir)

    return {
        "replaced_workflow_indices": replaced,
        "attempted_workflow_indices": [record.get("workflow_index") for record in retry_runs],
        "retry_dir": str(retry_dir),
    }


def replace_single_workflow_result(
    run_dir: Path,
    retry_dir: Path,
    original_record: dict[str, Any],
    retry_record: dict[str, Any],
    max_parallel_nodes: int,
) -> dict[str, Any]:
    original_result_path = Path(str(original_record.get("result_json") or ""))
    retry_result_path = Path(str(retry_record.get("result_json") or ""))
    if not retry_result_path.exists():
        raise FileNotFoundError(f"Retry result does not exist: {retry_result_path}")

    state_dst = original_state_path_for_record(run_dir, original_record)
    result = load_json(retry_result_path, {})
    state_src = nested_get(result, ["execution_control", "state"])
    result.setdefault("retry_metadata", {})
    result["retry_metadata"].update(
        {
            "replaced_failed_result": True,
            "retry_dir": str(retry_dir),
            "merged_at_utc": utc_now(),
            "max_parallel_nodes": max_parallel_nodes,
        }
    )
    execution_control = result.get("execution_control")
    if isinstance(execution_control, dict):
        execution_control["deployment_json"] = str(run_dir / "deployment" / "workflow_deployment.json")
        if original_record.get("runtime_config"):
            execution_control["runtime_config"] = original_record.get("runtime_config")
        if original_record.get("result_json"):
            execution_control["merged_result_json"] = original_record.get("result_json")

    original_result_path.parent.mkdir(parents=True, exist_ok=True)
    original_result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    copy_optional_path(retry_record.get("visualization"), original_record.get("visualization"))
    copy_optional_path(retry_runtime_config_path(retry_dir, retry_record), original_record.get("runtime_config"))
    copy_optional_path(state_src, state_dst)

    record_updates = {
        "status": "completed",
        "total_wall_ms": retry_record.get("total_wall_ms"),
        "started_epoch_s": retry_record.get("started_epoch_s"),
        "finished_epoch_s": retry_record.get("finished_epoch_s"),
        "resource_summary": retry_record.get("resource_summary"),
        "retry_replaced_failed_result": True,
        "retry_replaced_at_utc": utc_now(),
        "retry_dir": str(retry_dir),
        "retry_max_parallel_nodes": max_parallel_nodes,
    }
    return {"record_updates": record_updates}


def retry_runtime_config_path(retry_dir: Path, retry_record: dict[str, Any]) -> str:
    raw = retry_record.get("runtime_config")
    if raw:
        return str(raw)
    return str(retry_dir / "deployment" / "runtime_configs")


def original_state_path_for_record(run_dir: Path, record: dict[str, Any]) -> str:
    result = load_json(Path(str(record.get("result_json") or "")), {})
    state = nested_get(result, ["execution_control", "state"])
    if state:
        return str(state)
    result_path = Path(str(record.get("result_json") or ""))
    return str(run_dir / "execution" / "tool_state" / result_path.name.replace(".result.json", ".json"))


def update_resource_by_workflow(output_dir: Path, retry_output_dir: Path, replaced_indices: list[int]) -> None:
    resource_path = output_dir / "resources" / "resource_by_workflow.json"
    retry_resource_path = retry_output_dir / "resources" / "resource_by_workflow.json"
    resources = load_json(resource_path, {})
    retry_resources = load_json(retry_resource_path, {})
    changed = False
    for index in replaced_indices:
        key = str(index)
        if isinstance(retry_resources.get(key), dict):
            resources[key] = retry_resources[key]
            changed = True
    if changed:
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        resource_path.write_text(json.dumps(resources, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_resource_summary_from_workflows(output_dir: Path) -> None:
    summary_path = output_dir / "resources" / "resource_summary.json"
    by_workflow_path = output_dir / "resources" / "resource_by_workflow.json"
    summary = load_json(summary_path, {})
    by_workflow = load_json(by_workflow_path, {})
    resources = [
        resource
        for _, resource in sorted(by_workflow.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 999999)
        if isinstance(resource, dict)
    ]
    if not resources:
        return
    refreshed = dict(summary)
    refreshed["sample_count"] = sum(int(resource.get("sample_count") or 0) for resource in resources)
    starts = [resource.get("started_at_utc") for resource in resources if resource.get("started_at_utc")]
    finishes = [resource.get("finished_at_utc") for resource in resources if resource.get("finished_at_utc")]
    if starts:
        refreshed["started_at_utc"] = min(starts)
    if finishes:
        refreshed["finished_at_utc"] = max(finishes)
    refreshed["duration_s"] = round(sum(float(resource.get("duration_s") or 0.0) for resource in resources), 3)
    refreshed["metrics"] = merge_resource_metrics(resources)
    refreshed["merged_retry_resource_summary"] = True
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(refreshed, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_resource_metrics(resources: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            metric_name
            for resource in resources
            for metric_name in ((resource.get("metrics") or {}).keys())
        }
    )
    merged: dict[str, Any] = {}
    for metric_name in metric_names:
        stats = [resource.get("metrics", {}).get(metric_name) or {} for resource in resources]
        total_count = sum(int(stat.get("count") or 0) for stat in stats)
        if total_count <= 0:
            continue
        avg_numerator = sum(float(stat.get("avg") or 0.0) * int(stat.get("count") or 0) for stat in stats)
        mins = [float(stat["min"]) for stat in stats if stat.get("min") is not None]
        maxs = [float(stat["max"]) for stat in stats if stat.get("max") is not None]
        lasts = [stat.get("last") for stat in stats if stat.get("last") is not None]
        merged[metric_name] = {
            "count": total_count,
            "avg": avg_numerator / total_count,
            "min": min(mins) if mins else None,
            "max": max(maxs) if maxs else None,
            "last": lasts[-1] if lasts else None,
        }
    return merged


def regenerate_execution_dashboard(output_dir: Path) -> None:
    try:
        from visualizations.visualize_execution_performance import load_dashboard_payload, render_dashboard

        payload = load_dashboard_payload(output_dir)
        (output_dir / "performance_overview.html").write_text(render_dashboard(output_dir, payload), encoding="utf-8")
    except Exception as exc:
        print(f"[bench] execution dashboard regeneration skipped: {exc}", flush=True)


def copy_optional_path(source: Any, destination: Any) -> None:
    if not source or not destination:
        return
    src = Path(str(source))
    dst = Path(str(destination))
    if not src.exists() or src.is_dir():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def append_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def apply_attempt_to_record(record: dict[str, Any], attempt: dict[str, Any]) -> None:
    record["status"] = attempt["status"]
    record["commands"] = attempt.get("commands", [])
    record["cleanup"] = attempt.get("cleanup", [])
    record["errors"] = attempt.get("errors", [])
    record["max_parallel_nodes"] = attempt.get("max_parallel_nodes")
    record["final_attempt_index"] = attempt.get("attempt_index")
    if attempt.get("is_retry"):
        record["retry_count"] = int(record.get("retry_count") or 0) + 1
    else:
        record["retry_count"] = max(0, int(attempt.get("attempt_index") or 0))
    record["retryable"] = attempt.get("retryable", False)
    if attempt.get("retryability"):
        record["retryability"] = attempt["retryability"]
    if attempt.get("execution_failures"):
        record["execution_failures"] = attempt["execution_failures"]
    record["executed"] = bool(attempt.get("commands"))


def run_command(cmd: list[str]) -> dict[str, Any]:
    print("[bench] " + " ".join(quote_part(part) for part in cmd), flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    record = {
        "command": cmd,
        "returncode": result.returncode,
        "duration_s": round(time.time() - started, 3),
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": utc_now(),
    }
    print(f"[bench] returncode={result.returncode} duration_s={record['duration_s']}", flush=True)
    return record


def stop_stale_phone_runtimes(args: argparse.Namespace) -> dict[str, Any]:
    cleanup = {
        "started_at_utc": utc_now(),
        "adb": "",
        "commands": [],
        "status": "skipped",
    }
    adb = best_effort_adb_path(args.adb)
    if not adb:
        cleanup["reason"] = "adb_not_found"
        return cleanup
    cleanup["adb"] = str(adb)
    base = [str(adb)]
    if args.adb_serial:
        base.extend(["-s", args.adb_serial])
    commands = [
        [*base, "shell", "sh", "-c", "pkill -f llama-server >/dev/null 2>&1 || true; pkill -f llama-cli >/dev/null 2>&1 || true"],
        [*base, "forward", "--remove-all"],
    ]
    cleanup["status"] = "attempted"
    for cmd in commands:
        started = time.time()
        try:
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
            cleanup["commands"].append(
                {
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:],
                    "duration_s": round(time.time() - started, 3),
                }
            )
        except Exception as exc:
            cleanup["commands"].append(
                {
                    "command": cmd,
                    "error": str(exc),
                    "duration_s": round(time.time() - started, 3),
                }
            )
    cleanup["finished_at_utc"] = utc_now()
    return cleanup


def execution_failure_summary(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "execution" / "EXECUTION_MANIFEST.json"
    manifest = load_json(manifest_path, {})
    if not manifest:
        return {
            "workflow_count": 0,
            "failed_workflows": 1,
            "failed_workflow_ids": [],
            "failed_workflow_indices": [],
            "error_log": None,
            "retryability": {
                "retryable": True,
                "retryable_reasons": ["missing_execution_manifest"],
                "non_retryable_reasons": [],
            },
        }
    runs = manifest.get("runs") or []
    failed = [item for item in runs if item.get("status") != "completed"]
    error_log_path = run_dir / "execution" / "ERROR_LOG.jsonl"
    error_entries = read_error_log_entries(error_log_path)
    retryability = classify_retryability(error_entries, failed)
    return {
        "workflow_count": len(runs),
        "failed_workflows": len(failed),
        "failed_workflow_ids": [item.get("workflow_id") for item in failed],
        "failed_workflow_indices": [item.get("workflow_index") for item in failed],
        "error_log": str(error_log_path) if error_log_path.exists() else None,
        "retryability": retryability,
    }


def read_error_log_entries(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                entries.append({"raw": line})
            if len(entries) >= limit:
                break
    except OSError:
        return []
    return entries


def classify_retryability(error_entries: list[dict[str, Any]], failed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not failed_runs:
        return {"retryable": False, "retryable_reasons": [], "non_retryable_reasons": []}
    if not error_entries:
        return {
            "retryable": True,
            "retryable_reasons": ["failed_workflows_without_error_log"],
            "non_retryable_reasons": [],
        }

    retryable_reasons: list[str] = []
    non_retryable_reasons: list[str] = []
    runtime_patterns = {
        "llama_cli",
        "llama-server",
        "llama server",
        "ggml",
        "hexagon",
        "dspqueue",
        "returncode\": 134",
        "returncode': 134",
        "aborted",
        "timeout",
        "timed out",
        "adb",
        "device offline",
        "connection refused",
        "connection reset",
        "subprocess",
    }
    tool_patterns = {
        "json_cli",
        "no user found",
        "no order found",
        "no reservation found",
        "unknown tool",
        "missing required",
        "invalid argument",
        "invalid parameters",
        "not found for",
    }

    for entry in error_entries:
        text = json.dumps(entry, ensure_ascii=False).lower()
        workflow = entry.get("workflow_id") or entry.get("workflow_index") or "workflow"
        runtime_hit = next((pattern for pattern in runtime_patterns if pattern in text), None)
        if runtime_hit:
            retryable_reasons.append(f"{workflow}: {runtime_hit}")
            continue
        tool_hit = next((pattern for pattern in tool_patterns if pattern in text), None)
        if tool_hit:
            non_retryable_reasons.append(f"{workflow}: {tool_hit}")
            continue
        retryable_reasons.append(f"{workflow}: unknown_failure")

    return {
        "retryable": bool(retryable_reasons),
        "retryable_reasons": retryable_reasons,
        "non_retryable_reasons": non_retryable_reasons,
    }


def quote_part(part: object) -> str:
    text = str(part)
    if not text or any(char.isspace() for char in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def records_for_manifest(
    planned_runs: list[dict[str, Any]],
    record_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record_by_key[planned["combo_key"]]
        for planned in planned_runs
        if planned["combo_key"] in record_by_key
    ]


def write_state(
    manifest_path: Path,
    progress_path: Path,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    record_by_key: dict[str, dict[str, Any]],
    *,
    current: dict[str, Any] | None = None,
) -> None:
    manifest["updated_at_utc"] = utc_now()
    manifest["records"] = records_for_manifest(planned_runs, record_by_key)
    write_json(manifest_path, manifest)
    write_json(progress_path, progress_payload(manifest, planned_runs, record_by_key, current=current))


def progress_payload(
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    record_by_key: dict[str, dict[str, Any]],
    *,
    current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    counts: dict[str, int] = {"pending": 0}
    for sequence, planned in enumerate(planned_runs, start=1):
        record = record_by_key.get(planned["combo_key"]) or {}
        status = str(record.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
        rows.append(
            {
                "sequence": sequence,
                "combo_key": planned["combo_key"],
                "model_name": planned["model"]["model_name"],
                "model_path": planned["model"]["model_path"],
                "algorithm": planned["algorithm"],
                "run_dir": str(planned["run_dir"]),
                "status": status,
                "started_at_utc": record.get("started_at_utc"),
                "finished_at_utc": record.get("finished_at_utc"),
                "retry_count": record.get("retry_count", 0),
                "max_parallel_nodes": record.get("max_parallel_nodes"),
                "retryability": record.get("retryability"),
                "record_json": str(Path(planned["run_dir"]) / "benchmark_run_record.json"),
            }
        )
    completed = counts.get("completed", 0)
    terminal_non_retryable = counts.get("execution_partial_non_retryable", 0)
    failed = counts.get("failed", 0) + counts.get("retry_exhausted", 0)
    pending = counts.get("pending", 0)
    return {
        "entrypoint": ENTRYPOINT_NAME,
        "benchmark_dir": manifest.get("benchmark_dir"),
        "source_run_dir": manifest.get("source_run_dir"),
        "updated_at_utc": utc_now(),
        "resume": manifest.get("resume"),
        "retry_failed": manifest.get("retry_failed"),
        "retry_attempts": manifest.get("retry_attempts"),
        "retry_max_parallel_nodes": manifest.get("retry_max_parallel_nodes"),
        "retry_interval_s": manifest.get("retry_interval_s"),
        "inter_run_interval_s": manifest.get("inter_run_interval_s"),
        "total_runs": len(planned_runs),
        "completed_count": completed,
        "terminal_non_retryable_count": terminal_non_retryable,
        "failed_count": failed,
        "pending_count": pending,
        "status_counts": counts,
        "current": current or {},
        "runs": rows,
    }


def cooldown_between_runs(
    interval_s: float,
    manifest_path: Path,
    progress_path: Path,
    manifest: dict[str, Any],
    planned_runs: list[dict[str, Any]],
    record_by_key: dict[str, dict[str, Any]],
    completed_sequence_index: int,
    *,
    label: str = "next combination",
) -> None:
    if interval_s <= 0 or completed_sequence_index >= len(planned_runs):
        return
    deadline = time.time() + interval_s
    print(f"[bench] cooling down for {interval_s:.1f}s before {label}", flush=True)
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        write_state(
            manifest_path,
            progress_path,
            manifest,
            planned_runs,
            record_by_key,
            current={
                "status": "cooldown",
                "cooldown_label": label,
                "cooldown_seconds_remaining": round(max(0.0, remaining), 1),
                "completed_sequence_index": completed_sequence_index,
            },
        )
        sleep_s = min(30.0, remaining)
        time.sleep(max(0.0, sleep_s))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
