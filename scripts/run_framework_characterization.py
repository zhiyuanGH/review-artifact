"""Run MOBench workflow characterization across phones, frameworks, and models."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from MOBench.deployment_planning.core import best_effort_adb_path
except ImportError:
    from deployment_planning.core import best_effort_adb_path


DEFAULT_MATRIX = REPO_ROOT / "configs" / "characterization.example.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / f"framework_characterization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DEFAULT_GGUF_DIR = "/data/local/tmp/gguf"
ENTRYPOINT_NAME = "framework-characterization"
ENTRYPOINT_STAGES = ["deployment", "execution", "summary"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adb = best_effort_adb_path(args.adb)
    if adb is None:
        raise FileNotFoundError("adb not found. Pass --adb or set ADB.")
    devices = parse_csv_arg(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one adb serial.")
    workflow_indices = parse_index_args(args.workflow_index)
    if not workflow_indices:
        raise ValueError("At least one --workflow-index is required.")

    source_run_dir = args.source_run_dir.resolve()
    source_planning_dir = source_run_dir / "planning"
    if not (source_planning_dir / "workflow.json").exists():
        raise FileNotFoundError(f"Source run is missing planning/workflow.json: {source_planning_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = load_json(args.matrix.resolve())
    combos = planned_combos(matrix, args.gguf_dir)
    if not combos:
        raise ValueError(f"No framework/model entries found in matrix: {args.matrix}")

    manifest = {
        "entrypoint": ENTRYPOINT_NAME,
        "pipeline_stages": ENTRYPOINT_STAGES,
        "created_at_utc": utc_now(),
        "source_run_dir": str(source_run_dir),
        "source_planning_dir": str(source_planning_dir),
        "output_dir": str(output_dir),
        "matrix": str(args.matrix.resolve()),
        "devices": devices,
        "workflow_indices": workflow_indices,
        "parallel_devices": args.parallel_devices,
        "max_parallel_nodes": args.max_parallel_nodes,
        "llm_max_tokens": args.llm_max_tokens,
        "combo_count_per_device": len(combos),
        "dry_run": args.dry_run,
        "records": [],
    }
    write_json(output_dir / "framework_characterization_manifest.json", manifest)

    print(f"[char] source planning: {source_planning_dir}", flush=True)
    print(f"[char] output dir:      {output_dir}", flush=True)
    print(f"[char] devices:         {', '.join(devices)}", flush=True)
    print(f"[char] workflows:       {', '.join(str(item) for item in workflow_indices)}", flush=True)
    print(f"[char] combos/device:   {len(combos)}", flush=True)
    print(f"[char] parallel devices:{args.parallel_devices}", flush=True)

    rows: list[dict[str, Any]] = []
    status_code = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.parallel_devices, len(devices)))) as executor:
        futures = {
            executor.submit(run_device_matrix, device, combos, source_planning_dir, output_dir, adb, workflow_indices, args): device
            for device in devices
        }
        for future in as_completed(futures):
            device = futures[future]
            try:
                device_rows = future.result()
                rows.extend(device_rows)
                if any(row.get("status") != "completed" for row in device_rows):
                    status_code = 2
            except Exception as exc:
                status_code = 2
                rows.append(
                    {
                        "device": device,
                        "framework_id": "",
                        "model_id": "",
                        "workflow_index": "",
                        "status": "runner_error",
                        "error": str(exc),
                    }
                )

    rows = sorted(rows, key=lambda row: (str(row.get("device")), str(row.get("framework_id")), str(row.get("model_id")), str(row.get("workflow_index"))))
    manifest["finished_at_utc"] = utc_now()
    manifest["records"] = rows
    write_json(output_dir / "framework_characterization_manifest.json", manifest)
    write_json(output_dir / "summary.json", {"created_at_utc": utc_now(), "rows": rows})
    write_csv(output_dir / "summary.csv", rows)
    write_markdown(output_dir / "summary.md", rows, manifest)
    print(f"[char] wrote summary: {output_dir / 'summary.md'}", flush=True)
    return status_code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", required=True, help="Comma-separated adb serials.")
    parser.add_argument("--parallel-devices", type=int, default=2)
    parser.add_argument("--workflow-index", action="append", default=[])
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR)
    parser.add_argument("--adb", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--max-parallel-nodes", type=int, default=3)
    parser.add_argument("--llm-max-tokens", type=int, default=64)
    parser.add_argument("--ort-genai-prompt-char-limit", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resource-monitor", choices=["perfetto", "off"], default="perfetto")
    parser.add_argument("--resource-sample-interval", type=float, default=1.0)
    parser.add_argument("--perfetto-buffer-mb", type=int, default=128)
    parser.add_argument("--perfetto-tools-dir", type=Path, default=REPO_ROOT / ".wfgen_perfetto")
    parser.add_argument("--perfetto-required", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-deploy-visualize", action="store_true")
    parser.add_argument("--no-execute-visualize", action="store_true")
    return parser.parse_args(argv)


def run_device_matrix(
    device: str,
    combos: list[dict[str, Any]],
    source_planning_dir: Path,
    output_dir: Path,
    adb: Path,
    workflow_indices: list[int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, combo in enumerate(combos, start=1):
        print(
            f"[char:{device}] {index}/{len(combos)} {combo['framework_id']} :: {combo['model_id']}",
            flush=True,
        )
        run_dir = output_dir / safe_name(device) / safe_name(combo["framework_id"]) / safe_name(combo["model_id"])
        record = run_combo(device, combo, source_planning_dir, run_dir, adb, workflow_indices, args)
        rows.extend(record.get("rows") or [])
        write_json(run_dir / "framework_characterization_record.json", record)
    return rows


def run_combo(
    device: str,
    combo: dict[str, Any],
    source_planning_dir: Path,
    run_dir: Path,
    adb: Path,
    workflow_indices: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.time()
    prepare_run_dir(run_dir, source_planning_dir, overwrite=args.overwrite, resume=args.resume)
    python = str(args.python.resolve()) if args.python else sys.executable

    deploy_cmd = build_deploy_command(python, run_dir, device, combo, adb, args)
    execute_cmd = build_execute_command(python, run_dir, device, adb, workflow_indices, args)
    deploy_result = run_logged_command(deploy_cmd, run_dir / "logs", "deploy")
    execute_result = None
    if deploy_result["returncode"] == 0:
        execute_result = run_logged_command(execute_cmd, run_dir / "logs", "execute")
    else:
        execute_result = {"returncode": None, "skipped": True, "reason": "deploy_failed"}

    rows = summarize_run(run_dir, device, combo, deploy_result, execute_result)
    for row in rows:
        row["max_parallel_nodes"] = args.max_parallel_nodes
        row["llm_max_tokens"] = args.llm_max_tokens
    finished = time.time()
    return {
        "device": device,
        "framework": combo["framework_id"],
        "model": combo["model_id"],
        "max_parallel_nodes": args.max_parallel_nodes,
        "llm_max_tokens": args.llm_max_tokens,
        "run_dir": str(run_dir),
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_ms": (finished - started) * 1000.0,
        "deploy": deploy_result,
        "execute": execute_result,
        "rows": rows,
    }


def prepare_run_dir(run_dir: Path, source_planning_dir: Path, *, overwrite: bool, resume: bool) -> None:
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        elif not resume:
            raise FileExistsError(f"Run directory already exists. Use --resume or --overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    planning_dir = run_dir / "planning"
    if not planning_dir.exists():
        shutil.copytree(source_planning_dir, planning_dir)


def build_deploy_command(
    python: str,
    run_dir: Path,
    device: str,
    combo: dict[str, Any],
    adb: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        str(REPO_ROOT / "wfgen_deploy.py"),
        "--run-dir",
        str(run_dir),
        "--deployment-strategy",
        combo["strategy"],
        "--max-parallel-nodes",
        str(args.max_parallel_nodes),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--llm-timeout",
        str(args.timeout),
        "--llama-cli-timeout",
        str(args.timeout),
        "--llama-cli-warmup-runs",
        str(args.warmup_runs),
        "--adb",
        str(adb),
        "--adb-serial",
        device,
    ]
    runtime = combo.get("runtime")
    if runtime == "llama_cli":
        command.extend(["--llama-model-path", combo["model_path"]])
    elif runtime == "ort_genai_cli":
        command.extend(
            [
                "--ort-genai-dir",
                combo.get("runner_dir") or "/data/local/tmp/oga_generate",
                "--ort-genai-model-path",
                combo["model_path"],
                "--ort-genai-jar",
                combo.get("jar") or "oga-generate.jar",
                "--ort-genai-timeout",
                str(args.timeout),
                "--ort-genai-prompt-char-limit",
                str(args.ort_genai_prompt_char_limit),
            ]
        )
    elif runtime == "genie_cli":
        command.extend(
            [
                "--genie-bundle-dir",
                combo["model_path"],
                "--genie-config",
                combo.get("config") or "auto",
                "--genie-backend",
                combo.get("backend") or "npu",
                "--genie-timeout",
                str(args.timeout),
            ]
        )
    if args.no_deploy_visualize:
        command.append("--no-visualize")
    return command


def build_execute_command(
    python: str,
    run_dir: Path,
    device: str,
    adb: Path,
    workflow_indices: list[int],
    args: argparse.Namespace,
) -> list[str]:
    command = [
        python,
        str(REPO_ROOT / "wfgen_execute.py"),
        "--run-dir",
        str(run_dir),
        "--adb",
        str(adb),
        "--adb-serial",
        device,
        "--resource-monitor",
        args.resource_monitor,
        "--resource-sample-interval",
        str(args.resource_sample_interval),
        "--perfetto-buffer-mb",
        str(args.perfetto_buffer_mb),
        "--perfetto-tools-dir",
        str(args.perfetto_tools_dir),
    ]
    for workflow_index in workflow_indices:
        command.extend(["--workflow-index", str(workflow_index)])
    if args.perfetto_required:
        command.append("--perfetto-required")
    if args.dry_run:
        command.append("--dry-run")
    if args.no_execute_visualize:
        command.append("--no-visualize")
    return command


def run_logged_command(command: list[str], log_dir: Path, label: str) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True)
    stdout_path = log_dir / f"{label}.stdout.txt"
    stderr_path = log_dir / f"{label}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(proc.stderr, encoding="utf-8", errors="replace")
    return {
        "label": label,
        "command": command,
        "returncode": proc.returncode,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": utc_now(),
        "wall_ms": (time.time() - started) * 1000.0,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def summarize_run(
    run_dir: Path,
    device: str,
    combo: dict[str, Any],
    deploy_result: dict[str, Any],
    execute_result: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = load_json(run_dir / "execution" / "EXECUTION_MANIFEST.json", {})
    runs = manifest.get("runs") or []
    if not runs:
        return [
            base_row(device, combo)
            | {
                "workflow_index": "",
                "workflow_id": "",
                "status": "deploy_failed" if deploy_result.get("returncode") else "execution_missing",
                "deploy_returncode": deploy_result.get("returncode"),
                "execute_returncode": execute_result.get("returncode"),
                "run_dir": str(run_dir),
            }
        ]

    rows = []
    for run in runs:
        result = load_json(Path(run.get("result_json", "")), {}) if run.get("result_json") else {}
        llm_stats = summarize_llm_steps(result)
        resource = run.get("resource_summary") or {}
        rows.append(
            base_row(device, combo)
            | {
                "workflow_index": run.get("workflow_index"),
                "workflow_id": run.get("workflow_id"),
                "status": run.get("status"),
                "failure_reason": first_failure_reason(result),
                "wall_ms": run.get("total_wall_ms"),
                "deploy_returncode": deploy_result.get("returncode"),
                "execute_returncode": execute_result.get("returncode"),
                "run_dir": str(run_dir),
                "result_json": run.get("result_json"),
                "llm_node_count": llm_stats["llm_node_count"],
                "llm_failed_count": llm_stats["llm_failed_count"],
                "placements": llm_stats["placements"],
                "llm_inference_wall_ms_sum": llm_stats["inference_wall_ms_sum"],
                "ort_model_load_ms_sum": llm_stats["ort_model_load_ms_sum"],
                "ort_generation_ms_sum": llm_stats["ort_generation_ms_sum"],
                "genie_query_us_sum": llm_stats["genie_query_us_sum"],
                "genie_decode_tps_avg": llm_stats["genie_decode_tps_avg"],
                "resource_samples": resource.get("sample_count", ""),
                "cpu_avg_percent": resource.get("cpu_avg_percent", ""),
                "cpu_max_percent": resource.get("cpu_max_percent", ""),
                "memory_max_mb": resource.get("memory_max_mb", ""),
                "gpu_busy_avg_percent": resource.get("gpu_busy_avg_percent", ""),
                "gpu_busy_max_percent": resource.get("gpu_busy_max_percent", ""),
                "thermal_max_c": resource.get("thermal_max_c", ""),
            }
        )
    return rows


def base_row(device: str, combo: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": device,
        "framework_id": combo["framework_id"],
        "framework_label": combo.get("framework_label", ""),
        "runtime": combo.get("runtime", ""),
        "strategy": combo.get("strategy", ""),
        "supported_units": ",".join(combo.get("supported_units") or []),
        "model_id": combo["model_id"],
        "model_name": combo.get("model_name", ""),
        "model_path": combo.get("model_path", ""),
    }


def summarize_llm_steps(result: dict[str, Any]) -> dict[str, Any]:
    deployments = {
        item.get("node_id"): item
        for item in (((result.get("execution_control") or {}).get("node_deployments")) or [])
        if item.get("kind") == "llm"
    }
    steps = [step for step in result.get("steps", []) if step.get("kind") == "llm"]
    placement_counts: dict[str, int] = {}
    inference_wall = 0.0
    ort_model_load = 0.0
    ort_generation = 0.0
    genie_query = 0.0
    genie_decode_rates = []
    for step in steps:
        deployment = deployments.get(step.get("node_id"), {})
        device_class = deployment.get("device_class") or deployment.get("device") or "unknown"
        placement_counts[str(device_class)] = placement_counts.get(str(device_class), 0) + 1
        raw = step.get("raw") or {}
        inference_wall += numeric(raw.get("inference_wall_ms"))
        ort_model_load += numeric(raw.get("model_load_ms"))
        ort_generation += numeric(raw.get("generation_ms"))
        profile = raw.get("profile_summary") or {}
        genie_query += numeric(profile.get("query_us"))
        if profile.get("token_generation_rate") is not None:
            genie_decode_rates.append(numeric(profile.get("token_generation_rate")))
    return {
        "llm_node_count": len(steps),
        "llm_failed_count": sum(1 for step in steps if not step.get("success")),
        "placements": ",".join(f"{key}:{placement_counts[key]}" for key in sorted(placement_counts)),
        "inference_wall_ms_sum": round(inference_wall, 3),
        "ort_model_load_ms_sum": round(ort_model_load, 3),
        "ort_generation_ms_sum": round(ort_generation, 3),
        "genie_query_us_sum": round(genie_query, 3),
        "genie_decode_tps_avg": round(sum(genie_decode_rates) / len(genie_decode_rates), 3) if genie_decode_rates else "",
    }


def planned_combos(matrix: dict[str, Any], gguf_dir: str) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for framework in matrix.get("frameworks") or []:
        for model in framework.get("models") or []:
            model_path = str(model.get("path") or "")
            if framework.get("runtime") == "llama_cli" and model_path and not model_path.startswith("/"):
                model_path = f"{gguf_dir.rstrip('/')}/{model_path}"
            combos.append(
                {
                    "framework_id": str(framework.get("id") or "framework"),
                    "framework_label": str(framework.get("label") or framework.get("id") or ""),
                    "runtime": str(framework.get("runtime") or ""),
                    "strategy": str(framework.get("strategy") or ""),
                    "supported_units": list(framework.get("supported_units") or []),
                    "runner_dir": model.get("runner_dir") or framework.get("runner_dir"),
                    "jar": model.get("jar") or framework.get("jar"),
                    "config": model.get("config") or framework.get("config"),
                    "backend": model.get("backend") or framework.get("backend"),
                    "model_id": str(model.get("id") or safe_name(model_path)),
                    "model_name": str(model.get("name") or model.get("id") or model_path),
                    "model_path": model_path,
                }
            )
    return combos


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Framework Characterization Summary",
        "",
        f"Created UTC: `{manifest['created_at_utc']}`",
        f"Source run: `{manifest['source_run_dir']}`",
        f"Workflows: `{', '.join(str(item) for item in manifest['workflow_indices'])}`",
        f"Devices: `{', '.join(manifest['devices'])}`",
        "",
        "| Device | Framework | Model | Workflow | Parallel | Status | Wall ms | Placements | LLM nodes | Failed LLM | Failure | Run dir |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("device", "")),
                    str(row.get("framework_id", "")),
                    str(row.get("model_id", "")),
                    str(row.get("workflow_index", "")),
                    str(row.get("max_parallel_nodes", "")),
                    str(row.get("status", "")),
                    format_number(row.get("wall_ms")),
                    str(row.get("placements", "")),
                    str(row.get("llm_node_count", "")),
                    str(row.get("llm_failed_count", "")),
                    str(row.get("failure_reason", "")),
                    str(row.get("run_dir", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_index_args(values: list[str]) -> list[int]:
    out: list[int] = []
    for raw in values:
        for part in raw.split(","):
            item = part.strip()
            if not item:
                continue
            if "-" in item:
                start, end = item.split("-", 1)
                out.extend(range(int(start), int(end) + 1))
            else:
                out.append(int(item))
    return sorted(dict.fromkeys(out))


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def first_failure_reason(result: dict[str, Any], limit: int = 240) -> str:
    if result.get("status") == "completed":
        return ""
    for step in result.get("steps", []):
        if step.get("success", True):
            continue
        raw = step.get("raw") or {}
        reason = None
        stderr = raw.get("stderr")
        if isinstance(stderr, str) and stderr.strip():
            reason = stderr
        if reason is None:
            reason = raw.get("error")
        if isinstance(reason, str) and reason.strip().startswith("{"):
            parsed = parse_json_object(reason)
            if parsed:
                if parsed.get("stderr"):
                    reason = parsed["stderr"]
                elif parsed.get("returncode") not in {None, 0}:
                    stdout = str(parsed.get("stdout") or "").replace("\n", " ").strip()
                    reason = f"returncode={parsed.get('returncode')}; stdout={stdout}"
                else:
                    reason = parsed.get("error") or parsed.get("stdout") or reason
        if not reason and isinstance(raw.get("response"), dict):
            reason = raw["response"].get("error")
        if not reason:
            reason = raw.get("stderr") or raw.get("stdout")
        if not reason and raw.get("returncode") not in {None, 0}:
            reason = f"returncode={raw.get('returncode')}"
        text = str(reason or "workflow step failed").replace("\n", " ").strip()
        return (text[: limit - 3] + "...") if len(text) > limit else text
    return "workflow failed before producing step details"


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value))
    return cleaned.strip("._") or "item"


def numeric(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def format_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
