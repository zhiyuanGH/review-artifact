"""Visualize MOBench deployment benchmark results across algorithms."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_dir = resolve_benchmark_dir(args.benchmark_dir)
    paths = write_benchmark_visualizations(benchmark_dir)
    print(f"wfgen: wrote deployment benchmark JSON to {paths['json']}")
    print(f"wfgen: wrote deployment benchmark Markdown to {paths['markdown']}")
    print(f"wfgen: wrote deployment benchmark HTML to {paths['html']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MOBench deployment benchmark runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("benchmark_dir", nargs="?", type=Path, default=None, help="Benchmark root from run_deployment_benchmarks.py.")
    return parser.parse_args(argv)


def write_benchmark_visualizations(benchmark_dir: Path) -> dict[str, Path]:
    benchmark_dir = benchmark_dir.resolve()
    summary = load_benchmark_summary(benchmark_dir)
    json_path = benchmark_dir / "deployment_benchmark_summary.json"
    md_path = benchmark_dir / "deployment_benchmark_summary.md"
    html_path = benchmark_dir / "deployment_benchmark_overview.html"
    write_json(json_path, summary)
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    html_path.write_text(render_html(summary), encoding="utf-8")
    return {"json": json_path, "markdown": md_path, "html": html_path}


def load_benchmark_summary(benchmark_dir: Path) -> dict[str, Any]:
    manifest = load_json(benchmark_dir / "deployment_benchmark_manifest.json", {})
    run_dirs = discover_run_dirs(benchmark_dir, manifest)
    algorithms = [summarize_algorithm(path) for path in run_dirs]
    ranking = [
        {
            "rank": rank + 1,
            "algorithm": item["algorithm"],
            "model_name": item.get("model_name"),
            "run_label": item.get("run_label"),
            "total_wall_ms": item["total_wall_ms"],
            "completed_workflows": item["completed_workflows"],
        }
        for rank, item in enumerate(
            sorted(
                [item for item in algorithms if item.get("included_workflow_count", item.get("workflow_count", 0)) > 0],
                key=lambda item: (numeric(item.get("total_wall_ms")), item.get("run_label") or item["algorithm"]),
            )
        )
    ]
    return {
        "created_at_utc": utc_now(),
        "benchmark_dir": str(benchmark_dir),
        "source_run_dir": manifest.get("source_run_dir"),
        "source_planning_dir": manifest.get("source_planning_dir"),
        "manifest": manifest,
        "algorithms": algorithms,
        "models": model_summary(algorithms, manifest),
        "ranking": ranking,
    }


def discover_run_dirs(benchmark_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    ordered = []
    seen = set()
    for record in manifest.get("records") or []:
        run_dir = Path(str(record.get("run_dir") or ""))
        resolved = run_dir.resolve()
        if run_dir.is_dir() and resolved not in seen:
            ordered.append(resolved)
            seen.add(resolved)
    for algorithm in manifest.get("algorithms") or []:
        path = benchmark_dir / str(algorithm)
        resolved = path.resolve()
        if path.is_dir() and resolved not in seen:
            ordered.append(resolved)
            seen.add(resolved)
    for record_path in sorted(benchmark_dir.rglob("benchmark_run_record.json"), key=lambda item: str(item).lower()):
        path = record_path.parent.resolve()
        if path in seen:
            continue
        ordered.append(path)
        seen.add(path)
    for path in sorted(benchmark_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_dir() or path.resolve() in seen:
            continue
        if (path / "benchmark_run_record.json").exists() or (path / "deployment").exists() or (path / "execution").exists():
            ordered.append(path)
    return ordered


def model_summary(algorithms: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("models"):
        return list(manifest.get("models") or [])
    models = {}
    for item in algorithms:
        key = str(item.get("model_path") or item.get("model_name") or "")
        if key and key not in models:
            models[key] = {
                "model_path": item.get("model_path"),
                "model_name": item.get("model_name"),
                "model_id": item.get("model_id"),
            }
    return list(models.values())


def summarize_algorithm(run_dir: Path) -> dict[str, Any]:
    run_record = load_json(run_dir / "benchmark_run_record.json", {})
    deployment_plan = load_json(run_dir / "deployment" / "deployment_plan.json", {})
    execution_manifest = load_json(run_dir / "execution" / "EXECUTION_MANIFEST.json", {})
    resource_summary = load_json(run_dir / "execution" / "resources" / "resource_summary.json", {})
    resource_by_workflow = load_json(run_dir / "execution" / "resources" / "resource_by_workflow.json", {})
    attempts = run_record.get("attempts") or []
    retry_count = int(run_record.get("retry_count") or max(0, len(attempts) - 1))
    final_max_parallel_nodes = run_record.get("max_parallel_nodes")
    if final_max_parallel_nodes is None and attempts:
        final_max_parallel_nodes = attempts[-1].get("max_parallel_nodes")
    algorithm = str(deployment_plan.get("strategy") or run_record.get("algorithm") or run_dir.name)
    model_path = str(
        run_record.get("model_path")
        or (deployment_plan.get("runtime_defaults") or {}).get("llama_model_path")
        or ""
    )
    model_name = str(run_record.get("model_name") or model_basename(model_path) or "default")
    model_id = str(run_record.get("model_id") or sanitize_label(model_name))
    run_label = f"{model_name} :: {algorithm}" if model_name else algorithm
    runs = execution_manifest.get("runs") or []
    included_results = []
    timeout_results = []
    for record in runs:
        result = load_result_for_record(record)
        result_summary = summarize_loaded_result(result)
        if is_timeout_result(record, result):
            timeout_results.append(timeout_summary(record, result))
            continue
        included_results.append({"record": record, "summary": result_summary})
    included_records = [item["record"] for item in included_results]
    filtered_resource_summary = filtered_resource_summary_for_records(resource_summary, resource_by_workflow, included_records, timeout_results)
    completed = sum(1 for item in included_records if item.get("status") == "completed")
    failed = len(included_records) - completed
    total_wall = sum(numeric(item.get("total_wall_ms")) for item in included_records)
    llm_node_total = sum(item["summary"]["llm_node_total_wall_ms"] for item in included_results)
    llm_inference_total = sum(item["summary"]["llm_node_inference_total_ms"] for item in included_results)
    llm_node_count = sum(item["summary"]["llm_node_count"] for item in included_results)
    placement = deployment_placement_counts(deployment_plan)
    metrics = filtered_resource_summary.get("metrics") or {}
    errors = list(run_record.get("errors") or [])
    if execution_manifest.get("resource_monitor", {}).get("status") not in {None, "completed", "disabled"}:
        errors.append({"type": "resource_monitor", "message": str(execution_manifest.get("resource_monitor", {}).get("status"))})
    status = run_record.get("status") or ("execution_completed" if execution_manifest else "missing_execution")
    if execution_manifest and timeout_results and not included_records:
        status = "execution_timeout_filtered"
    elif execution_manifest and timeout_results and failed:
        status = "execution_partial_timeouts_filtered"
    elif execution_manifest and timeout_results:
        status = "execution_completed_timeouts_filtered"
    elif execution_manifest and failed:
        status = "execution_partial"
    elif execution_manifest and not failed:
        status = "execution_completed"
    return {
        "algorithm": algorithm,
        "model_name": model_name,
        "model_path": model_path,
        "model_id": model_id,
        "run_label": run_label,
        "run_dir": str(run_dir),
        "status": status,
        "retry_count": retry_count,
        "final_max_parallel_nodes": final_max_parallel_nodes,
        "retryability": run_record.get("retryability") or {},
        "errors": errors,
        "workflow_count": len(included_records),
        "raw_workflow_count": len(runs),
        "included_workflow_count": len(included_records),
        "excluded_timeout_workflows": len(timeout_results),
        "timeout_workflows": timeout_results,
        "completed_workflows": completed,
        "failed_workflows": failed,
        "total_wall_ms": total_wall,
        "avg_workflow_wall_ms": total_wall / len(included_records) if included_records else None,
        "max_workflow_wall_ms": max((numeric(item.get("total_wall_ms")) for item in included_records), default=0.0),
        "llm_node_count": llm_node_count,
        "llm_node_total_wall_ms": llm_node_total,
        "llm_node_avg_wall_ms": llm_node_total / llm_node_count if llm_node_count else None,
        "llm_node_inference_total_ms": llm_inference_total,
        "llm_node_inference_avg_ms": llm_inference_total / llm_node_count if llm_node_count else None,
        "placement": placement,
        "resources": {
            "sample_count": filtered_resource_summary.get("sample_count"),
            "raw_sample_count": filtered_resource_summary.get("raw_sample_count", resource_summary.get("sample_count")),
            "status": filtered_resource_summary.get("status"),
            "filtered_for_visualization": filtered_resource_summary.get("filtered_for_visualization", False),
            "imputed_metrics": filtered_resource_summary.get("imputed_metrics") or resource_summary.get("imputed_metrics") or {},
            "gpu_busy_imputed": "gpu_busy_percent"
            in ((filtered_resource_summary.get("imputed_metrics") or resource_summary.get("imputed_metrics") or {}).keys()),
            "cpu_avg_percent": metric_stat(metrics, "cpu_util_percent", "avg"),
            "cpu_max_percent": metric_stat(metrics, "cpu_util_percent", "max"),
            "memory_max_mb": metric_stat(metrics, "mem_used_mb", "max"),
            "gpu_busy_avg_percent": metric_stat(metrics, "gpu_busy_percent", "avg"),
            "gpu_busy_max_percent": metric_stat(metrics, "gpu_busy_percent", "max"),
            "thermal_max_c": metric_stat(metrics, "thermal_max_c", "max"),
            "battery_capacity_last_percent": metric_stat(metrics, "battery_capacity_percent", "last"),
            "battery_current_avg_ma": metric_stat(metrics, "battery_current_ma", "avg"),
            "battery_temp_max_c": metric_stat(metrics, "battery_temp_c", "max"),
        },
        "links": {
            "deployment_html": str(run_dir / "deployment" / "visualizations" / "deployment_overview.html"),
            "execution_html": str(run_dir / "execution" / "performance_overview.html"),
            "perfetto_trace": str(run_dir / "execution" / "resources" / "wfgen_resource_trace.perfetto-trace"),
            "resource_summary": str(run_dir / "execution" / "resources" / "resource_summary.json"),
        },
        "workflows": [
            {
                "workflow_index": record.get("workflow_index"),
                "workflow_id": record.get("workflow_id"),
                "status": record.get("status"),
                "wall_ms": record.get("total_wall_ms"),
                "llm_node_total_wall_ms": included_results[index]["summary"]["llm_node_total_wall_ms"],
                "llm_node_inference_total_ms": included_results[index]["summary"]["llm_node_inference_total_ms"],
                "llm_node_count": included_results[index]["summary"]["llm_node_count"],
                "visualization": record.get("visualization"),
            }
            for index, record in enumerate(included_records)
        ],
    }


def summarize_result(record: dict[str, Any]) -> dict[str, Any]:
    return summarize_loaded_result(load_result_for_record(record))


def load_result_for_record(record: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(str(record.get("result_json") or ""))
    return load_json(result_path, {}) if result_path.exists() else {}


def summarize_loaded_result(result: dict[str, Any]) -> dict[str, Any]:
    llm_steps = [step for step in result.get("steps", []) if step.get("kind") == "llm"]
    llm_inference_values = [
        numeric((step.get("raw") or {}).get("inference_wall_ms"))
        for step in llm_steps
        if (step.get("raw") or {}).get("inference_wall_ms") is not None
    ]
    return {
        "llm_node_count": len(llm_steps),
        "llm_node_total_wall_ms": sum(numeric(step.get("wall_ms")) for step in llm_steps),
        "llm_node_inference_total_ms": sum(llm_inference_values),
    }


def is_timeout_result(record: dict[str, Any], result: dict[str, Any]) -> bool:
    status = str(record.get("status") or result.get("status") or "").lower()
    if "timeout" in status or "timed out" in status:
        return True
    error = result.get("error") or {}
    if timeout_text(error.get("type")) or timeout_text(error.get("message")):
        return True
    for step in result.get("steps") or []:
        if step.get("success", True):
            continue
        raw = step.get("raw") or {}
        if timeout_text(raw.get("error")) or timeout_text(raw.get("stderr")):
            return True
    return False


def timeout_text(value: Any) -> bool:
    text = str(value or "").lower()
    return "timeoutexpired" in text or "timed out" in text or "timeout" in text


def timeout_summary(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_index": record.get("workflow_index"),
        "workflow_id": record.get("workflow_id"),
        "name": record.get("name"),
        "status": record.get("status") or result.get("status"),
        "total_wall_ms": record.get("total_wall_ms") or result.get("total_wall_ms"),
        "result_json": record.get("result_json"),
        "reason": timeout_reason(result),
    }


def timeout_reason(result: dict[str, Any]) -> str:
    error = result.get("error") or {}
    if timeout_text(error.get("type")) or timeout_text(error.get("message")):
        return "workflow error timeout"
    timed_out_steps = []
    for step in result.get("steps") or []:
        raw = step.get("raw") or {}
        if timeout_text(raw.get("error")) or timeout_text(raw.get("stderr")):
            timed_out_steps.append(str(step.get("node_id") or "unknown"))
    if timed_out_steps:
        return "timed out step(s): " + ", ".join(timed_out_steps[:4])
    return "timeout"


def filtered_resource_summary_for_records(
    resource_summary: dict[str, Any],
    resource_by_workflow: dict[str, Any],
    included_records: list[dict[str, Any]],
    timeout_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not timeout_results:
        return resource_summary
    included_indices = [str(record.get("workflow_index")) for record in included_records]
    included_resources = [
        resource_by_workflow[index]
        for index in included_indices
        if isinstance(resource_by_workflow.get(index), dict)
    ]
    filtered = dict(resource_summary)
    filtered["filtered_for_visualization"] = True
    filtered["filter_reason"] = "timeout workflows removed from benchmark comparisons"
    filtered["excluded_timeout_workflows"] = timeout_results
    filtered["raw_sample_count"] = resource_summary.get("sample_count")
    filtered["sample_count"] = sum(int(resource.get("sample_count") or 0) for resource in included_resources)
    filtered["metrics"] = merge_resource_metrics(included_resources)
    return filtered


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


def deployment_placement_counts(deployment_plan: dict[str, Any]) -> dict[str, Any]:
    llm_devices: Counter[str] = Counter()
    llm_device_classes: Counter[str] = Counter()
    runtimes: Counter[str] = Counter()
    all_device_classes: Counter[str] = Counter()
    ngl_values: Counter[str] = Counter()
    for workflow in deployment_plan.get("workflows") or []:
        for node in workflow.get("node_deployments") or []:
            runtimes[str(node.get("selected_runtime") or "")] += 1
            all_device_classes[str(node.get("device_class") or "")] += 1
            if node.get("kind") != "llm":
                continue
            llm_devices[str(node.get("device") or "")] += 1
            llm_device_classes[str(node.get("device_class") or "")] += 1
            ngl_values[str(node.get("ngl"))] += 1
    return {
        "llm_device_classes": dict(sorted(llm_device_classes.items())),
        "llm_devices": dict(sorted(llm_devices.items())),
        "selected_runtimes": dict(sorted(runtimes.items())),
        "all_device_classes": dict(sorted(all_device_classes.items())),
        "llm_ngl_values": dict(sorted(ngl_values.items())),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MOBench Deployment Benchmark",
        "",
        f"Generated at UTC: `{summary['created_at_utc']}`",
        f"Benchmark dir: `{summary['benchmark_dir']}`",
        f"Source run: `{summary.get('source_run_dir') or ''}`",
        "",
        "| Model | Algorithm | Status | Retries | Max parallel | Shown/raw workflows | Timeouts removed | Total wall | Avg workflow | LLM inference total | LLM node wall | LLM placements | GPU busy avg/max | CPU avg/max | Mem max | Thermal max |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for item in summary.get("algorithms") or []:
        gpu_text = f"{format_percent(item['resources'].get('gpu_busy_avg_percent'))} / {format_percent(item['resources'].get('gpu_busy_max_percent'))}"
        if item["resources"].get("gpu_busy_imputed"):
            gpu_text = f"{gpu_text} (imputed)"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("model_name") or ""),
                    str(item["algorithm"]),
                    str(item["status"]),
                    str(item.get("retry_count", 0)),
                    str(item.get("final_max_parallel_nodes") or ""),
                    f"{item.get('workflow_count', 0)}/{item.get('raw_workflow_count', item.get('workflow_count', 0))}",
                    str(item.get("excluded_timeout_workflows", 0)),
                    format_ms(item.get("total_wall_ms")),
                    format_ms(item.get("avg_workflow_wall_ms")),
                    format_ms(item.get("llm_node_inference_total_ms")),
                    format_ms(item.get("llm_node_total_wall_ms")),
                    placement_text(item),
                    gpu_text,
                    f"{format_percent(item['resources'].get('cpu_avg_percent'))} / {format_percent(item['resources'].get('cpu_max_percent'))}",
                    format_mb(item["resources"].get("memory_max_mb")),
                    format_c(item["resources"].get("thermal_max_c")),
                ]
            )
            + " |"
        )
    if summary.get("ranking"):
        lines.extend(["", "## Ranking", ""])
        for item in summary["ranking"]:
            label = item.get("run_label") or item.get("algorithm")
            lines.append(f"{item['rank']}. `{label}` - {format_ms(item['total_wall_ms'])}")
    return "\n".join(lines) + "\n"


def render_html(summary: dict[str, Any]) -> str:
    algorithms = summary.get("algorithms") or []
    max_wall = max((numeric(item.get("total_wall_ms")) for item in algorithms), default=0.0)
    cards = [
        ("Runs", len(algorithms)),
        ("Models", len(summary.get("models") or [])),
        ("Completed", sum(1 for item in algorithms if str(item.get("status") or "").startswith("execution_completed"))),
        ("Timeouts Removed", sum(int(item.get("excluded_timeout_workflows") or 0) for item in algorithms)),
        ("Best Total", best_total_text(summary)),
        ("Source", Path(str(summary.get("source_run_dir") or "")).name),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>'
        for label, value in cards
    )
    imputation_note = render_imputation_note(algorithms)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOBench Deployment Benchmark</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #fff;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d9dee8;
      --bar: #0f766e;
      --ok: #15803d;
      --bad: #b91c1c;
      --warn: #a16207;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 "Segoe UI", Arial, sans-serif; }}
    header {{ padding: 24px 28px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    main {{ padding: 20px 28px 32px; display: grid; gap: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 24px rgba(15, 23, 42, .08); }}
    .card {{ padding: 14px 16px; min-height: 84px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .value {{ margin-top: 6px; font-size: 21px; font-weight: 700; word-break: break-word; }}
    .panel {{ padding: 16px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1280px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #344054; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; background: #f8fafc; }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .note {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; color: #713f12; padding: 10px 12px; margin-bottom: 12px; }}
    .bar {{ height: 8px; min-width: 110px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 5px; }}
    .bar > span {{ display: block; height: 100%; background: var(--bar); }}
  </style>
</head>
<body>
  <header>
    <h1>MOBench Deployment Benchmark</h1>
    <p>{esc(str(summary.get("benchmark_dir") or ""))}</p>
  </header>
  <main>
    <section class="cards">{card_html}</section>
    <section class="panel">
      <h2>Model And Algorithm Comparison</h2>
      {imputation_note}
      {render_algorithm_table(algorithms, max_wall)}
    </section>
    {render_timeout_table(algorithms)}
    <section class="panel">
      <h2>Per-Workflow Time</h2>
      {render_workflow_table(algorithms)}
    </section>
  </main>
</body>
</html>
"""


def render_algorithm_table(algorithms: list[dict[str, Any]], max_wall: float) -> str:
    rows = []
    for item in algorithms:
        resources = item.get("resources") or {}
        wall = numeric(item.get("total_wall_ms"))
        gpu_busy_text = f"{format_percent(resources.get('gpu_busy_avg_percent'))} / {format_percent(resources.get('gpu_busy_max_percent'))}"
        if resources.get("gpu_busy_imputed"):
            gpu_busy_text = f'{esc(gpu_busy_text)}<br><span class="muted">imputed</span>'
        else:
            gpu_busy_text = esc(gpu_busy_text)
        rows.append(
            "<tr>"
            f"<td><code>{esc(item.get('model_name'))}</code></td>"
            f"<td><code>{esc(item.get('algorithm'))}</code></td>"
            f"<td>{status_html(item.get('status'))}</td>"
            f"<td>{esc(retry_text(item))}</td>"
            f"<td>{esc(item.get('completed_workflows'))}/{esc(item.get('workflow_count'))}<br><span class=\"muted\">raw {esc(item.get('raw_workflow_count', item.get('workflow_count')))}, timeouts removed {esc(item.get('excluded_timeout_workflows', 0))}</span></td>"
            f"<td>{esc(format_ms(wall))}<div class=\"bar\"><span style=\"width:{bar_pct(wall, max_wall):.1f}%\"></span></div></td>"
            f"<td>{esc(format_ms(item.get('avg_workflow_wall_ms')))}</td>"
            f"<td>{esc(format_ms(item.get('llm_node_inference_total_ms')))}<br><span class=\"muted\">wall {esc(format_ms(item.get('llm_node_total_wall_ms')))}, {esc(item.get('llm_node_count'))} nodes</span></td>"
            f"<td>{esc(placement_text(item))}</td>"
            f"<td>{gpu_busy_text}</td>"
            f"<td>{esc(format_percent(resources.get('cpu_avg_percent')))} / {esc(format_percent(resources.get('cpu_max_percent')))}</td>"
            f"<td>{esc(format_mb(resources.get('memory_max_mb')))}</td>"
            f"<td>{esc(format_c(resources.get('thermal_max_c')))}</td>"
            f"<td>{esc(format_percent(resources.get('battery_capacity_last_percent')))}<br><span class=\"muted\">temp {esc(format_c(resources.get('battery_temp_max_c')))}</span></td>"
            f"<td>{path_link(item['links'].get('deployment_html'))}<br>{path_link(item['links'].get('execution_html'))}<br>{path_link(item['links'].get('perfetto_trace'))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="muted">No benchmark runs found.</p>'
    return (
        "<table><thead><tr><th>Model</th><th>Algorithm</th><th>Status</th><th>Retry</th><th>Completed / Shown</th><th>Total Wall</th>"
        "<th>Avg Workflow</th><th>LLM Inference Time</th><th>LLM Placement</th><th>GPU Busy Avg/Max</th>"
        "<th>CPU Avg/Max</th><th>Memory Max</th><th>Thermal Max</th><th>Battery</th><th>Links</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_imputation_note(algorithms: list[dict[str, Any]]) -> str:
    imputed = [
        item
        for item in algorithms
        if isinstance(item.get("resources"), dict) and item["resources"].get("gpu_busy_imputed")
    ]
    if not imputed:
        return ""
    sources = sorted(
        {
            str(
                ((item["resources"].get("imputed_metrics") or {}).get("gpu_busy_percent") or {}).get(
                    "source_benchmark_dir"
                )
                or ""
            )
            for item in imputed
        }
    )
    source_text = "; ".join(Path(source).name for source in sources if source) or "same-family source run"
    return (
        '<div class="note">'
        "GPU busy values marked imputed were copied from a same-family source benchmark because this device did not expose "
        f"readable KGSL GPU busy counters to adb. Source: <code>{esc(source_text)}</code>."
        "</div>"
    )


def render_timeout_table(algorithms: list[dict[str, Any]]) -> str:
    rows = []
    for item in algorithms:
        for workflow in item.get("timeout_workflows") or []:
            rows.append(
                "<tr>"
                f"<td><code>{esc(item.get('model_name'))}</code></td>"
                f"<td><code>{esc(item.get('algorithm'))}</code></td>"
                f"<td>{esc(workflow.get('workflow_index'))}</td>"
                f"<td><code>{esc(workflow.get('workflow_id'))}</code></td>"
                f"<td>{esc(format_ms(workflow.get('total_wall_ms')))}</td>"
                f"<td>{esc(workflow.get('reason'))}</td>"
                f"<td>{path_link(workflow.get('result_json'))}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return (
        '<section class="panel">'
        "<h2>Timeout Results Removed From Comparison</h2>"
        '<p class="muted">Raw result files are preserved. These workflow rows are excluded from algorithm timing and resource aggregates.</p>'
        "<table><thead><tr><th>Model</th><th>Algorithm</th><th>Index</th><th>Workflow</th><th>Wall</th><th>Reason</th><th>Result</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_workflow_table(algorithms: list[dict[str, Any]]) -> str:
    rows = []
    for item in algorithms:
        for workflow in item.get("workflows") or []:
            rows.append(
                "<tr>"
                f"<td><code>{esc(item.get('model_name'))}</code></td>"
                f"<td><code>{esc(item.get('algorithm'))}</code></td>"
                f"<td>{esc(workflow.get('workflow_index'))}</td>"
                f"<td><code>{esc(workflow.get('workflow_id'))}</code></td>"
                f"<td>{status_html(workflow.get('status'))}</td>"
                f"<td>{esc(format_ms(workflow.get('wall_ms')))}</td>"
                f"<td>{esc(format_ms(workflow.get('llm_node_inference_total_ms')))}<br><span class=\"muted\">wall {esc(format_ms(workflow.get('llm_node_total_wall_ms')))}, {esc(workflow.get('llm_node_count'))} nodes</span></td>"
                f"<td>{path_link(workflow.get('visualization'))}</td>"
                "</tr>"
            )
    if not rows:
        return '<p class="muted">No workflow rows found.</p>'
    return (
        "<table><thead><tr><th>Model</th><th>Algorithm</th><th>Index</th><th>Workflow</th><th>Status</th>"
        "<th>Wall</th><th>LLM Inference Time</th><th>Workflow HTML</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def placement_text(item: dict[str, Any]) -> str:
    counts = (item.get("placement") or {}).get("llm_device_classes") or {}
    pieces = [f"{key}:{counts[key]}" for key in ("cpu", "gpu", "npu") if counts.get(key)]
    return ", ".join(pieces) if pieces else ""


def retry_text(item: dict[str, Any]) -> str:
    retry_count = int(item.get("retry_count") or 0)
    max_parallel = item.get("final_max_parallel_nodes")
    text = f"{retry_count} retry" + ("" if retry_count == 1 else "ies")
    if max_parallel not in {None, ""}:
        text += f", p={max_parallel}"
    retryability = item.get("retryability") or {}
    reasons = retryability.get("retryable_reasons") or retryability.get("non_retryable_reasons") or []
    if reasons:
        text += f" - {str(reasons[0])[:80]}"
    return text


def best_total_text(summary: dict[str, Any]) -> str:
    ranking = summary.get("ranking") or []
    if not ranking:
        return ""
    first = ranking[0]
    return f"{first.get('run_label') or first['algorithm']} ({format_ms(first['total_wall_ms'])})"


def model_basename(model_path: str) -> str:
    return model_path.rstrip("/").rsplit("/", 1)[-1] if model_path else ""


def sanitize_label(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or "model"


def resolve_benchmark_dir(explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(f"benchmark_dir not found: {path}")
        return path
    candidates = []
    if DEFAULT_RESULTS_DIR.exists():
        for path in DEFAULT_RESULTS_DIR.glob("deployment_benchmark_*"):
            if path.is_dir():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No deployment_benchmark_* directory found under results/.")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1].resolve()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def metric_stat(metrics: dict[str, Any], metric: str, stat: str) -> Any:
    value = (metrics.get(metric) or {}).get(stat)
    return "" if value is None else value


def status_html(status: Any) -> str:
    text = str(status or "")
    cls = "ok" if text in {"completed", "execution_completed", "success", "ok"} else "bad" if "fail" in text or "error" in text else "warn"
    return f'<span class="{cls}">{esc(text)}</span>'


def bar_pct(value: float, max_value: float) -> float:
    if max_value <= 0 or value <= 0:
        return 0.0
    return max(2.0, min(100.0, value / max_value * 100.0))


def format_ms(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return ""
    if number >= 1000:
        return f"{number / 1000:.2f} s"
    return f"{number:.1f} ms"


def format_percent(value: Any) -> str:
    number = numeric_or_none(value)
    return "" if number is None else f"{number:.1f}%"


def format_mb(value: Any) -> str:
    number = numeric_or_none(value)
    return "" if number is None else f"{number:.1f} MB"


def format_c(value: Any) -> str:
    number = numeric_or_none(value)
    return "" if number is None else f"{number:.1f} C"


def path_link(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    if not path.exists():
        return f'<code>{esc(path.name or str(value))}</code>'
    return f'<a href="{esc(path.resolve().as_uri())}"><code>{esc(path.name)}</code></a>'


def numeric(value: Any) -> float:
    number = numeric_or_none(value)
    return 0.0 if number is None else number


def numeric_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
