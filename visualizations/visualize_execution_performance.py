"""Create an aggregate MOBench execution performance/resource dashboard."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    execute_dir = resolve_execute_dir(args.execute_dir)
    output = args.output or execute_dir / "performance_overview.html"
    payload = load_dashboard_payload(execute_dir)
    output.write_text(render_dashboard(execute_dir, payload), encoding="utf-8")
    print(f"wfgen: wrote execution performance dashboard to {output}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize aggregate MOBench execution time and resource usage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("execute_dir", nargs="?", type=Path, default=None, help="MOBench execute result directory.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path.")
    return parser.parse_args(argv)


def resolve_execute_dir(explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(f"execute_dir not found: {path}")
        return path
    candidates = []
    results_dir = REPO_ROOT / "results"
    for manifest_path in results_dir.rglob("EXECUTION_MANIFEST.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("entrypoint") in {"real-execution", "execute-visualize"}:
            candidates.append(manifest_path.parent)
    if not candidates:
        raise FileNotFoundError("No execute result directory found under results/.")
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1].resolve()


def load_dashboard_payload(execute_dir: Path) -> dict[str, Any]:
    manifest = load_json(execute_dir / "EXECUTION_MANIFEST.json", {})
    resource_summary = load_json(execute_dir / "resources" / "resource_summary.json", {})
    resource_by_workflow = load_json(execute_dir / "resources" / "resource_by_workflow.json", {})
    trace_processor = load_json(execute_dir / "resources" / "trace_processor_output.json", {})
    runs = []
    excluded_timeout_runs = []
    for record in manifest.get("runs", []):
        result_path = Path(record.get("result_json", ""))
        result = load_json(result_path, {}) if result_path.exists() else {}
        workflow_resource = resource_by_workflow.get(str(record.get("workflow_index")), {})
        run = {
            "record": record,
            "result": result,
            "resource": workflow_resource,
            "node_rows": node_rows(result),
            "inference_rows": inference_rows(result),
        }
        if is_timeout_result(record, result):
            excluded_timeout_runs.append(timeout_summary(record, result))
            continue
        runs.append(run)
    filtered_resource_summary = filtered_resource_summary_for_runs(
        resource_summary,
        resource_by_workflow,
        [run["record"] for run in runs],
        excluded_timeout_runs,
    )
    return {
        "manifest": manifest,
        "resource_summary": filtered_resource_summary,
        "resource_by_workflow": resource_by_workflow,
        "trace_processor": trace_processor,
        "runs": runs,
        "excluded_timeout_runs": excluded_timeout_runs,
    }


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


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


def filtered_resource_summary_for_runs(
    resource_summary: dict[str, Any],
    resource_by_workflow: dict[str, Any],
    included_records: list[dict[str, Any]],
    excluded_timeout_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not excluded_timeout_runs:
        return resource_summary
    included_indices = [str(record.get("workflow_index")) for record in included_records]
    included_resources = [
        resource_by_workflow[index]
        for index in included_indices
        if isinstance(resource_by_workflow.get(index), dict)
    ]
    filtered = dict(resource_summary)
    filtered["filtered_for_visualization"] = True
    filtered["filter_reason"] = "timeout workflows removed from dashboard comparisons"
    filtered["excluded_timeout_workflows"] = excluded_timeout_runs
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


def node_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in result.get("steps", []):
        rows.append(
            {
                "node_id": step.get("node_id"),
                "kind": step.get("kind"),
                "entrypoint": step.get("entrypoint"),
                "success": step.get("success"),
                "wall_ms": numeric(step.get("wall_ms")),
                "inference_wall_ms": numeric_or_none((step.get("raw") or {}).get("inference_wall_ms")),
            }
        )
    return rows


def inference_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in result.get("steps", []):
        if step.get("kind") != "llm":
            continue
        raw = step.get("raw") or {}
        response = raw.get("response") or {}
        timings = raw.get("timings") or response.get("timings") or {}
        usage = response.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        cached = prompt_details.get("cached_tokens", timings.get("cache_n"))
        prompt_tokens = usage.get("prompt_tokens")
        prefill_tokens = timings.get("prompt_n")
        if prefill_tokens is None and isinstance(prompt_tokens, (int, float)) and isinstance(cached, (int, float)):
            prefill_tokens = max(0, int(prompt_tokens) - int(cached))
        rows.append(
            {
                "node_id": step.get("node_id"),
                "model": response.get("model") or raw.get("model") or "",
                "runtime": raw.get("mode") or raw.get("runtime") or response.get("runtime") or "",
                "wall_ms": numeric(step.get("wall_ms")),
                "inference_wall_ms": timings.get("inference_wall_ms") or raw.get("inference_wall_ms"),
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "prefill_tokens": prefill_tokens if prefill_tokens is not None else timings.get("prompt_tokens"),
                "prefill_ms": timings.get("prompt_ms"),
                "prefill_tps": timings.get("prompt_per_second"),
                "decode_tokens": timings.get("predicted_n", usage.get("completion_tokens", timings.get("decode_tokens"))),
                "decode_ms": timings.get("predicted_ms", timings.get("decode_ms")),
                "decode_tps": timings.get("predicted_per_second", timings.get("decode_per_second")),
                "total_tokens": usage.get("total_tokens"),
                "finish": finish_reason(response),
            }
        )
    return rows


def finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices and isinstance(choices[0], dict):
        return choices[0].get("finish_reason") or ""
    return ""


def render_dashboard(execute_dir: Path, payload: dict[str, Any]) -> str:
    runs = payload["runs"]
    resource_summary = payload["resource_summary"]
    trace_processor = payload["trace_processor"]
    excluded_timeout_runs = payload.get("excluded_timeout_runs") or []
    workflow_max = max((numeric(run["record"].get("total_wall_ms")) for run in runs), default=0.0)
    node_max = max((row["wall_ms"] for run in runs for row in run["node_rows"]), default=0.0)
    cards = overview_cards(runs, resource_summary, trace_processor, excluded_timeout_runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOBench Execution Performance</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #fff;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d9dee8;
      --bar: #2563eb;
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
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 24px rgba(15, 23, 42, .08); }}
    .card {{ padding: 14px 16px; min-height: 84px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .value {{ margin-top: 6px; font-size: 22px; font-weight: 700; word-break: break-word; }}
    .panel {{ padding: 16px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #344054; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; background: #f8fafc; }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .warn {{ color: var(--warn); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .note {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; color: #713f12; padding: 10px 12px; }}
    .bar {{ height: 8px; min-width: 120px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 5px; }}
    .bar > span {{ display: block; height: 100%; background: var(--bar); }}
    .split {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>MOBench Execution Performance</h1>
    <p>{esc(str(execute_dir))}</p>
  </header>
  <main>
    <section class="cards">{cards}</section>
    {render_gpu_imputation_note(resource_summary)}
    {render_timeout_filter_note(excluded_timeout_runs)}
    <section class="panel">
      <h2>Workflow Time And Resource Summary</h2>
      {render_workflow_table(runs, workflow_max)}
    </section>
    <section class="panel">
      <h2>Model Inference Performance</h2>
      {render_inference_table(runs)}
    </section>
    <section class="panel">
      <h2>Node Time Breakdown</h2>
      {render_node_table(runs, node_max)}
    </section>
    <section class="split">
      <div class="panel">
        <h2>Perfetto Artifacts</h2>
        {render_artifacts(payload)}
      </div>
      <div class="panel">
        <h2>Trace Processor Summary</h2>
        <pre>{esc(trace_processor_summary(trace_processor))}</pre>
      </div>
    </section>
  </main>
</body>
</html>
"""


def render_gpu_imputation_note(resource_summary: dict[str, Any]) -> str:
    imputed = (resource_summary.get("imputed_metrics") or {}).get("gpu_busy_percent") or {}
    if not imputed:
        return ""
    source = Path(str(imputed.get("source_benchmark_dir") or "")).name or "same-family source run"
    return (
        '<section class="note">'
        "GPU busy values on this page were imputed from a same-family benchmark because this device did not expose "
        f"readable KGSL GPU busy counters to adb. Source: <code>{esc(source)}</code>."
        "</section>"
    )


def overview_cards(
    runs: list[dict[str, Any]],
    resource_summary: dict[str, Any],
    trace_processor: dict[str, Any],
    excluded_timeout_runs: list[dict[str, Any]],
) -> str:
    completed = sum(1 for run in runs if run["record"].get("status") == "completed")
    failed = len(runs) - completed
    total_wall = sum(numeric(run["record"].get("total_wall_ms")) for run in runs)
    metrics = resource_summary.get("metrics") or {}
    items = [
        ("Shown Workflows", str(len(runs))),
        ("Timeouts Removed", str(len(excluded_timeout_runs))),
        ("Completed", str(completed)),
        ("Failed", str(failed)),
        ("Total Workflow Wall", format_ms(total_wall)),
        ("CPU Avg", format_percent(metric_stat(metrics, "cpu_util_percent", "avg"))),
        ("Memory Max", format_mb(metric_stat(metrics, "mem_used_mb", "max"))),
        ("GPU Busy Max", format_percent(metric_stat(metrics, "gpu_busy_percent", "max"))),
        ("Thermal Max", format_c(metric_stat(metrics, "thermal_max_c", "max"))),
        ("Trace Processor", "ok" if trace_processor.get("returncode") == 0 else "check output"),
    ]
    return "".join(f'<div class="card"><div class="label">{esc(k)}</div><div class="value">{esc(v)}</div></div>' for k, v in items)


def render_timeout_filter_note(excluded_timeout_runs: list[dict[str, Any]]) -> str:
    if not excluded_timeout_runs:
        return ""
    rows = []
    for item in excluded_timeout_runs:
        rows.append(
            "<tr>"
            f"<td>{esc(item.get('workflow_index'))}</td>"
            f"<td><code>{esc(item.get('workflow_id'))}</code></td>"
            f"<td>{esc(format_ms(item.get('total_wall_ms')))}</td>"
            f"<td>{esc(item.get('reason'))}</td>"
            f"<td>{path_link(item.get('result_json'))}</td>"
            "</tr>"
        )
    return (
        '<section class="panel">'
        "<h2>Timeout Results Removed From Charts</h2>"
        '<p class="muted">Raw result files are preserved. These workflow rows are excluded from timing and resource comparisons.</p>'
        "<table><thead><tr><th>Index</th><th>Workflow</th><th>Wall Time</th><th>Reason</th><th>Result</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def render_workflow_table(runs: list[dict[str, Any]], max_wall: float) -> str:
    rows = []
    for run in runs:
        record = run["record"]
        resource = run["resource"]
        metrics = resource.get("metrics") or {}
        wall = numeric(record.get("total_wall_ms"))
        rows.append(
            "<tr>"
            f"<td>{esc(record.get('workflow_index'))}</td>"
            f"<td><code>{esc(record.get('workflow_id'))}</code><br><span class=\"muted\">{esc(record.get('name'))}</span></td>"
            f"<td>{status_html(record.get('status'))}</td>"
            f"<td>{esc(format_ms(wall))}<div class=\"bar\"><span style=\"width:{bar_pct(wall, max_wall):.1f}%\"></span></div></td>"
            f"<td>{esc(resource.get('sample_count', ''))}</td>"
            f"<td>{esc(format_percent(metric_stat(metrics, 'cpu_util_percent', 'avg')))}</td>"
            f"<td>{esc(format_percent(metric_stat(metrics, 'cpu_util_percent', 'max')))}</td>"
            f"<td>{esc(format_mb(metric_stat(metrics, 'mem_used_mb', 'max')))}</td>"
            f"<td>{esc(format_percent(metric_stat(metrics, 'gpu_busy_percent', 'avg')))}</td>"
            f"<td>{esc(format_percent(metric_stat(metrics, 'gpu_busy_percent', 'max')))}</td>"
            f"<td>{esc(format_c(metric_stat(metrics, 'thermal_max_c', 'max')))}</td>"
            f"<td>{path_link(record.get('visualization'))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Index</th><th>Workflow</th><th>Status</th><th>Wall Time</th>"
        "<th>Samples</th><th>CPU Avg</th><th>CPU Max</th><th>Memory Max</th><th>GPU Busy Avg</th>"
        "<th>GPU Busy Max</th><th>Thermal Max</th><th>Workflow HTML</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_inference_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        record = run["record"]
        for row in run["inference_rows"]:
            rows.append(
                "<tr>"
                f"<td>{esc(record.get('workflow_index'))}</td>"
                f"<td><code>{esc(row['node_id'])}</code></td>"
                f"<td><code>{esc(row['model'])}</code></td>"
                f"<td>{esc(row.get('runtime', ''))}</td>"
                f"<td>{esc(row.get('prompt_tokens', ''))}</td>"
                f"<td>{esc(row.get('cached_tokens', ''))}</td>"
                f"<td>{esc(row.get('prefill_tokens', ''))}</td>"
                f"<td>{esc(format_ms(row.get('prefill_ms')))}</td>"
                f"<td>{esc(format_tps(row.get('prefill_tps')))}</td>"
                f"<td>{esc(row.get('decode_tokens', ''))}</td>"
                f"<td>{esc(format_ms(row.get('decode_ms')))}</td>"
                f"<td>{esc(format_tps(row.get('decode_tps')))}</td>"
                f"<td>{esc(row.get('total_tokens', ''))}</td>"
                f"<td>{esc(format_ms(row.get('inference_wall_ms')))}</td>"
                f"<td>{esc(format_ms(row.get('wall_ms')))}</td>"
                f"<td>{esc(row.get('finish', ''))}</td>"
                "</tr>"
            )
    if not rows:
        return '<p class="muted">No model timing rows found.</p>'
    return (
        "<table><thead><tr><th>Workflow</th><th>Node</th><th>Model</th><th>Runtime</th><th>Prompt</th><th>Cached</th>"
        "<th>Prefill</th><th>Prefill Time</th><th>Prefill Speed</th><th>Decode</th><th>Decode Time</th>"
        "<th>Decode Speed</th><th>Total</th><th>Inference Time</th><th>Node Wall</th><th>Finish</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_node_table(runs: list[dict[str, Any]], max_wall: float) -> str:
    rows = []
    for run in runs:
        record = run["record"]
        for row in run["node_rows"]:
            wall = numeric(row.get("wall_ms"))
            inference = row.get("inference_wall_ms")
            rows.append(
                "<tr>"
                f"<td>{esc(record.get('workflow_index'))}</td>"
                f"<td><code>{esc(row.get('node_id'))}</code></td>"
                f"<td>{esc(row.get('kind'))}</td>"
                f"<td><code>{esc(row.get('entrypoint'))}</code></td>"
                f"<td>{status_html('completed' if row.get('success') else 'failed')}</td>"
                f"<td>{esc(format_ms(inference))}<br><span class=\"muted\">wall {esc(format_ms(wall))}</span>"
                f"<div class=\"bar\"><span style=\"width:{bar_pct(wall, max_wall):.1f}%\"></span></div></td>"
                "</tr>"
            )
    if not rows:
        return '<p class="muted">No node timing rows found.</p>'
    return (
        "<table><thead><tr><th>Workflow</th><th>Node</th><th>Kind</th><th>Entrypoint</th><th>Status</th><th>Time</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def render_artifacts(payload: dict[str, Any]) -> str:
    artifacts = ((payload.get("manifest") or {}).get("resource_monitor") or {}).get("artifacts") or {}
    if not artifacts:
        artifacts = (payload.get("resource_summary") or {}).get("artifacts") or {}
    rows = []
    for key, value in artifacts.items():
        rows.append(f"<tr><td><code>{esc(key)}</code></td><td>{path_link(value)}</td></tr>")
    if not rows:
        return '<p class="muted">No resource artifacts recorded.</p>'
    return "<table><thead><tr><th>Artifact</th><th>Path</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def trace_processor_summary(trace_processor: dict[str, Any]) -> str:
    pieces = [
        f"attempted: {trace_processor.get('attempted')}",
        f"available: {trace_processor.get('available')}",
        f"returncode: {trace_processor.get('returncode')}",
    ]
    stdout = (trace_processor.get("stdout") or "").strip()
    stderr = (trace_processor.get("stderr") or "").strip()
    if stdout:
        pieces.append("")
        pieces.append(stdout)
    if stderr:
        pieces.append("")
        pieces.append(stderr[-4000:])
    return "\n".join(pieces)


def status_html(status: Any) -> str:
    text = str(status or "")
    cls = "ok" if text in {"completed", "success", "ok"} else "bad" if text in {"failed", "error"} else "warn"
    return f'<span class="{cls}">{esc(text)}</span>'


def metric_stat(metrics: dict[str, Any], metric: str, stat: str) -> Any:
    value = (metrics.get(metric) or {}).get(stat)
    return "" if value is None else value


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


def format_tps(value: Any) -> str:
    number = numeric_or_none(value)
    return "" if number is None else f"{number:.2f} tok/s"


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
    text = str(value)
    try:
        href = Path(text).resolve().as_uri()
    except (OSError, ValueError):
        href = text
    label = Path(text).name or text
    return f'<a href="{esc(href)}"><code>{esc(label)}</code></a>'


def numeric(value: Any) -> float:
    number = numeric_or_none(value)
    return 0.0 if number is None else number


def numeric_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
