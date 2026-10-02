"""Render MOBench workflow JSON and execution results as a standalone HTML view."""

from __future__ import annotations

import argparse
import html
import json
from collections import deque
from pathlib import Path
from typing import Any


KIND_COLORS = {
    "app": ("#e8f4ff", "#1d5f91"),
    "llm": ("#fff2cc", "#8a5a00"),
    "dnn": ("#e9f8ef", "#23693a"),
    "control": ("#f1f0ff", "#4b4398"),
    "data": ("#f4f4f5", "#52525b"),
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doc = json.loads(args.workflow_json.read_text(encoding="utf-8"))
    result = json.loads(args.result_json.read_text(encoding="utf-8")) if args.result_json else None
    workflow = select_workflow(doc, args.workflow_index, args.workflow_id)
    html_text = render_html(workflow, doc, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"wfgen: wrote workflow visualization to {args.output}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a MOBench workflow and optional execution result.")
    parser.add_argument("workflow_json", type=Path, help="Workflow JSON document.")
    parser.add_argument("-r", "--result-json", type=Path, help="Optional runtime result JSON.")
    parser.add_argument("-o", "--output", type=Path, default=Path("workflow_visualization.html"))
    parser.add_argument("--workflow-index", type=int, default=0)
    parser.add_argument("--workflow-id")
    return parser.parse_args(argv)


def select_workflow(doc: dict[str, Any], index: int, workflow_id: str | None) -> dict[str, Any]:
    workflows = doc.get("workflows") or []
    if workflow_id:
        for workflow in workflows:
            if workflow.get("workflow_id") == workflow_id:
                return workflow
        raise ValueError(f"workflow_id not found: {workflow_id}")
    if index < 0 or index >= len(workflows):
        raise IndexError(f"workflow_index {index} out of range for {len(workflows)} workflows")
    return workflows[index]


def render_html(workflow: dict[str, Any], doc: dict[str, Any], result: dict[str, Any] | None) -> str:
    steps = {step["node_id"]: step for step in (result or {}).get("steps", [])}
    waves = (result or {}).get("waves") or workflow.get("execution_hints", {}).get("parallel_waves") or topological_waves(workflow)
    positions, svg_size = layout_nodes(workflow, waves)
    edge_svg = render_edges(workflow, positions)
    node_svg = render_nodes(workflow, positions, steps)
    critical = workflow.get("execution_hints", {}).get("critical_path") or []
    metrics = summarize_metrics(workflow, result, waves, steps)
    inference_rows = extract_inference_rows(workflow, steps)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(workflow.get("name") or workflow.get("workflow_id"))} | MOBench Visualization</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d8dee8;
      --edge: #7b8794;
      --ok: #15803d;
      --bad: #b91c1c;
      --warn: #a16207;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 "Segoe UI", Arial, sans-serif;
    }}
    header {{
      padding: 24px 28px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 20px 28px 32px; display: grid; gap: 18px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{ padding: 14px 16px; min-height: 84px; }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    .metric .value {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 700;
      word-break: break-word;
    }}
    .panel {{ padding: 16px; overflow: auto; }}
    .graph-wrap {{ min-height: 480px; }}
    svg {{ max-width: 100%; height: auto; display: block; }}
    .edge-label {{
      font-size: 11px;
      fill: #334155;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 4px;
      stroke-linejoin: round;
    }}
    .wave-label {{
      font-size: 12px;
      fill: #475569;
      font-weight: 700;
    }}
    .node-title {{ font-size: 13px; font-weight: 700; fill: #111827; }}
    .node-meta {{ font-size: 11px; fill: #475569; }}
    .status-ok {{ fill: var(--ok); }}
    .status-bad {{ fill: var(--bad); }}
    .status-none {{ fill: var(--warn); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; background: #f8fafc; }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: #f8fafc;
    }}
    .ok {{ color: var(--ok); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .bar {{
      height: 8px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      min-width: 120px;
    }}
    .bar > span {{ display: block; height: 100%; background: #2563eb; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 Consolas, "SFMono-Regular", monospace;
      color: #334155;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{esc(workflow.get("name") or workflow.get("workflow_id"))}</h1>
    <p>{esc(workflow.get("description") or "")}</p>
  </header>
  <main>
    {render_metrics(metrics)}
    {render_resource_summary(result)}
    <section class="panel graph-wrap">
      <h2>Workflow DAG</h2>
      <svg viewBox="0 0 {svg_size[0]} {svg_size[1]}" role="img" aria-label="Workflow directed acyclic graph">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#7b8794"></path>
          </marker>
          <filter id="cardShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#0f172a" flood-opacity="0.12"/>
          </filter>
        </defs>
        {render_wave_labels(waves, positions)}
        {edge_svg}
        {node_svg}
      </svg>
    </section>
    <section class="panel">
      <h2>Model Inference Performance</h2>
      {render_inference_table(inference_rows)}
    </section>
    <section class="panel">
      <h2>Execution Timeline</h2>
      {render_steps_table(workflow, steps, metrics["max_step_wall_ms"])}
    </section>
    <section class="panel">
      <h2>Edges</h2>
      {render_edges_table(workflow)}
    </section>
    <section class="panel">
      <h2>Execution Waves</h2>
      {render_waves_table(waves, steps)}
    </section>
    <section class="panel">
      <h2>Critical Path</h2>
      <p><code>{esc(" -> ".join(critical) if critical else "not provided")}</code></p>
    </section>
    <section class="panel">
      <h2>Final Artifacts</h2>
      <pre>{esc(json.dumps((result or {}).get("artifacts", {}), indent=2, ensure_ascii=False) if result else "No execution result supplied.")}</pre>
    </section>
  </main>
</body>
</html>
"""


def render_metrics(metrics: dict[str, Any]) -> str:
    items = [
        ("Status", metrics["status"]),
        ("Total Wall", format_ms(metrics["total_wall_ms"])),
        ("Nodes", str(metrics["node_count"])),
        ("Edges", str(metrics["edge_count"])),
        ("Waves", str(metrics["wave_count"])),
        ("Slowest Node", metrics["slowest_node"]),
    ]
    cards = []
    for label, value in items:
        cards.append(
            f'<div class="metric"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div></div>'
        )
    return '<section class="metrics">' + "".join(cards) + "</section>"


def render_resource_summary(result: dict[str, Any] | None) -> str:
    if not result:
        return ""
    monitor = result.get("resource_monitor") or {}
    resource = result.get("resource_usage") or {}
    if not monitor and not resource:
        return ""
    metrics = resource.get("metrics") or {}
    cards = [
        ("Resource Monitor", monitor.get("status") or "unknown"),
        ("Resource Samples", str(resource.get("sample_count", monitor.get("sample_count", "")) or "")),
        ("CPU Avg", format_percent(metric_value(metrics, "cpu_util_percent", "avg"))),
        ("CPU Max", format_percent(metric_value(metrics, "cpu_util_percent", "max"))),
        ("Memory Max", format_mb(metric_value(metrics, "mem_used_mb", "max"))),
        ("GPU Busy Avg", format_percent(metric_value(metrics, "gpu_busy_percent", "avg"))),
        ("GPU Busy Max", format_percent(metric_value(metrics, "gpu_busy_percent", "max"))),
        ("Thermal Max", format_celsius(metric_value(metrics, "thermal_max_c", "max"))),
    ]
    card_html = []
    for label, value in cards:
        card_html.append(
            f'<div class="metric"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div></div>'
        )
    artifacts = (monitor.get("artifacts") or {})
    links = []
    for label, key in [
        ("Perfetto Trace", "perfetto_trace"),
        ("Samples CSV", "resource_samples"),
        ("Summary JSON", "resource_summary"),
        ("Workflow JSON", "resource_by_workflow"),
    ]:
        path = artifacts.get(key)
        if path:
            links.append(f'<li>{esc(label)}: {path_link(path)}</li>')
    note = (
        "NPU metrics are NNAPI/FastRPC/CDSP-adjacent and do not represent true HTP utilization percent."
    )
    return (
        '<section class="panel">'
        '<h2>Resource Usage</h2>'
        '<div class="metrics">'
        + "".join(card_html)
        + "</div>"
        + (f'<ul>{"".join(links)}</ul>' if links else "")
        + f'<p class="muted">{esc(note)}</p>'
        + "</section>"
    )


def layout_nodes(
    workflow: dict[str, Any],
    waves: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], tuple[int, int]]:
    node_width = 220
    node_height = 96
    x_gap = 120
    y_gap = 58
    left = 72
    top = 74
    max_wave_len = max((len(wave.get("node_ids", [])) for wave in waves), default=1)
    height = max(460, top * 2 + max_wave_len * node_height + (max_wave_len - 1) * y_gap)
    width = max(900, left * 2 + len(waves) * node_width + max(0, len(waves) - 1) * x_gap)
    positions: dict[str, dict[str, float]] = {}
    for wave_idx, wave in enumerate(waves):
        node_ids = wave.get("node_ids", [])
        block_height = len(node_ids) * node_height + max(0, len(node_ids) - 1) * y_gap
        start_y = max(top, (height - block_height) / 2)
        x = left + wave_idx * (node_width + x_gap)
        for row, node_id in enumerate(node_ids):
            positions[node_id] = {
                "x": x,
                "y": start_y + row * (node_height + y_gap),
                "w": node_width,
                "h": node_height,
                "wave": wave_idx,
            }
    for node in workflow.get("nodes", []):
        positions.setdefault(
            node["node_id"],
            {"x": left, "y": top, "w": node_width, "h": node_height, "wave": 0},
        )
    return positions, (int(width), int(height))


def render_wave_labels(waves: list[dict[str, Any]], positions: dict[str, dict[str, float]]) -> str:
    labels = []
    for wave in waves:
        node_ids = wave.get("node_ids") or []
        if not node_ids:
            continue
        first = positions[node_ids[0]]
        labels.append(
            f'<text class="wave-label" x="{first["x"]}" y="34">'
            f'Wave {esc(str(wave.get("wave_index", "")))}'
            f'</text>'
        )
    return "\n".join(labels)


def render_edges(workflow: dict[str, Any], positions: dict[str, dict[str, float]]) -> str:
    parts = []
    for edge in workflow.get("edges", []):
        src = positions.get(edge.get("from_node"))
        dst = positions.get(edge.get("to_node"))
        if not src or not dst:
            continue
        x1 = src["x"] + src["w"]
        y1 = src["y"] + src["h"] / 2
        x2 = dst["x"]
        y2 = dst["y"] + dst["h"] / 2
        mid = max(28, (x2 - x1) / 2)
        if x2 <= x1:
            mid = 70
        path = f"M{x1:.1f},{y1:.1f} C{x1 + mid:.1f},{y1:.1f} {x2 - mid:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
        label_x = (x1 + x2) / 2
        label_y = (y1 + y2) / 2 - 7
        artifact = edge.get("artifact", {})
        label = artifact.get("name") or "artifact"
        parts.append(f'<path d="{path}" fill="none" stroke="#7b8794" stroke-width="2" marker-end="url(#arrow)"></path>')
        parts.append(f'<text class="edge-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle">{esc(label)}</text>')
    return "\n".join(parts)


def render_nodes(
    workflow: dict[str, Any],
    positions: dict[str, dict[str, float]],
    steps: dict[str, dict[str, Any]],
) -> str:
    node_by_id = {node["node_id"]: node for node in workflow.get("nodes", [])}
    parts = []
    for node_id, pos in positions.items():
        node = node_by_id.get(node_id, {"node_id": node_id, "kind": "unknown"})
        step = steps.get(node_id)
        kind = node.get("kind", "unknown")
        fill, stroke = KIND_COLORS.get(kind, ("#f8fafc", "#64748b"))
        status_class = "status-none"
        status_text = "not run"
        if step:
            status_class = "status-ok" if step.get("success") else "status-bad"
            status_text = "ok" if step.get("success") else "failed"
        wall = format_ms(step.get("wall_ms")) if step else "no timing"
        title = truncate(node.get("label") or node_id, 28)
        entrypoint = truncate(node.get("entrypoint") or "", 31)
        backend = node.get("preferred_backend") or "unknown"
        x = pos["x"]
        y = pos["y"]
        parts.append(
            f'<g>'
            f'<rect x="{x}" y="{y}" width="{pos["w"]}" height="{pos["h"]}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2" filter="url(#cardShadow)"></rect>'
            f'<text class="node-title" x="{x + 14}" y="{y + 24}">{esc(title)}</text>'
            f'<text class="node-meta" x="{x + 14}" y="{y + 44}">{esc(kind)} | {esc(entrypoint)}</text>'
            f'<text class="node-meta" x="{x + 14}" y="{y + 64}">backend: {esc(str(backend))}</text>'
            f'<text class="node-meta" x="{x + 14}" y="{y + 84}">wall: {esc(wall)}</text>'
            f'<circle cx="{x + pos["w"] - 24}" cy="{y + 24}" r="7" class="{status_class}"></circle>'
            f'<text class="node-meta" x="{x + pos["w"] - 16}" y="{y + 28}">{esc(status_text)}</text>'
            f'</g>'
        )
    return "\n".join(parts)


def render_steps_table(workflow: dict[str, Any], steps: dict[str, dict[str, Any]], max_wall_ms: float) -> str:
    node_order = [node["node_id"] for node in workflow.get("nodes", [])]
    rows = []
    for node_id in node_order:
        node = next(node for node in workflow.get("nodes", []) if node["node_id"] == node_id)
        step = steps.get(node_id)
        wall_ms = float(step.get("wall_ms", 0.0)) if step else 0.0
        pct = 0 if max_wall_ms <= 0 else max(2, min(100, wall_ms / max_wall_ms * 100))
        status = '<span class="muted">not run</span>'
        if step:
            status = '<span class="ok">success</span>' if step.get("success") else '<span class="bad">failed</span>'
        params = ", ".join(f"{param.get('name')}={param.get('value')}" for param in node.get("parameters", []))
        rows.append(
            "<tr>"
            f"<td><code>{esc(node_id)}</code></td>"
            f"<td><span class=\"pill\">{esc(node.get('kind', ''))}</span></td>"
            f"<td>{status}</td>"
            f"<td>{esc(format_ms(wall_ms) if step else '')}<div class=\"bar\"><span style=\"width:{pct:.1f}%\"></span></div></td>"
            f"<td><code>{esc(node.get('entrypoint', ''))}</code></td>"
            f"<td>{esc(params)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Node</th><th>Kind</th><th>Status</th><th>Wall Time</th>"
        "<th>Entrypoint</th><th>Parameters</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def extract_inference_rows(
    workflow: dict[str, Any],
    steps: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    node_order = [node["node_id"] for node in workflow.get("nodes", [])]
    for node_id in node_order:
        step = steps.get(node_id)
        if not step or step.get("kind") != "llm":
            continue
        raw = step.get("raw") or {}
        response = raw.get("response") or {}
        timings = response.get("timings") or {}
        usage = response.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        choices = response.get("choices") or []
        finish_reason = ""
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason") or ""
        cached_tokens = prompt_details.get("cached_tokens", timings.get("cache_n", ""))
        prompt_tokens = usage.get("prompt_tokens", "")
        prefill_tokens = timings.get("prompt_n", "")
        if prefill_tokens == "" and isinstance(prompt_tokens, (int, float)) and isinstance(cached_tokens, (int, float)):
            prefill_tokens = max(0, int(prompt_tokens) - int(cached_tokens))
        rows.append(
            {
                "node_id": node_id,
                "model": response.get("model") or raw.get("model") or "",
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "prefill_tokens": prefill_tokens,
                "prefill_ms": timings.get("prompt_ms", ""),
                "prefill_tps": timings.get("prompt_per_second", ""),
                "decode_tokens": timings.get("predicted_n", usage.get("completion_tokens", "")),
                "decode_ms": timings.get("predicted_ms", ""),
                "decode_tps": timings.get("predicted_per_second", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "wall_ms": step.get("wall_ms", ""),
            }
        )
    return rows


def render_inference_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">No model timing data found in the execution result.</p>'
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{esc(row['node_id'])}</code></td>"
            f"<td><code>{esc(row['model'])}</code></td>"
            f"<td>{esc(row['prompt_tokens'])}</td>"
            f"<td>{esc(row['cached_tokens'])}</td>"
            f"<td>{esc(row['prefill_tokens'])}</td>"
            f"<td>{esc(format_ms(row['prefill_ms']))}</td>"
            f"<td>{esc(format_tps(row['prefill_tps']))}</td>"
            f"<td>{esc(row['decode_tokens'])}</td>"
            f"<td>{esc(format_ms(row['decode_ms']))}</td>"
            f"<td>{esc(format_tps(row['decode_tps']))}</td>"
            f"<td>{esc(row['total_tokens'])}</td>"
            f"<td>{esc(format_ms(row['wall_ms']))}</td>"
            f"<td>{esc(row['finish_reason'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Node</th><th>Model</th><th>Prompt Tokens</th><th>Cached</th>"
        "<th>Prefill Tokens</th><th>Prefill Time</th><th>Prefill Speed</th>"
        "<th>Decode Tokens</th><th>Decode Time</th><th>Decode Speed</th>"
        "<th>Total Tokens</th><th>Node Wall</th><th>Finish</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_edges_table(workflow: dict[str, Any]) -> str:
    rows = []
    for edge in workflow.get("edges", []):
        artifact = edge.get("artifact", {})
        rows.append(
            "<tr>"
            f"<td><code>{esc(edge.get('from_node', ''))}</code></td>"
            f"<td><code>{esc(edge.get('to_node', ''))}</code></td>"
            f"<td><code>{esc(artifact.get('name', ''))}</code> <span class=\"muted\">{esc(artifact.get('artifact_type', ''))}</span></td>"
            f"<td>{esc(edge.get('reason', ''))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>From</th><th>To</th><th>Artifact</th><th>Reason</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def render_waves_table(waves: list[dict[str, Any]], steps: dict[str, dict[str, Any]]) -> str:
    rows = []
    for wave in waves:
        node_ids = wave.get("node_ids", [])
        wall_ms = max((float(steps.get(node_id, {}).get("wall_ms", 0.0)) for node_id in node_ids), default=0.0)
        rows.append(
            "<tr>"
            f"<td>{esc(str(wave.get('wave_index', '')))}</td>"
            f"<td>{', '.join(f'<code>{esc(node_id)}</code>' for node_id in node_ids)}</td>"
            f"<td>{esc(format_ms(wall_ms)) if wall_ms else ''}</td>"
            f"<td>{esc(wave.get('reason', ''))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Wave</th><th>Nodes</th><th>Estimated Wall</th><th>Reason</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def summarize_metrics(
    workflow: dict[str, Any],
    result: dict[str, Any] | None,
    waves: list[dict[str, Any]],
    steps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    step_wall = {node_id: float(step.get("wall_ms", 0.0)) for node_id, step in steps.items()}
    slowest_node = ""
    if step_wall:
        slowest_id = max(step_wall, key=step_wall.get)
        slowest_node = f"{slowest_id} ({format_ms(step_wall[slowest_id])})"
    return {
        "status": (result or {}).get("status", "not run"),
        "total_wall_ms": float((result or {}).get("total_wall_ms", 0.0)),
        "node_count": len(workflow.get("nodes", [])),
        "edge_count": len(workflow.get("edges", [])),
        "wave_count": len(waves),
        "max_step_wall_ms": max(step_wall.values(), default=0.0),
        "slowest_node": slowest_node or "not run",
    }


def topological_waves(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    node_ids = [node["node_id"] for node in workflow.get("nodes", [])]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in workflow.get("edges", []):
        outgoing[edge["from_node"]].append(edge["to_node"])
        indegree[edge["to_node"]] += 1
    ready = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    waves = []
    while ready:
        wave_ids = list(ready)
        ready.clear()
        waves.append({"wave_index": len(waves), "node_ids": wave_ids, "reason": "Derived by visualizer."})
        for node_id in wave_ids:
            for dst in outgoing[node_id]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
    return waves


def format_ms(value: Any) -> str:
    try:
        ms = float(value)
    except (TypeError, ValueError):
        return ""
    if ms <= 0:
        return "0 ms"
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{ms:.1f} ms"


def format_tps(value: Any) -> str:
    try:
        tps = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{tps:.2f} tok/s"


def metric_value(metrics: dict[str, Any], name: str, stat: str) -> Any:
    return (metrics.get(name) or {}).get(stat)


def format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.1f}%"


def format_mb(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.1f} MB"


def format_celsius(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number:.1f} C"


def path_link(path: Any) -> str:
    text = str(path)
    try:
        href = Path(text).resolve().as_uri()
    except (OSError, ValueError):
        href = text
    return f'<a href="{esc(href)}"><code>{esc(text)}</code></a>'


def truncate(value: str, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
