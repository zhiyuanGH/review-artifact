"""HTML rendering for MOBench deployment plans."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any


def render_deployment_html(deployment_doc: dict[str, Any], workflow_index: int | None = None) -> str:
    plan = deployment_doc.get("deployment_plan") or {}
    workflows = plan.get("workflows") or []
    if workflow_index is not None:
        workflows = [item for item in workflows if int(item.get("workflow_index", -1)) == workflow_index]
    rows = []
    for workflow in workflows:
        for node in workflow.get("node_deployments") or []:
            rows.append(
                "<tr>"
                f"<td>{workflow.get('workflow_index')}</td>"
                f"<td><code>{esc(workflow.get('workflow_id'))}</code></td>"
                f"<td><code>{esc(node.get('node_id'))}</code></td>"
                f"<td>{esc(node.get('kind'))}</td>"
                f"<td><code>{esc(node.get('entrypoint'))}</code></td>"
                f"<td>{esc(node.get('algorithm_name'))}</td>"
                f"<td>{esc(node.get('fair_comparison'))}</td>"
                f"<td>{esc(node.get('selected_runtime'))}</td>"
                f"<td>{esc(node.get('device_class'))}</td>"
                f"<td><code>{esc(node.get('device'))}</code></td>"
                f"<td>{esc(node.get('ngl'))}</td>"
                f"<td>{esc(node.get('concurrency_group'))}</td>"
                f"<td>{esc(node.get('placement_reason'))}</td>"
                "</tr>"
            )
    group_rows = []
    for workflow in workflows:
        for group in workflow.get("concurrency_groups") or []:
            group_rows.append(
                "<tr>"
                f"<td>{workflow.get('workflow_index')}</td>"
                f"<td>{esc(group.get('group_id'))}</td>"
                f"<td>{', '.join(esc(item) for item in group.get('node_ids') or [])}</td>"
                f"<td>{', '.join(str(item) for item in group.get('wave_ids') or [])}</td>"
                "</tr>"
            )
    cards = [
        ("Strategy", plan.get("strategy")),
        ("Fair", plan.get("fair_comparison")),
        ("Workflows", len(plan.get("workflows") or [])),
        ("Nodes", sum(len(item.get("node_deployments") or []) for item in plan.get("workflows") or [])),
        ("Schema", deployment_doc.get("schema_version")),
    ]
    card_html = "".join(
        f'<div class="card"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>'
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MOBench Deployment Plan</title>
  <style>
    :root {{
      --bg: #f7f8fa;
      --panel: #fff;
      --ink: #1f2937;
      --muted: #64748b;
      --line: #d8dee8;
      --shadow: 0 8px 24px rgba(15, 23, 42, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 "Segoe UI", Arial, sans-serif; }}
    header {{ padding: 24px 28px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    main {{ padding: 20px 28px 32px; display: grid; gap: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    .card {{ padding: 14px 16px; min-height: 84px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .value {{ margin-top: 6px; font-size: 20px; font-weight: 700; word-break: break-word; }}
    .panel {{ padding: 16px; overflow: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 960px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; background: #f8fafc; }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>MOBench Deployment Plan</h1>
    <p>{esc(str(deployment_doc.get("run_dir") or ""))}</p>
  </header>
  <main>
    <section class="cards">{card_html}</section>
    <section class="panel">
      <h2>Node Placement</h2>
      <table>
        <thead><tr><th>Workflow</th><th>ID</th><th>Node</th><th>Kind</th><th>Entrypoint</th><th>Algorithm</th><th>Fair</th><th>Runtime</th><th>Device Class</th><th>Device</th><th>NGL</th><th>Group</th><th>Reason</th></tr></thead>
        <tbody>{''.join(rows) or '<tr><td colspan="13">No deployment rows.</td></tr>'}</tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Concurrency Groups</h2>
      <table>
        <thead><tr><th>Workflow</th><th>Group</th><th>Nodes</th><th>Waves</th></tr></thead>
        <tbody>{''.join(group_rows) or '<tr><td colspan="4">No groups.</td></tr>'}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def write_deployment_visualizations(deployment_doc: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    plan = deployment_doc.get("deployment_plan") or {}
    for workflow in plan.get("workflows") or []:
        idx = int(workflow.get("workflow_index", 0))
        name = sanitize(str(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{idx}"))
        path = output_dir / f"{idx:02d}_{name}.html"
        path.write_text(render_deployment_html(deployment_doc, idx), encoding="utf-8")
        paths.append(path)
    overview = output_dir / "deployment_overview.html"
    overview.write_text(render_deployment_html(deployment_doc), encoding="utf-8")
    paths.append(overview)
    return paths


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def sanitize(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    return clean.strip("_") or "workflow"
