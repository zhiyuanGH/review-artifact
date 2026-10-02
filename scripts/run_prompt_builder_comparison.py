"""Compare full and compact-chain cloud prompt builders on the same samples."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from MOBench.workflow_planning.generate import TAU_BENCH_CONFIG, TAU_BENCH_REPO, TAU_BENCH_SPLIT  # noqa: E402
from MOBench.workflow_generation.sample_selector import select_tau_bench_samples  # noqa: E402


PROMPT_BUILDER_MODES = ["full", "compact-chain"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = (args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare] output_dir={output_dir}", flush=True)

    selected_payload, selection = select_tau_bench_samples(
        sample_count=args.sample_count,
        selection_strategy=args.selection_strategy,
        sample_offset=args.sample_offset,
        seed=args.seed,
        include_empty_actions=args.include_empty_actions,
        tau_repo=args.tau_repo,
        tau_config=args.tau_config,
        tau_split=args.tau_split,
        selected_samples=args.selected_samples,
    )
    selected_path = output_dir / "selected_samples.json"
    write_json(selected_path, selected_payload)
    print(
        f"[compare] selected {len(selected_payload.get('samples') or [])} sample(s) "
        f"indices={selection.get('indices')} artifact={selected_path}",
        flush=True,
    )

    results = []
    for mode in parse_modes(args.modes):
        mode_dir = output_dir / mode
        started = time.time()
        returncode = run_planning(mode, mode_dir, selected_path, args, output_dir / f"{mode}.log")
        run_wall_ms = (time.time() - started) * 1000.0
        stats = collect_mode_stats(mode_dir, mode, returncode, run_wall_ms)
        results.append(stats)
        print(
            f"[compare] {mode}: status={stats['status']} "
            f"avg_prompt_chars={stats['totals'].get('avg_prompt_chars', 0):.1f} "
            f"prompt_tokens={stats['totals'].get('prompt_tokens', 0)} "
            f"completion_tokens={stats['totals'].get('completion_tokens', 0)} "
            f"qwen_wall_ms={stats['totals'].get('qwen_wall_ms', 0):.1f}",
            flush=True,
        )

    report = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "sample_count": len(selected_payload.get("samples") or []),
        "selection": selection,
        "selected_samples": str(selected_path),
        "modes": results,
    }
    json_path = output_dir / "prompt_builder_comparison.json"
    md_path = output_dir / "prompt_builder_comparison.md"
    html_path = output_dir / "prompt_builder_comparison.html"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    print(f"[compare] wrote {json_path}", flush=True)
    print(f"[compare] wrote {md_path}", flush=True)
    print(f"[compare] wrote {html_path}", flush=True)
    return 0 if all(item["status"] == "ok" for item in results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full and compact-chain cloud planning on the same selected tau-bench samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-count", type=int, default=3, help="Number of samples to compare.")
    parser.add_argument("--modes", default="full,compact-chain", help="Comma-separated prompt builders to run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Comparison output directory.")
    parser.add_argument("--python", type=Path, default=None, help="Python executable used to run wfgen_plan.py.")
    parser.add_argument("--selection-strategy", choices=["diverse", "fixed"], default="diverse")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-empty-actions", action="store_true", default=False)
    parser.add_argument("--selected-samples", type=Path, default=None)
    parser.add_argument("--tau-repo", default=TAU_BENCH_REPO)
    parser.add_argument("--tau-config", default=TAU_BENCH_CONFIG)
    parser.add_argument("--tau-split", default=TAU_BENCH_SPLIT)
    parser.add_argument("--cloud-provider", default="qwen")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--cloud-config", type=Path, default=REPO_ROOT / "configs" / "qwen.local.json")
    parser.add_argument("--cloud-timeout", type=int, default=120)
    parser.add_argument("--cloud-temperature", type=float, default=0.0)
    parser.add_argument("--cloud-max-tokens", type=int, default=8192)
    parser.add_argument("--extra-instruction", default=None)
    parser.add_argument("--run-api-smoke-test", action="store_true", default=False, help="Run Qwen smoke test in each planning mode.")
    parser.add_argument("--no-visualize", action="store_true", default=False, help="Skip per-mode planning visualizations.")
    return parser.parse_args(argv)


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in PROMPT_BUILDER_MODES]
    if unknown:
        raise ValueError(f"Unknown prompt-builder mode(s): {', '.join(unknown)}")
    if not modes:
        raise ValueError("--modes must include at least one prompt builder.")
    return modes


def run_planning(mode: str, run_dir: Path, selected_path: Path, args: argparse.Namespace, log_path: Path) -> int:
    python = str(args.python.resolve()) if args.python else sys.executable
    cmd = [
        python,
        str(REPO_ROOT / "wfgen_plan.py"),
        "--run-dir",
        str(run_dir),
        "--sample-count",
        str(args.sample_count),
        "--selected-samples",
        str(selected_path),
        "--generation-mode",
        "cloud",
        "--prompt-builder",
        mode,
        "--cloud-provider",
        args.cloud_provider,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key-env",
        args.api_key_env,
        "--cloud-config",
        str(args.cloud_config),
        "--cloud-timeout",
        str(args.cloud_timeout),
        "--cloud-temperature",
        str(args.cloud_temperature),
        "--cloud-max-tokens",
        str(args.cloud_max_tokens),
    ]
    if not args.run_api_smoke_test:
        cmd.append("--skip-api-smoke-test")
    if args.extra_instruction:
        cmd.extend(["--extra-instruction", args.extra_instruction])
    if args.no_visualize:
        cmd.append("--no-visualize")

    print("", flush=True)
    print("[compare] " + " ".join(quote_part(part) for part in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[command] " + " ".join(quote_part(part) for part in cmd) + "\n")
        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def collect_mode_stats(run_dir: Path, mode: str, returncode: int, run_wall_ms: float) -> dict[str, Any]:
    planning_dir = run_dir / "planning"
    samples_dir = planning_dir / "samples"
    sample_stats = []
    if samples_dir.exists():
        for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
            sample_stats.append(collect_sample_stats(sample_dir))

    workflow_path = planning_dir / "workflow.json"
    workflow_stats = collect_workflow_stats(workflow_path)
    totals = aggregate_sample_stats(sample_stats)
    totals["run_wall_ms"] = run_wall_ms
    status = "ok" if returncode == 0 and workflow_stats.get("validation_status") == "ok" else "error"
    error = ""
    if returncode != 0:
        error = f"wfgen_plan.py exited with code {returncode}"
    elif workflow_stats.get("validation_status") != "ok":
        error = str(workflow_stats.get("validation_error") or "workflow validation failed")
    return {
        "mode": mode,
        "status": status,
        "error": error,
        "run_dir": str(run_dir),
        "planning_dir": str(planning_dir),
        "sample_count": len(sample_stats),
        "samples": sample_stats,
        "workflow": workflow_stats,
        "totals": totals,
    }


def collect_sample_stats(sample_dir: Path) -> dict[str, Any]:
    prompt_dir = sample_dir / "prompts"
    llm_dir = sample_dir / "llm"
    raw_path = llm_dir / "raw_response.json"
    raw = read_json(raw_path)
    usage = raw.get("usage") if isinstance(raw, dict) else {}
    content = str(raw.get("content") or "") if isinstance(raw, dict) else ""
    system_chars = text_len(prompt_dir / "system_prompt.txt")
    user_chars = text_len(prompt_dir / "user_prompt.txt")
    schema_chars = text_len(prompt_dir / "response_schema.json")
    tool_list_chars = text_len(prompt_dir / "tool_list.json")
    chain_registry_chars = text_len(prompt_dir / "chain_registry.json")
    return {
        "sample_dir": str(sample_dir),
        "system_chars": system_chars,
        "user_prompt_chars": user_chars,
        "schema_chars": schema_chars,
        "catalog_chars": chain_registry_chars or tool_list_chars,
        "tool_list_chars": tool_list_chars,
        "chain_registry_chars": chain_registry_chars,
        "prompt_chars": system_chars + user_chars,
        "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
        "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
        "total_tokens": int((usage or {}).get("total_tokens") or 0),
        "response_bytes": len(content.encode("utf-8")),
        "qwen_wall_ms": float(raw.get("wall_ms") or 0.0) if isinstance(raw, dict) else 0.0,
        "response_status": raw.get("status", "missing") if isinstance(raw, dict) else "missing",
        "response_format": raw.get("response_format", "") if isinstance(raw, dict) else "",
    }


def aggregate_sample_stats(sample_stats: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float] = {
        "system_chars": 0.0,
        "user_prompt_chars": 0.0,
        "schema_chars": 0.0,
        "catalog_chars": 0.0,
        "prompt_chars": 0.0,
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "response_bytes": 0.0,
        "qwen_wall_ms": 0.0,
    }
    for sample in sample_stats:
        for key in list(totals):
            totals[key] += float(sample.get(key) or 0.0)
    count = max(len(sample_stats), 1)
    totals.update(
        {
            "avg_prompt_chars": totals["prompt_chars"] / count,
            "avg_user_prompt_chars": totals["user_prompt_chars"] / count,
            "avg_prompt_tokens": totals["prompt_tokens"] / count,
            "avg_completion_tokens": totals["completion_tokens"] / count,
            "avg_response_bytes": totals["response_bytes"] / count,
            "avg_qwen_wall_ms": totals["qwen_wall_ms"] / count,
        }
    )
    return totals


def collect_workflow_stats(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not payload:
        return {
            "workflow_json": str(path),
            "validation_status": "missing",
            "validation_error": "workflow.json missing or unreadable",
            "workflow_count": 0,
            "expanded_node_count": 0,
            "chain_node_count": 0,
        }
    workflows = payload.get("workflows") or []
    chain_node_count = 0
    tool_node_count = 0
    llm_node_count = 0
    expanded_node_count = 0
    for workflow in workflows:
        for node in workflow.get("nodes") or []:
            expanded_node_count += 1
            if node.get("kind") == "chain" or str(node.get("entrypoint") or "").startswith("chain:"):
                chain_node_count += 1
            if node.get("kind") in {"app", "dnn"}:
                tool_node_count += 1
            if node.get("kind") == "llm":
                llm_node_count += 1
    status = "ok" if chain_node_count == 0 else "error"
    return {
        "workflow_json": str(path),
        "validation_status": status,
        "validation_error": "" if status == "ok" else "expanded workflow contains chain nodes",
        "workflow_count": len(workflows),
        "expanded_node_count": expanded_node_count,
        "tool_node_count": tool_node_count,
        "llm_node_count": llm_node_count,
        "chain_node_count": chain_node_count,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Prompt Builder Comparison",
        "",
        f"Created UTC: `{report['created_at_utc']}`",
        f"Sample count: `{report['sample_count']}`",
        f"Selected samples: `{report['selected_samples']}`",
        "",
        "| Mode | Status | Avg prompt chars | Prompt tokens | Completion tokens | Response bytes | Qwen wall ms | Workflows | Nodes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in report["modes"]:
        totals = mode.get("totals") or {}
        workflow = mode.get("workflow") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    mode["mode"],
                    mode["status"],
                    f"{totals.get('avg_prompt_chars', 0):.1f}",
                    str(int(totals.get("prompt_tokens", 0))),
                    str(int(totals.get("completion_tokens", 0))),
                    str(int(totals.get("response_bytes", 0))),
                    f"{totals.get('qwen_wall_ms', 0):.1f}",
                    str(workflow.get("workflow_count", 0)),
                    str(workflow.get("expanded_node_count", 0)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_html(report: dict[str, Any]) -> str:
    rows = []
    for mode in report["modes"]:
        totals = mode.get("totals") or {}
        workflow = mode.get("workflow") or {}
        rows.append(
            "<tr>"
            f"<td>{esc(mode['mode'])}</td>"
            f"<td>{esc(mode['status'])}</td>"
            f"<td>{totals.get('avg_prompt_chars', 0):.1f}</td>"
            f"<td>{int(totals.get('prompt_tokens', 0))}</td>"
            f"<td>{int(totals.get('completion_tokens', 0))}</td>"
            f"<td>{int(totals.get('response_bytes', 0))}</td>"
            f"<td>{totals.get('qwen_wall_ms', 0):.1f}</td>"
            f"<td>{workflow.get('workflow_count', 0)}</td>"
            f"<td>{workflow.get('expanded_node_count', 0)}</td>"
            f"<td><a href=\"{file_href(mode['planning_dir'])}\">planning</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MOBench Prompt Builder Comparison</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #1f2937; background: #fafafa; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #4b5563; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child {{ text-align: left; }}
    th {{ background: #f3f4f6; font-weight: 600; }}
    a {{ color: #0f766e; }}
  </style>
</head>
<body>
  <h1>MOBench Prompt Builder Comparison</h1>
  <div class="meta">Created UTC: {esc(report['created_at_utc'])} · Samples: {report['sample_count']}</div>
  <table>
    <thead>
      <tr>
        <th>Mode</th><th>Status</th><th>Avg prompt chars</th><th>Prompt tokens</th>
        <th>Completion tokens</th><th>Response bytes</th><th>Qwen wall ms</th>
        <th>Workflows</th><th>Nodes</th><th>Artifacts</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def text_len(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8"))
    except OSError:
        return 0


def default_output_dir() -> Path:
    return REPO_ROOT / "results" / f"prompt_builder_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def quote_part(part: object) -> str:
    text = str(part)
    if not text or any(char.isspace() for char in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def esc(value: object) -> str:
    return html.escape(str(value))


def file_href(path: object) -> str:
    return Path(str(path)).resolve().as_uri()


if __name__ == "__main__":
    raise SystemExit(main())
