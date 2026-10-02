"""Benchmark a phone-side Qualcomm Genie bundle with warmup-aware timing."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADB = Path(os.environ.get("ADB") or shutil.which("adb") or "adb")
DEFAULT_BUNDLE_DIR = "/data/local/tmp/genie_bundle"
DEFAULT_CONFIG = "genie_config.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adb = discover_adb(args.adb)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[genie] output_dir={output_dir}", flush=True)
    print(f"[genie] adb={adb}", flush=True)
    print(f"[genie] serial={args.adb_serial or '<default>'}", flush=True)
    print(f"[genie] bundle={args.bundle_dir} config={args.config}", flush=True)

    backends = parse_backends(args.backends)
    generated_configs = []
    rows = []
    measured = []
    warmup = []
    for backend in backends:
        print(f"[genie] backend={backend}", flush=True)
        config_info = prepare_backend_config(adb, args, output_dir, backend)
        generated_configs.append(config_info)
        for index in range(max(0, args.warmup_runs)):
            print(f"[genie] {backend} warmup {index + 1}/{args.warmup_runs}", flush=True)
            run_dir = output_dir / "warmup" / backend / f"{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            warmup.append(run_once(adb, args, run_dir, config_info, profile=False, label=f"{backend}_warmup_{index}"))

        for index in range(max(1, args.repetitions)):
            print(f"[genie] {backend} measured {index + 1}/{args.repetitions}", flush=True)
            run_dir = output_dir / "runs" / backend / f"{index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = run_once(adb, args, run_dir, config_info, profile=True, label=f"{backend}_run_{index}")
            measured.append(result)
            rows.append(row_from_result(result, index, backend))
            print(
                f"[genie] {backend} run {index} status={result['status']} "
                f"wall_s={result['wall_ms'] / 1000.0:.2f} "
                f"decode_tps={format_number((result.get('profile_summary') or {}).get('token_generation_rate'))}",
                flush=True,
            )

    summary = {
        "created_at_utc": utc_now(),
        "adb": str(adb),
        "adb_serial": args.adb_serial or "",
        "bundle_dir": args.bundle_dir,
        "config": args.config,
        "backends": backends,
        "max_num_tokens": args.max_num_tokens,
        "prompt": args.prompt,
        "timeout_seconds": args.timeout,
        "warmup_runs": args.warmup_runs,
        "repetitions": args.repetitions,
        "warmup": warmup,
        "runs": measured,
        "generated_configs": generated_configs,
        "rows": rows,
        "aggregate": aggregate_rows(rows),
    }
    write_json(output_dir / "genie_benchmark_summary.json", summary)
    write_csv(output_dir / "genie_benchmark_summary.csv", rows)
    (output_dir / "genie_benchmark_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_dir / "genie_benchmark_summary.html").write_text(render_html(summary), encoding="utf-8")
    print(f"[genie] wrote {output_dir / 'genie_benchmark_summary.html'}", flush=True)
    return 0 if measured and all(item.get("status") == "ok" for item in measured) else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phone-side Genie text generation and parse Genie profile timing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB if DEFAULT_ADB.exists() else None)
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config filename or absolute phone path.")
    parser.add_argument("--backends", default="npu", help="Comma-separated Genie backends: cpu,gpu,npu.")
    parser.add_argument("--prompt", default="OK")
    parser.add_argument("--max-num-tokens", type=int, default=40, help="Injected dialog max-num-tokens for generated backend configs; use 0 to preserve config.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--log-level", default="warn", choices=["error", "warn", "info", "verbose"])
    return parser.parse_args(argv)


def run_once(
    adb: Path,
    args: argparse.Namespace,
    run_dir: Path,
    config_info: dict[str, Any],
    *,
    profile: bool,
    label: str,
) -> dict[str, Any]:
    token = f"{int(time.time() * 1000)}_{os.getpid()}_{label}"
    prompt_remote = f"{args.bundle_dir.rstrip('/')}/wfgen_prompt_{token}.txt"
    profile_remote = f"{args.bundle_dir.rstrip('/')}/wfgen_profile_{token}.json"
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    profile_path = run_dir / "profile.json"

    write_remote_file(adb, args.adb_serial, prompt_remote, args.prompt)
    profile_arg = f" --profile {sh_quote(profile_remote)}" if profile else ""
    remote_cmd = (
        f"cd {sh_quote(args.bundle_dir)} && "
        f"rm -f {sh_quote(profile_remote)} && "
        "LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        f"timeout {int(args.timeout)} ./genie-t2t-run "
        f"-c {sh_quote(config_info['remote_config'])} "
        f"--prompt_file {sh_quote(prompt_remote)} "
        f"--log {sh_quote(args.log_level)}"
        f"{profile_arg}"
    )
    started = time.time()
    stdout = ""
    stderr = ""
    returncode = -1
    status = "error"
    error = ""
    try:
        completed = adb_shell(adb, args.adb_serial, remote_cmd, timeout=args.timeout + 20)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        status = "ok" if completed.returncode == 0 else ("timeout" if completed.returncode == 124 else "error")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "timeout"
        error = f"Timed out after {args.timeout + 20} seconds."
    finally:
        adb_shell(adb, args.adb_serial, f"rm -f {sh_quote(prompt_remote)}", timeout=30, check=False)
    finished = time.time()
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

    profile_summary: dict[str, Any] = {}
    if profile:
        pulled = pull_remote_profile(adb, args.adb_serial, profile_remote, profile_path)
        if pulled:
            profile_summary = parse_profile(profile_path)
        adb_shell(adb, args.adb_serial, f"rm -f {sh_quote(profile_remote)}", timeout=30, check=False)

    return {
        "label": label,
        "status": status,
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_ms": (finished - started) * 1000.0,
        "returncode": returncode,
        "backend": config_info["backend"],
        "remote_config": config_info["remote_config"],
        "local_config": config_info["local_config"],
        "error": error or first_error_line(stderr, stdout) if status not in {"ok", "timeout"} else error,
        "command": remote_cmd,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "profile": str(profile_path) if profile_path.exists() else None,
        "profile_summary": profile_summary,
    }


def parse_profile(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    events = []
    for component in data.get("components") or []:
        events.extend(component.get("events") or [])
    create = next((event for event in events if event.get("type") == "GenieDialog_create"), {})
    query = next((event for event in events if event.get("type") == "GenieDialog_query"), {})
    return {
        "init_us": metric_value(create, "init-time", create.get("duration")),
        "query_us": query.get("duration"),
        "prompt_tokens": metric_value(query, "num-prompt-tokens"),
        "prompt_processing_rate": metric_value(query, "prompt-processing-rate"),
        "time_to_first_token_us": metric_value(query, "time-to-first-token"),
        "generated_tokens": metric_value(query, "num-generated-tokens"),
        "token_generation_rate": metric_value(query, "token-generation-rate"),
        "token_generation_time_us": metric_value(query, "token-generation-time"),
    }


def metric_value(event: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = event.get(key)
    if isinstance(value, dict):
        return value.get("value")
    return fallback if value is None else value


def parse_backends(raw: str) -> list[str]:
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    aliases = {"htp": "npu", "qnnhtp": "npu", "qnn-gpu": "gpu", "qnncpu": "cpu"}
    backends = [aliases.get(item, item) for item in backends]
    unknown = [item for item in backends if item not in {"cpu", "gpu", "npu"}]
    if unknown:
        raise ValueError(f"Unknown Genie backend(s): {', '.join(unknown)}")
    return backends or ["npu"]


def prepare_backend_config(
    adb: Path,
    args: argparse.Namespace,
    output_dir: Path,
    backend: str,
) -> dict[str, Any]:
    base_remote = args.config if str(args.config).startswith("/") else f"{args.bundle_dir.rstrip('/')}/{args.config}"
    completed = adb_shell(adb, args.adb_serial, f"cat {sh_quote(base_remote)}", timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to read Genie config {base_remote}: {completed.stderr or completed.stdout}")
    config = json.loads(completed.stdout)
    mutate_config_for_backend(config, backend, args.max_num_tokens)
    local_path = output_dir / "generated_configs" / f"genie_config_{backend}.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    remote_path = f"{args.bundle_dir.rstrip('/')}/wfgen_genie_config_{backend}_{int(time.time() * 1000)}.json"
    push_file(adb, args.adb_serial, local_path, remote_path)
    return {
        "backend": backend,
        "base_remote_config": base_remote,
        "remote_config": remote_path,
        "local_config": str(local_path),
        "note": backend_note(config, backend),
    }


def mutate_config_for_backend(config: dict[str, Any], backend: str, max_num_tokens: int) -> None:
    dialog = config.setdefault("dialog", {})
    if max_num_tokens > 0:
        dialog["max-num-tokens"] = int(max_num_tokens)
    engine = dialog.setdefault("engine", {})
    current_backend = engine.setdefault("backend", {})
    if backend == "npu":
        htp = current_backend.get("QnnHtp") if isinstance(current_backend.get("QnnHtp"), dict) else {}
        engine["backend"] = {
            "version": int(current_backend.get("version", 1)),
            "type": "QnnHtp",
            "QnnHtp": htp or {
                "version": 1,
                "use-mmap": True,
                "spill-fill-bufsize": 0,
                "mmap-budget": 0,
                "poll": True,
                "cpu-mask": "0xe0",
                "allow-async-init": False,
            },
        }
        if current_backend.get("extensions"):
            engine["backend"]["extensions"] = current_backend["extensions"]
        return
    if backend == "gpu":
        engine["backend"] = {"version": int(current_backend.get("version", 1)), "type": "QnnGpu"}
        return
    engine["backend"] = {
        "version": int(current_backend.get("version", 1)),
        "type": "QnnGenAiTransformer",
        "QnnGenAiTransformer": {"version": 1, "kv-quantization": False},
    }


def backend_note(config: dict[str, Any], backend: str) -> str:
    model = (((config.get("dialog") or {}).get("engine") or {}).get("model") or {})
    model_type = model.get("type")
    if backend == "cpu" and model_type != "library":
        return "CPU Genie backend usually expects model.type=library; binary ctx-bins may fail."
    if backend == "gpu" and model_type != "binary":
        return "GPU Genie backend examples use model.type=binary."
    return ""


def row_from_result(result: dict[str, Any], index: int, backend: str) -> dict[str, Any]:
    profile = result.get("profile_summary") or {}
    return {
        "index": index,
        "backend": backend,
        "status": result.get("status"),
        "wall_ms": result.get("wall_ms"),
        "init_us": profile.get("init_us"),
        "query_us": profile.get("query_us"),
        "prompt_tokens": profile.get("prompt_tokens"),
        "prompt_processing_rate": profile.get("prompt_processing_rate"),
        "time_to_first_token_us": profile.get("time_to_first_token_us"),
        "generated_tokens": profile.get("generated_tokens"),
        "token_generation_rate": profile.get("token_generation_rate"),
        "token_generation_time_us": profile.get("token_generation_time_us"),
        "stdout": result.get("stdout"),
        "stderr": result.get("stderr"),
        "profile": result.get("profile"),
        "local_config": result.get("local_config"),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    return {
        "ok_count": len(ok),
        "count": len(rows),
        "avg_wall_ms": avg(row.get("wall_ms") for row in ok),
        "avg_query_us": avg(row.get("query_us") for row in ok),
        "avg_init_us": avg(row.get("init_us") for row in ok),
        "avg_prompt_processing_rate": avg(row.get("prompt_processing_rate") for row in ok),
        "avg_token_generation_rate": avg(row.get("token_generation_rate") for row in ok),
    }


def avg(values: Any) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "backend",
        "status",
        "wall_ms",
        "init_us",
        "query_us",
        "prompt_tokens",
        "prompt_processing_rate",
        "time_to_first_token_us",
        "generated_tokens",
        "token_generation_rate",
        "token_generation_time_us",
        "profile",
        "local_config",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phone Genie Benchmark",
        "",
        f"Created UTC: `{summary['created_at_utc']}`",
        f"Bundle: `{summary['bundle_dir']}`",
        f"Config: `{summary['config']}`",
        f"Warmup runs before measured rows: `{summary['warmup_runs']}`",
        "",
        "| Run | Backend | Status | Wall | Init | Query | Prompt tok/s | Generated | Decode tok/s | Profile |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["index"]),
                    str(row.get("backend") or ""),
                    str(row["status"]),
                    format_ms(row.get("wall_ms")),
                    format_us(row.get("init_us")),
                    format_us(row.get("query_us")),
                    format_number(row.get("prompt_processing_rate")),
                    str(row.get("generated_tokens") or ""),
                    format_number(row.get("token_generation_rate")),
                    str(row.get("profile") or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_html(summary: dict[str, Any]) -> str:
    rows = []
    for row in summary["rows"]:
        rows.append(
            "<tr>"
            f"<td>{esc(row.get('index'))}</td>"
            f"<td>{esc(row.get('backend'))}</td>"
            f"<td>{esc(row.get('status'))}</td>"
            f"<td>{esc(format_ms(row.get('wall_ms')))}</td>"
            f"<td>{esc(format_us(row.get('init_us')))}</td>"
            f"<td>{esc(format_us(row.get('query_us')))}</td>"
            f"<td>{esc(format_number(row.get('prompt_processing_rate')))}</td>"
            f"<td>{esc(row.get('generated_tokens') or '')}</td>"
            f"<td>{esc(format_number(row.get('token_generation_rate')))}</td>"
            f"<td>{path_link(row.get('profile'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phone Genie Benchmark</title>
  <style>
    body {{ margin: 28px; font: 14px/1.45 "Segoe UI", Arial, sans-serif; color: #1f2937; background: #f6f7f9; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ margin: 0 0 18px; color: #667085; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #d9dee8; text-align: left; }}
    th {{ background: #f8fafc; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    code {{ font-family: Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Phone Genie Benchmark</h1>
  <p>{esc(summary.get('bundle_dir'))} | config={esc(summary.get('config'))} | warmup={summary.get('warmup_runs')} run(s), excluded from measured rows</p>
  <table>
    <thead><tr><th>Run</th><th>Backend</th><th>Status</th><th>Wall</th><th>Init</th><th>Query</th><th>Prompt tok/s</th><th>Generated</th><th>Decode tok/s</th><th>Profile</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def write_remote_file(adb: Path, serial: str | None, remote_path: str, content: str) -> None:
    completed = adb_shell_stdin(adb, serial, f"cat > {sh_quote(remote_path)}", content, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to write remote prompt: {completed.stderr or completed.stdout}")


def pull_remote_profile(adb: Path, serial: str | None, remote_path: str, local_path: Path) -> bool:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["pull", remote_path, str(local_path)])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    return result.returncode == 0 and local_path.exists()


def push_file(adb: Path, serial: str | None, local_path: Path, remote_path: str) -> None:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["push", str(local_path), remote_path])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to push {local_path} to {remote_path}: {result.stderr or result.stdout}")


def adb_shell(
    adb: Path,
    serial: str | None,
    command: str,
    *,
    timeout: int,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["shell", command])
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result


def adb_shell_stdin(
    adb: Path,
    serial: str | None,
    command: str,
    input_text: str,
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["shell", "sh", "-c", command])
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def discover_adb(explicit: Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.getenv("ADB")
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_ADB)
    found = shutil.which("adb")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("adb not found. Pass --adb or set ADB.")


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    return (REPO_ROOT / "results" / f"genie_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()


def first_error_line(*texts: str) -> str:
    for text in texts:
        for line in str(text or "").splitlines():
            clean = line.strip()
            if clean:
                return clean[:500]
    return ""


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sh_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def format_number(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def format_ms(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{number / 1000.0:.2f} s" if number >= 1000 else f"{number:.1f} ms"


def format_us(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return format_ms(number / 1000.0)


def path_link(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    return f'<a href="{esc(path.resolve().as_uri())}"><code>{esc(path.name)}</code></a>' if path.exists() else ""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
