"""Benchmark phone-side GGUF models across CPU, GPU, and NPU llama.cpp backends."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADB = Path(os.environ.get("ADB") or shutil.which("adb") or "adb")
DEFAULT_REMOTE_DIR = "/data/local/tmp/llama.cpp"
DEFAULT_GGUF_DIR = "/data/local/tmp/gguf"
DEFAULT_SDCARD_SEARCH_ROOTS = ["/storage/emulated/0"]

BACKENDS: dict[str, dict[str, Any]] = {
    "cpu": {"device": "none", "ngl": 0, "hexagon_ndev": 0},
    "gpu": {"device": "GPUOpenCL", "ngl": 99, "hexagon_ndev": 0},
    "npu": {"device": "HTP0", "ngl": 99, "hexagon_ndev": 1},
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adb = discover_adb(args.adb)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] output_dir={output_dir}", flush=True)
    print(f"[bench] adb={adb}", flush=True)

    if args.move_sdcard_gguf:
        moved = move_sdcard_ggufs(adb, args)
        write_json(output_dir / "move_sdcard_gguf.json", moved)
        print(f"[bench] moved/copied {moved.get('moved_count', 0)} sdcard GGUF file(s)", flush=True)

    models = resolve_models(adb, args)
    write_json(output_dir / "models.json", [model.to_json() for model in models])
    if args.list:
        for model in models:
            print(f"{model.remote_path}\t{model.size_bytes} bytes")
        return 0
    if not models:
        raise RuntimeError(f"No GGUF models found under {args.gguf_dir}")

    backends = parse_backends(args.backends)
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for backend_name in backends:
            backend = BACKENDS[backend_name]
            label = f"{model_index:02d}_{safe_name(model.name)}_{backend_name}"
            print(f"[bench] {model.name} backend={backend_name}", flush=True)
            run_dir = output_dir / "runs" / label
            run_dir.mkdir(parents=True, exist_ok=True)
            result = run_benchmark(adb, args, model, backend_name, backend, run_dir)
            rows.extend(result["rows"])
            write_json(run_dir / "result.json", result)
            print(
                f"[bench] {model.name} {backend_name} status={result['status']} "
                f"wall_s={result['wall_ms'] / 1000.0:.2f}",
                flush=True,
            )

    summary = {
        "created_at_utc": utc_now(),
        "adb": str(adb),
        "adb_serial": args.adb_serial or "",
        "remote_dir": args.remote_dir,
        "gguf_dir": args.gguf_dir,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "threads": args.threads,
        "cpu_mask": args.cpu_mask,
        "ubatch_size": args.ubatch_size,
        "timeout_seconds": args.timeout,
        "warmup_runs": args.warmup_runs,
        "models": [model.to_json() for model in models],
        "backends": backends,
        "rows": rows,
        "best_by_model_test": best_by_model_test(rows),
    }
    write_json(output_dir / "benchmark_summary.json", summary)
    write_csv(output_dir / "benchmark_summary.csv", rows)
    (output_dir / "benchmark_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    (output_dir / "benchmark_summary.html").write_text(render_html(summary), encoding="utf-8")
    print(f"[bench] wrote {output_dir / 'benchmark_summary.html'}", flush=True)
    return 0 if all(row.get("status") == "ok" for row in rows) else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark all phone-side GGUF models in /data/local/tmp/gguf across "
            "CPU, GPUOpenCL, and HTP0/NPU using llama.cpp llama-bench."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB if DEFAULT_ADB.exists() else None, help="adb executable.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb serial.")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Phone llama.cpp directory.")
    parser.add_argument("--gguf-dir", default=DEFAULT_GGUF_DIR, help="Phone directory containing GGUF models.")
    parser.add_argument("--models", default="", help="Comma-separated remote GGUF paths or basenames. Empty means all under --gguf-dir.")
    parser.add_argument("--backends", default="cpu,gpu,npu", help="Comma-separated backends: cpu,gpu,npu.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to results/gguf_backend_benchmark_<timestamp>.")
    parser.add_argument("--prompt-tokens", type=int, default=16, help="llama-bench prompt tokens.")
    parser.add_argument("--decode-tokens", type=int, default=8, help="llama-bench decode tokens.")
    parser.add_argument("--repetitions", type=int, default=1, help="llama-bench repetitions.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup llama-bench executions before the measured run; use 0 for cold-start timing.")
    parser.add_argument("--threads", type=int, default=6, help="CPU helper threads.")
    parser.add_argument("--cpu-mask", default="0xfc", help="CPU mask.")
    parser.add_argument("--ubatch-size", type=int, default=1024, help="llama-bench ubatch size.")
    parser.add_argument("--timeout", type=int, default=900, help="Per model/backend timeout seconds.")
    parser.add_argument("--max-models", type=int, default=0, help="Limit number of models; 0 means no limit.")
    parser.add_argument("--include-zero-size", action="store_true", help="Include zero-byte GGUF files.")
    parser.add_argument("--move-sdcard-gguf", action="store_true", help="Move GGUF files from /sdcard-style storage into --gguf-dir before benchmarking.")
    parser.add_argument("--sdcard-root", action="append", default=[], help="Root to search when --move-sdcard-gguf is enabled. Can be repeated.")
    parser.add_argument("--list", action="store_true", help="List selected models and exit.")
    return parser.parse_args(argv)


class Model:
    def __init__(self, remote_path: str, size_bytes: int):
        self.remote_path = remote_path
        self.size_bytes = size_bytes
        self.name = remote_path.rstrip("/").rsplit("/", 1)[-1]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "remote_path": self.remote_path, "size_bytes": self.size_bytes}


def run_benchmark(
    adb: Path,
    args: argparse.Namespace,
    model: Model,
    backend_name: str,
    backend: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    command = build_bench_command(args, model.remote_path, backend)
    warmup_results = []
    warmup_started = time.time()
    for index in range(max(0, int(args.warmup_runs))):
        warmup_command = build_bench_command(args, model.remote_path, backend, repetitions=1)
        item_started = time.time()
        item: dict[str, Any] = {
            "index": index,
            "command": warmup_command,
            "started_at_utc": datetime.fromtimestamp(item_started, timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            completed = adb_shell(adb, args.adb_serial, warmup_command, timeout=args.timeout)
            item.update(
                {
                    "status": "ok" if completed.returncode == 0 else "error",
                    "returncode": completed.returncode,
                    "wall_ms": (time.time() - item_started) * 1000.0,
                    "rows": parse_llama_bench_rows(completed.stdout),
                    "stdout_tail": (completed.stdout or "")[-4000:],
                    "stderr_tail": (completed.stderr or "")[-4000:],
                }
            )
        except subprocess.TimeoutExpired as exc:
            item.update(
                {
                    "status": "timeout",
                    "returncode": None,
                    "wall_ms": (time.time() - item_started) * 1000.0,
                    "error": f"Timed out after {args.timeout} seconds.",
                    "stdout_tail": str(exc.stdout or "")[-4000:],
                    "stderr_tail": str(exc.stderr or "")[-4000:],
                }
            )
        except Exception as exc:
            item.update(
                {
                    "status": "error",
                    "returncode": None,
                    "wall_ms": (time.time() - item_started) * 1000.0,
                    "error": str(exc),
                }
            )
        warmup_results.append(item)
    warmup_finished = time.time()
    started = time.time()
    stdout = ""
    stderr = ""
    returncode = -1
    status = "error"
    error = ""
    try:
        completed = adb_shell(adb, args.adb_serial, command, timeout=args.timeout)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        status = "ok" if completed.returncode == 0 else "error"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        status = "timeout"
        error = f"Timed out after {args.timeout} seconds."
    except Exception as exc:
        status = "error"
        error = str(exc)
    finished = time.time()
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    parsed_rows = parse_llama_bench_rows(stdout)
    rows = [
        {
            "status": status if parsed_rows else ("no_rows" if status == "ok" else status),
            "model": model.name,
            "model_path": model.remote_path,
            "model_size_bytes": model.size_bytes,
            "backend": backend_name,
            "device": backend["device"],
            "ngl": backend["ngl"],
            "hexagon_ndev": backend["hexagon_ndev"],
            "test": row.get("test", ""),
            "tokens_per_second": row.get("tokens_per_second"),
            "raw_speed": row.get("raw_speed", ""),
            "wall_ms": (finished - started) * 1000.0,
            "returncode": returncode,
            "error": error,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        for row in parsed_rows
    ]
    if not rows:
        rows.append(
            {
                "status": status,
                "model": model.name,
                "model_path": model.remote_path,
                "model_size_bytes": model.size_bytes,
                "backend": backend_name,
                "device": backend["device"],
                "ngl": backend["ngl"],
                "hexagon_ndev": backend["hexagon_ndev"],
                "test": "",
                "tokens_per_second": None,
                "raw_speed": "",
                "wall_ms": (finished - started) * 1000.0,
                "returncode": returncode,
                "error": error or first_error_line(stderr, stdout),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
    return {
        "status": status,
        "warmup": {
            "runs": warmup_results,
            "run_count": len(warmup_results),
            "wall_ms": (warmup_finished - warmup_started) * 1000.0,
            "status": "ok" if all(item.get("status") == "ok" for item in warmup_results) else ("skipped" if not warmup_results else "error"),
        },
        "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.fromtimestamp(finished, timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_ms": (finished - started) * 1000.0,
        "model": model.to_json(),
        "backend": backend_name,
        "backend_config": backend,
        "command": command,
        "returncode": returncode,
        "error": error,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "rows": rows,
    }


def build_bench_command(
    args: argparse.Namespace,
    model_path: str,
    backend: dict[str, Any],
    *,
    repetitions: int | None = None,
) -> str:
    remote = sh_quote(args.remote_dir)
    lib = sh_quote(f"{args.remote_dir}/lib")
    model = sh_quote(model_path)
    hexagon_verbose = " GGML_HEXAGON_VERBOSE=1" if backend["hexagon_ndev"] else ""
    repetition_count = int(args.repetitions if repetitions is None else repetitions)
    repetition = f" -r {repetition_count}" if repetition_count > 0 else ""
    return (
        f"cd {remote} && "
        f"GGML_HEXAGON_NDEV={int(backend['hexagon_ndev'])}{hexagon_verbose} "
        f"LD_LIBRARY_PATH={lib} ADSP_LIBRARY_PATH={lib} "
        "./bin/llama-bench "
        f"--device {sh_quote(backend['device'])} "
        "--mmap 0 "
        f"-m {model} "
        "--poll 1000 "
        f"-t {int(args.threads)} "
        f"--cpu-mask {sh_quote(args.cpu_mask)} "
        "--cpu-strict 1 "
        f"--ubatch-size {int(args.ubatch_size)} "
        "-fa 1 "
        f"-ngl {int(backend['ngl'])} "
        f"-p {int(args.prompt_tokens)} "
        f"-n {int(args.decode_tokens)}"
        f"{repetition}"
    )


def resolve_models(adb: Path, args: argparse.Namespace) -> list[Model]:
    if args.models.strip():
        requested = [item.strip() for item in args.models.split(",") if item.strip()]
        paths = [item if item.startswith("/") else f"{args.gguf_dir.rstrip('/')}/{item}" for item in requested]
    else:
        command = f"find {sh_quote(args.gguf_dir)} -maxdepth 1 -type f -iname '*.gguf' 2>/dev/null | sort"
        completed = adb_shell(adb, args.adb_serial, command, timeout=60)
        paths = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    models = []
    for path in paths:
        size = remote_size(adb, args.adb_serial, path)
        if size == 0 and not args.include_zero_size:
            continue
        models.append(Model(path, size))
    models.sort(key=lambda item: item.name.lower())
    if args.max_models > 0:
        models = models[: args.max_models]
    return models


def remote_size(adb: Path, serial: str | None, path: str) -> int:
    completed = adb_shell(adb, serial, f"wc -c < {sh_quote(path)} 2>/dev/null || echo 0", timeout=30)
    text = (completed.stdout or "0").strip().splitlines()[-1]
    try:
        return int(text)
    except ValueError:
        return 0


def move_sdcard_ggufs(adb: Path, args: argparse.Namespace) -> dict[str, Any]:
    roots = args.sdcard_root or DEFAULT_SDCARD_SEARCH_ROOTS
    root_args = " ".join(sh_quote(root) for root in roots)
    script = f"""
set -u
dst_dir={sh_quote(args.gguf_dir)}
mkdir -p "$dst_dir"
tmp_list=/data/local/tmp/wfgen_sdcard_gguf_list_$$.txt
find {root_args} -maxdepth 4 -type f -iname '*.gguf' 2>/dev/null | sort -u > "$tmp_list"
moved=0
skipped=0
failed=0
while IFS= read -r src; do
  [ -n "$src" ] || continue
  case "$src" in
    "$dst_dir"/*) skipped=$((skipped+1)); echo "SKIP already in target $src"; continue ;;
  esac
  base="$(basename "$src")"
  dst="$dst_dir/$base"
  src_size="$(wc -c < "$src" 2>/dev/null || echo 0)"
  if [ -L "$dst" ]; then
    src_real="$(readlink -f "$src" 2>/dev/null || echo "$src")"
    dst_real="$(readlink -f "$dst" 2>/dev/null || echo "$dst")"
    if [ "$src_real" = "$dst_real" ]; then
      tmp="$dst_dir/.${{base}}.wfgen_move_$$"
      if cp -p "$src" "$tmp" 2>/dev/null; then
        tmp_size="$(wc -c < "$tmp" 2>/dev/null || echo -1)"
        if [ "$src_size" = "$tmp_size" ]; then
          rm -f "$dst" && mv "$tmp" "$dst" && rm -f "$src" && moved=$((moved+1))
          echo "REPLACE_SYMLINK_WITH_FILE $src -> $dst"
          continue
        fi
        rm -f "$tmp"
      fi
      failed=$((failed+1))
      echo "FAIL replace symlink $src -> $dst"
      continue
    fi
  fi
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    dst_size="$(wc -c < "$dst" 2>/dev/null || echo -1)"
    if [ "$src_size" = "$dst_size" ]; then
      rm -f "$src" && moved=$((moved+1)) && echo "REMOVE duplicate source $src -> $dst"
      continue
    fi
    echo "FAIL target exists with different size $src -> $dst"
    failed=$((failed+1))
    continue
  fi
  if mv "$src" "$dst" 2>/dev/null; then
    moved=$((moved+1))
    echo "MOVE $src -> $dst"
    continue
  fi
  if cp -p "$src" "$dst" 2>/dev/null; then
    new_size="$(wc -c < "$dst" 2>/dev/null || echo -1)"
    if [ "$src_size" = "$new_size" ]; then
      rm -f "$src" && moved=$((moved+1)) && echo "COPY_REMOVE $src -> $dst"
    else
      rm -f "$dst"
      failed=$((failed+1))
      echo "FAIL size mismatch $src -> $dst"
    fi
  else
    failed=$((failed+1))
    echo "FAIL copy $src -> $dst"
  fi
done < "$tmp_list"
rm -f "$tmp_list"
echo "__WFGEN_COUNTS__ moved=$moved skipped=$skipped failed=$failed"
"""
    completed = adb_shell_stdin(adb, args.adb_serial, script, timeout=max(args.timeout, 900))
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "moved_count": parse_count(completed.stdout, "moved"),
        "skipped_count": parse_count(completed.stdout, "skipped"),
        "failed_count": parse_count(completed.stdout, "failed"),
        "script": script,
    }


def parse_llama_bench_rows(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text.startswith("|"):
            continue
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        if len(cells) < 6 or not any(re.search(r"\b(pp|tg)\d+", cell) for cell in cells):
            continue
        speed_cell = cells[-1] if cells else ""
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", speed_cell)
        speed = float(match.group(1)) if match else None
        test = next((cell for cell in cells if re.search(r"\b(pp|tg)\d+", cell)), "")
        rows.append({"test": test, "raw_speed": speed_cell, "tokens_per_second": speed})
    return rows


def parse_backends(raw: str) -> list[str]:
    backends = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [backend for backend in backends if backend not in BACKENDS]
    if unknown:
        raise ValueError(f"Unknown backend(s): {', '.join(unknown)}")
    return backends or ["cpu", "gpu", "npu"]


def best_by_model_test(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("tokens_per_second") is None:
            continue
        key = (str(row.get("model")), str(row.get("test")))
        if key not in best or float(row["tokens_per_second"]) > float(best[key]["tokens_per_second"]):
            best[key] = row
    return list(best.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status",
        "model",
        "backend",
        "device",
        "ngl",
        "test",
        "tokens_per_second",
        "wall_ms",
        "returncode",
        "error",
        "model_path",
        "stdout",
        "stderr",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phone GGUF Backend Benchmark",
        "",
        f"Created UTC: `{summary['created_at_utc']}`",
        f"GGUF dir: `{summary['gguf_dir']}`",
        f"Warmup runs before measured wall: `{summary.get('warmup_runs', 0)}`",
        "",
        "| Model | Backend | Device | Test | tok/s | Status |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model")),
                    str(row.get("backend")),
                    str(row.get("device")),
                    str(row.get("test") or ""),
                    format_number(row.get("tokens_per_second")),
                    str(row.get("status")),
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
            f"<td><code>{esc(row.get('model'))}</code></td>"
            f"<td>{esc(row.get('backend'))}</td>"
            f"<td>{esc(row.get('device'))}</td>"
            f"<td>{esc(row.get('test'))}</td>"
            f"<td>{esc(format_number(row.get('tokens_per_second')))}</td>"
            f"<td>{esc(row.get('status'))}</td>"
            f"<td>{esc(format_ms(row.get('wall_ms')))}</td>"
            f"<td>{path_link(row.get('stdout'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Phone GGUF Backend Benchmark</title>
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
  <h1>Phone GGUF Backend Benchmark</h1>
  <p>{esc(summary.get('gguf_dir'))} | p={summary.get('prompt_tokens')} | n={summary.get('decode_tokens')} | warmup={summary.get('warmup_runs', 0)} run(s), excluded from wall</p>
  <table>
    <thead><tr><th>Model</th><th>Backend</th><th>Device</th><th>Test</th><th>tok/s</th><th>Status</th><th>Wall</th><th>Stdout</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def adb_shell(adb: Path, serial: str | None, command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["shell", command])
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def adb_shell_stdin(adb: Path, serial: str | None, script: str, timeout: int) -> subprocess.CompletedProcess[str]:
    cmd = [str(adb)]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(["shell", "sh"])
    return subprocess.run(cmd, input=script, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


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


def parse_count(text: str, key: str) -> int:
    match = re.search(rf"\b{re.escape(key)}=(\d+)", text)
    return int(match.group(1)) if match else 0


def first_error_line(*texts: str) -> str:
    for text in texts:
        for line in str(text or "").splitlines():
            clean = line.strip()
            if clean:
                return clean[:500]
    return ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit:
        return explicit.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "results" / f"gguf_backend_benchmark_{stamp}").resolve()


def sh_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")
    return clean or "model"


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


def path_link(value: Any) -> str:
    if not value:
        return ""
    path = Path(str(value))
    return f'<a href="{esc(path.resolve().as_uri())}"><code>{esc(path.name)}</code></a>'


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
