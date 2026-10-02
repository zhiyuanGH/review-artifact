#!/usr/bin/env python3
"""Measure paired llama-cli speedup from running two backends concurrently.

This script reuses the length-sweep benchmark helpers, but for each prompt
length and max-token setting it runs two llama-cli invocations:

1. sequentially, left backend followed by right backend
2. concurrently, one invocation on each backend

The resulting speedup is sequential wall time divided by parallel wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_llama_cli_length_sweep import (  # noqa: E402
    DEFAULT_METHOD_CSV,
    DEFAULT_MODEL_ID,
    base_llama_config,
    infer_runtime_config,
    parse_int_list,
    run_case,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-csv", type=Path, default=DEFAULT_METHOD_CSV)
    parser.add_argument("--source-method", default="wave_opt")
    parser.add_argument("--device", default="<device-serial>", help="Phone serial in method_time_comparison.csv.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Optional explicit workflow runtime config JSON to reuse.",
    )
    parser.add_argument("--left-device", default="none", help="First llama.cpp backend. Default: CPU.")
    parser.add_argument("--right-device", default="HTP0", help="Second llama.cpp backend. Default: NPU HTP0.")
    parser.add_argument("--prompt-chars", default="512")
    parser.add_argument("--max-tokens", default="64,256")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--ubatch-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--cpu-mask", default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/llama_cli_parallel_length_sweep_<timestamp>.",
    )
    return parser.parse_args()


def backend_base_config(base_cfg: dict[str, Any], label: str, run_token: str) -> dict[str, Any]:
    cfg = dict(base_cfg)
    base_remote = str(cfg.get("remote_prompt_dir") or "/data/local/tmp/wfgen_llama_prompts").rstrip("/")
    safe_label = label.replace("/", "_").replace(" ", "_")
    cfg["remote_prompt_dir"] = f"{base_remote}_parallel_{run_token}_{safe_label}"
    return cfg


def is_ok(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok" and not row.get("stdout_error") and row.get("output_length_ok") is True


def paired_run(
    *,
    base_cfg: dict[str, Any],
    left_device: str,
    right_device: str,
    prompt_chars: int,
    max_tokens: int,
    repeat: int,
    args: argparse.Namespace,
    run_token: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    left_cfg = backend_base_config(base_cfg, left_device, f"{run_token}_{prompt_chars}_{max_tokens}_{repeat}")
    right_cfg = backend_base_config(base_cfg, right_device, f"{run_token}_{prompt_chars}_{max_tokens}_{repeat}")

    print(
        f"[sequential] {left_device}+{right_device} prompt_chars={prompt_chars} "
        f"max_tokens={max_tokens} repeat={repeat}",
        flush=True,
    )
    sequential_started = time.time()
    left_seq = run_case(
        base_cfg=left_cfg,
        llama_device=left_device,
        prompt_chars=prompt_chars,
        max_tokens=max_tokens,
        repeat=repeat,
        args=args,
    )
    right_seq = run_case(
        base_cfg=right_cfg,
        llama_device=right_device,
        prompt_chars=prompt_chars,
        max_tokens=max_tokens,
        repeat=repeat,
        args=args,
    )
    sequential_wall_ms = (time.time() - sequential_started) * 1000.0

    print(
        f"[parallel]   {left_device}+{right_device} prompt_chars={prompt_chars} "
        f"max_tokens={max_tokens} repeat={repeat}",
        flush=True,
    )
    parallel_started = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        left_future = pool.submit(
            run_case,
            base_cfg=left_cfg,
            llama_device=left_device,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            repeat=repeat,
            args=args,
        )
        right_future = pool.submit(
            run_case,
            base_cfg=right_cfg,
            llama_device=right_device,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            repeat=repeat,
            args=args,
        )
        left_parallel = left_future.result()
        right_parallel = right_future.result()
    parallel_wall_ms = (time.time() - parallel_started) * 1000.0

    sequential_component_sum_ms = float(left_seq["wall_ms"]) + float(right_seq["wall_ms"])
    parallel_component_max_ms = max(float(left_parallel["wall_ms"]), float(right_parallel["wall_ms"]))
    speedup = sequential_wall_ms / parallel_wall_ms if parallel_wall_ms > 0 else 0.0
    component_speedup = (
        sequential_component_sum_ms / parallel_component_max_ms if parallel_component_max_ms > 0 else 0.0
    )
    ok = all(is_ok(row) for row in [left_seq, right_seq, left_parallel, right_parallel])

    pair_row = {
        "status": "ok" if ok else "error",
        "left_device": left_device,
        "right_device": right_device,
        "prompt_chars_target": prompt_chars,
        "max_tokens": max_tokens,
        "repeat": repeat,
        "sequential_wall_ms": round(sequential_wall_ms, 3),
        "parallel_wall_ms": round(parallel_wall_ms, 3),
        "speedup": round(speedup, 4),
        "sequential_component_sum_ms": round(sequential_component_sum_ms, 3),
        "parallel_component_max_ms": round(parallel_component_max_ms, 3),
        "component_speedup": round(component_speedup, 4),
        "left_seq_wall_ms": left_seq["wall_ms"],
        "right_seq_wall_ms": right_seq["wall_ms"],
        "left_parallel_wall_ms": left_parallel["wall_ms"],
        "right_parallel_wall_ms": right_parallel["wall_ms"],
        "left_seq_generation_tps": left_seq["generation_tps"],
        "right_seq_generation_tps": right_seq["generation_tps"],
        "left_parallel_generation_tps": left_parallel["generation_tps"],
        "right_parallel_generation_tps": right_parallel["generation_tps"],
        "left_seq_ok": is_ok(left_seq),
        "right_seq_ok": is_ok(right_seq),
        "left_parallel_ok": is_ok(left_parallel),
        "right_parallel_ok": is_ok(right_parallel),
    }

    details: list[dict[str, Any]] = []
    for phase, side, row in [
        ("sequential", "left", left_seq),
        ("sequential", "right", right_seq),
        ("parallel", "left", left_parallel),
        ("parallel", "right", right_parallel),
    ]:
        detail = dict(row)
        detail.update(
            {
                "phase": phase,
                "side": side,
                "pair_left_device": left_device,
                "pair_right_device": right_device,
            }
        )
        details.append(detail)

    print(
        f"      seq={sequential_wall_ms / 1000.0:.2f}s par={parallel_wall_ms / 1000.0:.2f}s "
        f"speedup={speedup:.3f}x ok={ok}",
        flush=True,
    )
    return pair_row, details


def summarize_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["prompt_chars_target"], row["max_tokens"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for prompt_chars, max_tokens in keys:
        items = [
            row
            for row in rows
            if row["prompt_chars_target"] == prompt_chars and row["max_tokens"] == max_tokens
        ]
        ok = [row for row in items if row["status"] == "ok"]
        speedups = [float(row["speedup"]) for row in ok]
        component_speedups = [float(row["component_speedup"]) for row in ok]
        seq_wall = [float(row["sequential_wall_ms"]) for row in ok]
        par_wall = [float(row["parallel_wall_ms"]) for row in ok]
        summary.append(
            {
                "prompt_chars_target": prompt_chars,
                "max_tokens": max_tokens,
                "runs": len(items),
                "ok_runs": len(ok),
                "error_runs": len(items) - len(ok),
                "sequential_wall_ms_mean": round(mean(seq_wall), 3) if seq_wall else "",
                "parallel_wall_ms_mean": round(mean(par_wall), 3) if par_wall else "",
                "speedup_mean": round(mean(speedups), 4) if speedups else "",
                "speedup_median": round(median(speedups), 4) if speedups else "",
                "component_speedup_mean": round(mean(component_speedups), 4) if component_speedups else "",
            }
        )
    return summary


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], runtime_config: Path, args: argparse.Namespace) -> None:
    lines = [
        "# llama-cli parallel length sweep",
        "",
        f"Runtime config: `{runtime_config}`",
        f"Pair: `{args.left_device}` + `{args.right_device}`",
        "",
        "| Prompt chars | Max tokens | Runs | OK | Sequential s | Parallel s | Speedup | Component speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        seq_s = float(row["sequential_wall_ms_mean"]) / 1000.0 if row["sequential_wall_ms_mean"] != "" else ""
        par_s = float(row["parallel_wall_ms_mean"]) / 1000.0 if row["parallel_wall_ms_mean"] != "" else ""
        lines.append(
            f"| {row['prompt_chars_target']} | {row['max_tokens']} | {row['runs']} | {row['ok_runs']} | "
            f"{seq_s:.3f} | {par_s:.3f} | {row['speedup_mean']} | {row['component_speedup_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompt_chars_list = parse_int_list(args.prompt_chars)
    max_tokens_list = parse_int_list(args.max_tokens)
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / f"llama_cli_parallel_length_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = infer_runtime_config(args)
    base_cfg = base_llama_config(runtime_config)
    run_token = str(int(time.time() * 1000))

    pair_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for prompt_chars in prompt_chars_list:
        for max_tokens in max_tokens_list:
            for repeat in range(args.repeats):
                pair_row, details = paired_run(
                    base_cfg=base_cfg,
                    left_device=args.left_device,
                    right_device=args.right_device,
                    prompt_chars=prompt_chars,
                    max_tokens=max_tokens,
                    repeat=repeat,
                    args=args,
                    run_token=run_token,
                )
                pair_rows.append(pair_row)
                detail_rows.extend(details)

    summary_rows = summarize_pair_rows(pair_rows)
    pair_csv = output_dir / "parallel_speedup_pairs.csv"
    detail_csv = output_dir / "parallel_speedup_details.csv"
    summary_csv = output_dir / "parallel_speedup_summary.csv"
    summary_json = output_dir / "parallel_speedup_summary.json"
    summary_md = output_dir / "PARALLEL_SPEEDUP_SUMMARY.md"

    write_csv(pair_csv, pair_rows)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "runtime_config": str(runtime_config),
                "pair_csv": str(pair_csv),
                "detail_csv": str(detail_csv),
                "summary_csv": str(summary_csv),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_md, summary_rows, runtime_config, args)

    print()
    print(f"wrote {pair_csv}")
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    print()
    print("| Prompt chars | Max tokens | Runs | OK | Sequential s | Parallel s | Speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        seq_s = float(row["sequential_wall_ms_mean"]) / 1000.0 if row["sequential_wall_ms_mean"] != "" else 0.0
        par_s = float(row["parallel_wall_ms_mean"]) / 1000.0 if row["parallel_wall_ms_mean"] != "" else 0.0
        print(
            f"| {row['prompt_chars_target']} | {row['max_tokens']} | {row['runs']} | {row['ok_runs']} | "
            f"{seq_s:.3f} | {par_s:.3f} | {row['speedup_mean']} |"
        )


if __name__ == "__main__":
    main()
