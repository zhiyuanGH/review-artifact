#!/usr/bin/env python3
"""Run a controlled llama-cli prompt/output length sweep on a phone.

The script reuses an existing MOBench runtime config to get adb, model, and
llama.cpp paths, then varies synthetic prompt length and max generated tokens.
It records both llama.cpp reported prompt/generation throughput and end-to-end
wall time, so fixed overhead can be separated from decode work.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.llm_tau import LlamaCliClient, parse_llama_perf_timings  # noqa: E402


DEFAULT_METHOD_CSV = REPO_ROOT / "results" / "benchmark" / "method_time_comparison.csv"
DEFAULT_MODEL_ID = "llama_3_2_1b_instruct_q4_0"
TIMING_RE = re.compile(
    r"\[\s*Prompt:\s*(?P<prompt>[0-9.]+)\s*t/s\s*\|\s*Generation:\s*(?P<generation>[0-9.]+)\s*t/s\s*\]"
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
    parser.add_argument(
        "--llama-device",
        action="append",
        default=None,
        help="llama.cpp device to test. May repeat. Default: HTP0.",
    )
    parser.add_argument("--prompt-chars", default="512,2048,8192")
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
        "--min-decode-token-ratio",
        type=float,
        default=0.8,
        help="Minimum llama.cpp decode token ratio required for a valid output length.",
    )
    parser.add_argument(
        "--min-output-word-ratio",
        type=float,
        default=0.5,
        help="Fallback minimum generated word ratio when llama.cpp decode token counts are unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/llama_cli_length_sweep_<timestamp>.",
    )
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected a non-empty integer list, got {text!r}")
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def infer_runtime_config(args: argparse.Namespace) -> Path:
    if args.runtime_config:
        return args.runtime_config

    matches = [
        row
        for row in read_csv(args.method_csv)
        if row.get("method_id") == args.source_method
        and row.get("device") == args.device
        and row.get("model_id") == args.model_id
        and row.get("status_note") == "ok"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source row for {args.source_method}/{args.device}/{args.model_id}, "
            f"got {len(matches)}"
        )
    run_dir = Path(matches[0]["run_dir"])
    manifest_path = run_dir / "execution" / "EXECUTION_MANIFEST.json"
    manifest = read_json(manifest_path)
    runs = manifest.get("runs") or []
    if not runs:
        raise ValueError(f"No execution runs in {manifest_path}")
    runtime_config = Path(runs[0]["runtime_config"])
    if not runtime_config.exists():
        raise FileNotFoundError(f"Runtime config not found: {runtime_config}")
    return runtime_config


def base_llama_config(runtime_config: Path) -> dict[str, Any]:
    config = read_json(runtime_config)
    llm = config.get("llm") or {}
    candidates = []
    if isinstance(llm.get("llama_cli"), dict):
        candidates.append(llm["llama_cli"])
    for value in (llm.get("llama_cli_overrides") or {}).values():
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError(f"No llm.llama_cli config found in {runtime_config}")
    cfg = dict(candidates[0])
    cfg["json_schema"] = ""
    return cfg


def device_config(base_cfg: dict[str, Any], llama_device: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["llama_device"] = llama_device
    cfg["device"] = llama_device
    cfg["ngl"] = 0 if llama_device == "none" else int(cfg.get("ngl", 99) or 99)
    cfg["hexagon_env"] = llama_device.startswith("HTP")
    cfg["warmup_runs"] = args.warmup_runs
    cfg["json_schema"] = ""
    if args.timeout is not None:
        cfg["llama_timeout"] = args.timeout
        cfg["timeout_seconds"] = args.timeout
    if args.ctx_size is not None:
        cfg["ctx_size"] = args.ctx_size
    if args.ubatch_size is not None:
        cfg["ubatch_size"] = args.ubatch_size
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.cpu_mask is not None:
        cfg["cpu_mask"] = args.cpu_mask
    if args.extra_arg:
        cfg["extra_args"] = [*(cfg.get("extra_args") or []), *args.extra_arg]
    return cfg


def synthetic_prompt(target_chars: int, max_tokens: int) -> list[dict[str, str]]:
    system = (
        "You are a deterministic benchmark writer. Produce a long fluent English paragraph. "
        "Never answer with a single word. Do not use lists or markdown."
    )
    seed = (
        "This is synthetic benchmark context for an on-device agent workflow. "
        "It intentionally contains repeated structured facts so prefill work scales with prompt length. "
        "Fields: user_id=user_bench, order_id=order_bench, reservation_id=reservation_bench, "
        "status=pending, requested_action=validate_and_summarize. "
    )
    repeated = (seed * ((target_chars // len(seed)) + 2))[:target_chars]
    user = (
        "After the benchmark context, write one continuous English paragraph of at least 120 words "
        "about imaginary mobile AI benchmark observations. Do not copy the context. "
        "Keep writing until the generation budget ends.\n\n"
        "Benchmark context begins:\n"
        f"{repeated}\n"
        "Benchmark context ends.\n\n"
        "Paragraph:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def rendered_prompt_chars(messages: list[dict[str, str]]) -> int:
    return len("\n\n".join(message["content"] for message in messages if message.get("role") != "system"))


def last_bracket_timing(text: str) -> dict[str, float]:
    matches = list(TIMING_RE.finditer(text or ""))
    if not matches:
        return {}
    match = matches[-1]
    return {
        "prompt_tps_bracket": float(match.group("prompt")),
        "generation_tps_bracket": float(match.group("generation")),
    }


def output_word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def clean_generated_text(text: str) -> str:
    cleaned = re.sub(r"\n\[ Prompt:[\s\S]*$", "", text or "").strip()
    if "\n\n" in cleaned:
        cleaned = cleaned.rsplit("\n\n", 1)[-1].strip()
    cleaned = re.sub(r"^>\s*", "", cleaned).strip()
    return cleaned


def validate_output_length(
    *,
    text: str,
    timings: dict[str, Any],
    max_tokens: int,
    args: argparse.Namespace,
) -> tuple[bool, str, int, float | str]:
    ratio = float(getattr(args, "min_decode_token_ratio", 0.8))
    min_tokens = max(1, int(max_tokens * ratio))
    decode_tokens = timings.get("decode_tokens")
    if decode_tokens in ("", None):
        min_words = max(1, int(max_tokens * float(getattr(args, "min_output_word_ratio", 0.5))))
        words = output_word_count(text)
        if words >= min_words or len(text.strip()) >= max_tokens:
            return True, "fallback_words", min_tokens, ""
        return False, "missing_decode_tokens_short_output", min_tokens, ""
    try:
        decode_count = int(decode_tokens)
    except (TypeError, ValueError):
        return False, "invalid_decode_tokens", min_tokens, ""
    decode_ratio = decode_count / float(max_tokens) if max_tokens else 0.0
    if decode_count < min_tokens:
        return False, "short_decode", min_tokens, round(decode_ratio, 4)
    if len(text.strip()) == 0:
        return False, "empty_output", min_tokens, round(decode_ratio, 4)
    return True, "", min_tokens, round(decode_ratio, 4)


def run_case(
    *,
    base_cfg: dict[str, Any],
    llama_device: str,
    prompt_chars: int,
    max_tokens: int,
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = device_config(base_cfg, llama_device, args)
    cfg["max_tokens"] = max_tokens
    client = LlamaCliClient.from_config(cfg)
    messages = synthetic_prompt(prompt_chars, max_tokens)
    started = time.time()
    status = "ok"
    error = ""
    response: dict[str, Any] = {}
    try:
        response = client.chat(messages)
    except Exception as exc:
        status = "error"
        error = str(exc)
    wall_ms = (time.time() - started) * 1000.0
    stdout = str(response.get("stdout") or "")
    stderr = str(response.get("stderr") or "")
    timings = parse_llama_perf_timings(stdout + "\n" + stderr)
    bracket = last_bracket_timing(stdout + "\n" + stderr)
    stdout_error = "Error:" in stdout or "Failed to initialize" in stdout or status != "ok"
    output_text = clean_generated_text(str(response.get("text") or ""))
    length_ok, length_error, min_decode_tokens, decode_token_ratio = validate_output_length(
        text=output_text,
        timings=timings,
        max_tokens=max_tokens,
        args=args,
    )
    return {
        "status": status,
        "stdout_error": stdout_error,
        "error": error,
        "device_serial": cfg.get("device_serial", ""),
        "model_path": cfg.get("model_path", ""),
        "llama_device": llama_device,
        "ngl": cfg.get("ngl", ""),
        "hexagon_env": cfg.get("hexagon_env", ""),
        "prompt_chars_target": prompt_chars,
        "prompt_chars_rendered": rendered_prompt_chars(messages),
        "max_tokens": max_tokens,
        "repeat": repeat,
        "wall_ms": round(wall_ms, 3),
        "returncode": response.get("returncode", ""),
        "prompt_tps": bracket.get("prompt_tps_bracket", timings.get("prompt_per_second", "")),
        "generation_tps": bracket.get("generation_tps_bracket", timings.get("decode_per_second", "")),
        "prompt_ms": timings.get("prompt_ms", ""),
        "decode_ms": timings.get("decode_ms", ""),
        "prompt_tokens": timings.get("prompt_tokens", ""),
        "decode_tokens": timings.get("decode_tokens", ""),
        "min_decode_tokens": min_decode_tokens,
        "decode_token_ratio": decode_token_ratio,
        "output_chars": len(output_text),
        "output_word_count": output_word_count(output_text),
        "output_length_ok": length_ok,
        "output_length_error": length_error,
        "output_preview": output_text[:160].replace("\n", "\\n"),
        "stdout_tail": stdout[-800:].replace("\n", "\\n"),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["llama_device"], row["prompt_chars_target"], row["max_tokens"]) for row in rows})
    summary: list[dict[str, Any]] = []
    for llama_device, prompt_chars, max_tokens in keys:
        items = [
            row
            for row in rows
            if row["llama_device"] == llama_device
            and row["prompt_chars_target"] == prompt_chars
            and row["max_tokens"] == max_tokens
        ]
        ok = [
            row
            for row in items
            if row["status"] == "ok" and not row["stdout_error"] and row.get("output_length_ok") is True
        ]
        wall = [float(row["wall_ms"]) for row in items]
        prompt_tps = [float(row["prompt_tps"]) for row in ok if row["prompt_tps"] != ""]
        generation_tps = [float(row["generation_tps"]) for row in ok if row["generation_tps"] != ""]
        decode_seconds = [max_tokens / value for value in generation_tps if value > 0]
        decode_share = [
            decode_s / (float(row["wall_ms"]) / 1000.0)
            for row, decode_s in zip(
                [item for item in ok if item["generation_tps"] != "" and float(item["generation_tps"]) > 0],
                decode_seconds,
            )
            if float(row["wall_ms"]) > 0
        ]
        summary.append(
            {
                "llama_device": llama_device,
                "prompt_chars_target": prompt_chars,
                "max_tokens": max_tokens,
                "runs": len(items),
                "ok_runs": len(ok),
                "error_runs": len(items) - len(ok),
                "wall_ms_mean": round(mean(wall), 3) if wall else "",
                "wall_ms_median": round(median(wall), 3) if wall else "",
                "prompt_tps_mean": round(mean(prompt_tps), 3) if prompt_tps else "",
                "generation_tps_mean": round(mean(generation_tps), 3) if generation_tps else "",
                "generation_tps_median": round(median(generation_tps), 3) if generation_tps else "",
                "decode_seconds_est_mean": round(mean(decode_seconds), 3) if decode_seconds else "",
                "decode_wall_share_est_mean": round(mean(decode_share), 4) if decode_share else "",
            }
        )
    return summary


def write_markdown(path: Path, rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], runtime_config: Path) -> None:
    lines = [
        "# llama-cli length sweep",
        "",
        f"Runtime config: `{runtime_config}`",
        f"Cases: {len(rows)}",
        "",
        "| Backend | Prompt chars | Max tokens | Runs | OK | Errors | Wall ms | Prompt t/s | Generation t/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['llama_device']} | {row['prompt_chars_target']} | {row['max_tokens']} | "
            f"{row['runs']} | {row['ok_runs']} | {row['error_runs']} | {row['wall_ms_mean']} | "
            f"{row['prompt_tps_mean']} | {row['generation_tps_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompt_chars_list = parse_int_list(args.prompt_chars)
    max_tokens_list = parse_int_list(args.max_tokens)
    llama_devices = args.llama_device or ["HTP0"]
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / f"llama_cli_length_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = infer_runtime_config(args)
    base_cfg = base_llama_config(runtime_config)
    rows: list[dict[str, Any]] = []
    for llama_device in llama_devices:
        for prompt_chars in prompt_chars_list:
            for max_tokens in max_tokens_list:
                for repeat in range(args.repeats):
                    print(
                        f"[run] backend={llama_device} prompt_chars={prompt_chars} "
                        f"max_tokens={max_tokens} repeat={repeat}",
                        flush=True,
                    )
                    row = run_case(
                        base_cfg=base_cfg,
                        llama_device=llama_device,
                        prompt_chars=prompt_chars,
                        max_tokens=max_tokens,
                        repeat=repeat,
                        args=args,
                    )
                    rows.append(row)
                    print(
                        f"      status={row['status']} wall={row['wall_ms']}ms "
                        f"prompt_tps={row['prompt_tps']} generation_tps={row['generation_tps']} "
                        f"error={row['stdout_error']}",
                        flush=True,
                    )

    summary_rows = summarize(rows)
    result_csv = output_dir / "length_sweep_results.csv"
    summary_csv = output_dir / "length_sweep_summary.csv"
    summary_json = output_dir / "length_sweep_summary.json"
    summary_md = output_dir / "LENGTH_SWEEP_SUMMARY.md"
    write_csv(result_csv, rows)
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "runtime_config": str(runtime_config),
                "result_csv": str(result_csv),
                "summary_csv": str(summary_csv),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_md, rows, summary_rows, runtime_config)

    print()
    print(f"wrote {result_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    print()
    print("| Backend | Prompt chars | Max tokens | Runs | OK | Errors | Wall ms | Prompt t/s | Generation t/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        print(
            f"| {row['llama_device']} | {row['prompt_chars_target']} | {row['max_tokens']} | "
            f"{row['runs']} | {row['ok_runs']} | {row['error_runs']} | {row['wall_ms_mean']} | "
            f"{row['prompt_tps_mean']} | {row['generation_tps_mean']} |"
        )


if __name__ == "__main__":
    main()
