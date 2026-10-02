#!/usr/bin/env python3
"""Measure single-U serial baselines against multi-U llama-cli parallelism.

For each max-token setting, the script measures:

* single-call latency on CPU, GPU, and NPU to rank individual U choices
* serial baselines for d independent calls on one U only
* parallel executions for degree 2 and 3 using the fastest U choices

Speedup is computed as:

    serial wall time for d calls on one U / parallel wall time for d calls
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


DEFAULT_BACKENDS = {
    "CPU": "none",
    "GPU": "GPUOpenCL",
    "NPU": "HTP0",
}


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
    parser.add_argument("--prompt-chars", type=int, default=512)
    parser.add_argument("--max-tokens", default="64,256,512")
    parser.add_argument("--degrees", default="2,3")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--ctx-size", type=int, default=None)
    parser.add_argument("--ubatch-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--cpu-mask", default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help=(
            "Backend mapping LABEL=llama_device. May repeat. "
            "Default: CPU=none, GPU=GPUOpenCL, NPU=HTP0."
        ),
    )
    parser.add_argument(
        "--skip-gpu-alt-threshold",
        type=float,
        default=1.5,
        help="GPU is considered slow if it is this many times slower than the fastest single U.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/llama_cli_multi_u_speedup_<timestamp>.",
    )
    return parser.parse_args()


def parse_backends(items: list[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_BACKENDS)
    backends: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --backend LABEL=llama_device, got {item!r}")
        label, device = item.split("=", 1)
        label = label.strip()
        device = device.strip()
        if not label or not device:
            raise ValueError(f"Expected --backend LABEL=llama_device, got {item!r}")
        backends[label] = device
    return backends


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def config_for_slot(base_cfg: dict[str, Any], case_id: str, slot_index: int, label: str) -> dict[str, Any]:
    cfg = dict(base_cfg)
    base_remote = str(cfg.get("remote_prompt_dir") or "/data/local/tmp/wfgen_llama_prompts").rstrip("/")
    cfg["remote_prompt_dir"] = f"{base_remote}_multiu_{safe_name(case_id)}_s{slot_index}_{safe_name(label)}"
    return cfg


def row_ok(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok" and not row.get("stdout_error") and row.get("output_length_ok") is True


def run_slot(
    *,
    base_cfg: dict[str, Any],
    label: str,
    llama_device: str,
    prompt_chars: int,
    max_tokens: int,
    repeat: int,
    args: argparse.Namespace,
    case_id: str,
    slot_index: int,
) -> dict[str, Any]:
    cfg = config_for_slot(base_cfg, case_id, slot_index, label)
    row = run_case(
        base_cfg=cfg,
        llama_device=llama_device,
        prompt_chars=prompt_chars,
        max_tokens=max_tokens,
        repeat=repeat,
        args=args,
    )
    row.update(
        {
            "u_label": label,
            "case_id": case_id,
            "slot_index": slot_index,
        }
    )
    return row


def run_serial_group(
    *,
    base_cfg: dict[str, Any],
    label: str,
    llama_device: str,
    degree: int,
    prompt_chars: int,
    max_tokens: int,
    repeat: int,
    args: argparse.Namespace,
    case_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(
        f"[serial] {label} degree={degree} prompt_chars={prompt_chars} "
        f"max_tokens={max_tokens} repeat={repeat}",
        flush=True,
    )
    started = time.time()
    details = [
        run_slot(
            base_cfg=base_cfg,
            label=label,
            llama_device=llama_device,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            repeat=repeat,
            args=args,
            case_id=case_id,
            slot_index=index,
        )
        for index in range(degree)
    ]
    wall_ms = (time.time() - started) * 1000.0
    ok = all(row_ok(row) for row in details)
    row = {
        "kind": "serial",
        "status": "ok" if ok else "error",
        "config_id": f"serial_{label}",
        "config_labels": label,
        "config_devices": llama_device,
        "degree": degree,
        "prompt_chars_target": prompt_chars,
        "max_tokens": max_tokens,
        "repeat": repeat,
        "wall_ms": round(wall_ms, 3),
        "component_sum_ms": round(sum(float(row["wall_ms"]) for row in details), 3),
        "component_max_ms": round(max(float(row["wall_ms"]) for row in details), 3) if details else "",
        "ok_slots": sum(1 for row in details if row_ok(row)),
        "error_slots": sum(1 for row in details if not row_ok(row)),
    }
    print(f"      wall={wall_ms / 1000.0:.2f}s ok={ok}", flush=True)
    return row, details


def run_parallel_group(
    *,
    base_cfg: dict[str, Any],
    config_id: str,
    labels: list[str],
    devices: list[str],
    prompt_chars: int,
    max_tokens: int,
    repeat: int,
    args: argparse.Namespace,
    case_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    print(
        f"[parallel] {config_id} prompt_chars={prompt_chars} "
        f"max_tokens={max_tokens} repeat={repeat}",
        flush=True,
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(labels)) as pool:
        futures = [
            pool.submit(
                run_slot,
                base_cfg=base_cfg,
                label=label,
                llama_device=device,
                prompt_chars=prompt_chars,
                max_tokens=max_tokens,
                repeat=repeat,
                args=args,
                case_id=case_id,
                slot_index=index,
            )
            for index, (label, device) in enumerate(zip(labels, devices))
        ]
        details = [future.result() for future in futures]
    wall_ms = (time.time() - started) * 1000.0
    ok = all(row_ok(row) for row in details)
    row = {
        "kind": "parallel",
        "status": "ok" if ok else "error",
        "config_id": config_id,
        "config_labels": "+".join(labels),
        "config_devices": "+".join(devices),
        "degree": len(labels),
        "prompt_chars_target": prompt_chars,
        "max_tokens": max_tokens,
        "repeat": repeat,
        "wall_ms": round(wall_ms, 3),
        "component_sum_ms": round(sum(float(row["wall_ms"]) for row in details), 3),
        "component_max_ms": round(max(float(row["wall_ms"]) for row in details), 3) if details else "",
        "ok_slots": sum(1 for row in details if row_ok(row)),
        "error_slots": sum(1 for row in details if not row_ok(row)),
    }
    print(f"      wall={wall_ms / 1000.0:.2f}s ok={ok}", flush=True)
    return row, details


def ranked_labels(single_rows: list[dict[str, Any]]) -> list[str]:
    ok_rows = [row for row in single_rows if row_ok(row)]
    return [
        row["u_label"]
        for row in sorted(ok_rows, key=lambda item: float(item["wall_ms"]))
    ]


def make_parallel_configs(
    *,
    backends: dict[str, str],
    single_rows: list[dict[str, Any]],
    degree: int,
    gpu_alt_threshold: float,
) -> list[tuple[str, list[str], list[str]]]:
    labels_by_speed = ranked_labels(single_rows)
    configs: list[tuple[str, list[str], list[str]]] = []
    if len(labels_by_speed) >= degree:
        labels = labels_by_speed[:degree]
        configs.append((f"top{degree}_" + "_".join(labels), labels, [backends[label] for label in labels]))

    if degree == 3 and {"CPU", "NPU"}.issubset(backends):
        slow_gpu = True
        gpu_row = next((row for row in single_rows if row["u_label"] == "GPU" and row_ok(row)), None)
        fastest_row = min([row for row in single_rows if row_ok(row)], key=lambda row: float(row["wall_ms"]), default=None)
        if gpu_row is not None and fastest_row is not None:
            slow_gpu = float(gpu_row["wall_ms"]) >= gpu_alt_threshold * float(fastest_row["wall_ms"])
        if slow_gpu:
            configs.extend(
                [
                    ("alt_2CPU_1NPU", ["CPU", "CPU", "NPU"], [backends["CPU"], backends["CPU"], backends["NPU"]]),
                    ("alt_2NPU_1CPU", ["NPU", "NPU", "CPU"], [backends["NPU"], backends["NPU"], backends["CPU"]]),
                ]
            )

    deduped: list[tuple[str, list[str], list[str]]] = []
    seen: set[str] = set()
    for config in configs:
        if config[0] not in seen:
            deduped.append(config)
            seen.add(config[0])
    return deduped


def add_speedups(
    group_rows: list[dict[str, Any]],
    backends: dict[str, str],
) -> None:
    for row in group_rows:
        for label in backends:
            row.setdefault(f"speedup_vs_{label}", "")

    serial_lookup = {
        (row["prompt_chars_target"], row["max_tokens"], row["degree"], row["repeat"], row["config_labels"]): row
        for row in group_rows
        if row["kind"] == "serial" and row["status"] == "ok"
    }
    for row in group_rows:
        if row["kind"] != "parallel" or row["status"] != "ok":
            continue
        for label in backends:
            baseline = serial_lookup.get(
                (row["prompt_chars_target"], row["max_tokens"], row["degree"], row["repeat"], label)
            )
            key = f"speedup_vs_{label}"
            if baseline and float(row["wall_ms"]) > 0:
                row[key] = round(float(baseline["wall_ms"]) / float(row["wall_ms"]), 4)
            else:
                row[key] = ""


def summarize_groups(group_rows: list[dict[str, Any]], backends: dict[str, str]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                row["kind"],
                row["config_id"],
                row["config_labels"],
                row["degree"],
                row["prompt_chars_target"],
                row["max_tokens"],
            )
            for row in group_rows
        }
    )
    summary: list[dict[str, Any]] = []
    for kind, config_id, config_labels, degree, prompt_chars, max_tokens in keys:
        items = [
            row
            for row in group_rows
            if row["kind"] == kind
            and row["config_id"] == config_id
            and row["degree"] == degree
            and row["prompt_chars_target"] == prompt_chars
            and row["max_tokens"] == max_tokens
        ]
        ok = [row for row in items if row["status"] == "ok"]
        wall = [float(row["wall_ms"]) for row in ok]
        out: dict[str, Any] = {
            "kind": kind,
            "config_id": config_id,
            "config_labels": config_labels,
            "degree": degree,
            "prompt_chars_target": prompt_chars,
            "max_tokens": max_tokens,
            "runs": len(items),
            "ok_runs": len(ok),
            "error_runs": len(items) - len(ok),
            "wall_ms_mean": round(mean(wall), 3) if wall else "",
            "wall_ms_median": round(median(wall), 3) if wall else "",
        }
        for label in backends:
            key = f"speedup_vs_{label}"
            values = [float(row[key]) for row in ok if row.get(key) not in ("", None)]
            out[f"speedup_vs_{label}_mean"] = round(mean(values), 4) if values else ""
        summary.append(out)
    return summary


def summarize_single(single_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["u_label"], row["prompt_chars_target"], row["max_tokens"]) for row in single_rows})
    summary: list[dict[str, Any]] = []
    for label, prompt_chars, max_tokens in keys:
        items = [
            row
            for row in single_rows
            if row["u_label"] == label and row["prompt_chars_target"] == prompt_chars and row["max_tokens"] == max_tokens
        ]
        ok = [row for row in items if row_ok(row)]
        wall = [float(row["wall_ms"]) for row in ok]
        generation = [float(row["generation_tps"]) for row in ok if row.get("generation_tps") != ""]
        summary.append(
            {
                "u_label": label,
                "prompt_chars_target": prompt_chars,
                "max_tokens": max_tokens,
                "runs": len(items),
                "ok_runs": len(ok),
                "error_runs": len(items) - len(ok),
                "wall_ms_mean": round(mean(wall), 3) if wall else "",
                "generation_tps_mean": round(mean(generation), 3) if generation else "",
            }
        )
    return summary


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], single_summary: list[dict[str, Any]], backends: dict[str, str]) -> None:
    lines = [
        "# llama-cli multi-U speedup",
        "",
        "## Single-call ranking",
        "",
        "| U | Prompt chars | Max tokens | Wall s | Generation t/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in single_summary:
        wall_s = float(row["wall_ms_mean"]) / 1000.0 if row["wall_ms_mean"] != "" else 0.0
        lines.append(
            f"| {row['u_label']} | {row['prompt_chars_target']} | {row['max_tokens']} | "
            f"{wall_s:.3f} | {row['generation_tps_mean']} |"
        )

    lines.extend(
        [
            "",
            "## Serial Baselines And Parallel Speedups",
            "",
            "| Kind | Config | Degree | Max tokens | Wall s | "
            + " | ".join(f"Speedup vs {label}" for label in backends)
            + " |",
            "|---|---|---:|---:|---:|"
            + "|".join("---:" for _ in backends)
            + "|",
        ]
    )
    for row in summary_rows:
        wall_s = float(row["wall_ms_mean"]) / 1000.0 if row["wall_ms_mean"] != "" else 0.0
        speedups = " | ".join(str(row.get(f"speedup_vs_{label}_mean", "")) for label in backends)
        lines.append(
            f"| {row['kind']} | {row['config_labels']} | {row['degree']} | {row['max_tokens']} | "
            f"{wall_s:.3f} | {speedups} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    backends = parse_backends(args.backend)
    max_tokens_list = parse_int_list(args.max_tokens)
    degrees = parse_int_list(args.degrees)
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / f"llama_cli_multi_u_speedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = infer_runtime_config(args)
    base_cfg = base_llama_config(runtime_config)
    run_token = str(int(time.time() * 1000))

    single_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for max_tokens in max_tokens_list:
        for repeat in range(args.repeats):
            token_single_rows: list[dict[str, Any]] = []
            for label, device in backends.items():
                print(
                    f"[single] {label} prompt_chars={args.prompt_chars} "
                    f"max_tokens={max_tokens} repeat={repeat}",
                    flush=True,
                )
                single_row = run_slot(
                    base_cfg=base_cfg,
                    label=label,
                    llama_device=device,
                    prompt_chars=args.prompt_chars,
                    max_tokens=max_tokens,
                    repeat=repeat,
                    args=args,
                    case_id=f"{run_token}_single_{max_tokens}_{repeat}_{label}",
                    slot_index=0,
                )
                token_single_rows.append(single_row)
                single_rows.append(single_row)
                print(
                    f"      wall={float(single_row['wall_ms']) / 1000.0:.2f}s "
                    f"gen={single_row['generation_tps']} ok={row_ok(single_row)}",
                    flush=True,
                )

            for degree in degrees:
                for label, device in backends.items():
                    serial_row, details = run_serial_group(
                        base_cfg=base_cfg,
                        label=label,
                        llama_device=device,
                        degree=degree,
                        prompt_chars=args.prompt_chars,
                        max_tokens=max_tokens,
                        repeat=repeat,
                        args=args,
                        case_id=f"{run_token}_serial_{max_tokens}_{degree}_{repeat}_{label}",
                    )
                    group_rows.append(serial_row)
                    detail_rows.extend(details)

                for config_id, labels, devices in make_parallel_configs(
                    backends=backends,
                    single_rows=token_single_rows,
                    degree=degree,
                    gpu_alt_threshold=args.skip_gpu_alt_threshold,
                ):
                    parallel_row, details = run_parallel_group(
                        base_cfg=base_cfg,
                        config_id=config_id,
                        labels=labels,
                        devices=devices,
                        prompt_chars=args.prompt_chars,
                        max_tokens=max_tokens,
                        repeat=repeat,
                        args=args,
                        case_id=f"{run_token}_parallel_{max_tokens}_{degree}_{repeat}_{config_id}",
                    )
                    group_rows.append(parallel_row)
                    detail_rows.extend(details)

    add_speedups(group_rows, backends)
    summary_rows = summarize_groups(group_rows, backends)
    single_summary = summarize_single(single_rows)

    single_csv = output_dir / "single_call_results.csv"
    single_summary_csv = output_dir / "single_call_summary.csv"
    groups_csv = output_dir / "multi_u_group_results.csv"
    details_csv = output_dir / "multi_u_slot_details.csv"
    summary_csv = output_dir / "multi_u_speedup_summary.csv"
    summary_json = output_dir / "multi_u_speedup_summary.json"
    summary_md = output_dir / "MULTI_U_SPEEDUP_SUMMARY.md"

    write_csv(single_csv, single_rows)
    write_csv(single_summary_csv, single_summary)
    write_csv(groups_csv, group_rows)
    write_csv(details_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "runtime_config": str(runtime_config),
                "backends": backends,
                "single_call_results": str(single_csv),
                "single_call_summary": str(single_summary_csv),
                "group_results": str(groups_csv),
                "slot_details": str(details_csv),
                "summary_csv": str(summary_csv),
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_md, summary_rows, single_summary, backends)

    print()
    print(f"wrote {single_csv}")
    print(f"wrote {single_summary_csv}")
    print(f"wrote {groups_csv}")
    print(f"wrote {details_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_md}")
    print()
    print("| Kind | Config | Degree | Max tokens | Wall s | " + " | ".join(f"Speedup vs {label}" for label in backends) + " |")
    print("|---|---|---:|---:|---:|" + "|".join("---:" for _ in backends) + "|")
    for row in summary_rows:
        wall_s = float(row["wall_ms_mean"]) / 1000.0 if row["wall_ms_mean"] != "" else 0.0
        speedups = " | ".join(str(row.get(f"speedup_vs_{label}_mean", "")) for label in backends)
        print(
            f"| {row['kind']} | {row['config_labels']} | {row['degree']} | {row['max_tokens']} | "
            f"{wall_s:.3f} | {speedups} |"
        )


if __name__ == "__main__":
    main()
