#!/usr/bin/env python3
"""Compare p=3 single-U load against p=3 multi-U load balancing.

The workload is three independent llama-cli calls launched concurrently.  For
each device/model pair, the script measures:

* single-U p=3: CPU+CPU+CPU, GPU+GPU+GPU, NPU+NPU+NPU
* multi-U p=3 candidates: CPU+NPU+GPU, CPU+CPU+NPU, NPU+NPU+CPU

The best successful multi-U candidate is reported as the load-balanced result.
Failures are classified as timeout, llama_error, nonzero_exit, or exception.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_llama_cli_length_sweep import (  # noqa: E402
    DEFAULT_METHOD_CSV,
    base_llama_config,
    infer_runtime_config,
    run_case,
    write_csv,
)


DEFAULT_BACKENDS = {
    "CPU": "none",
    "GPU": "GPUOpenCL",
    "NPU": "HTP0",
}

SINGLE_U_CONFIGS = [
    ("single_CPU", ["CPU", "CPU", "CPU"]),
    ("single_GPU", ["GPU", "GPU", "GPU"]),
    ("single_NPU", ["NPU", "NPU", "NPU"]),
]

MULTI_U_CONFIGS = [
    ("multi_CPU_NPU_GPU", ["CPU", "NPU", "GPU"]),
    ("multi_2CPU_1NPU", ["CPU", "CPU", "NPU"]),
    ("multi_2NPU_1CPU", ["NPU", "NPU", "CPU"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-csv", type=Path, default=DEFAULT_METHOD_CSV)
    parser.add_argument("--source-method", default="wave_opt")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Optional explicit runtime config. Only valid for a single device/model run.",
    )
    parser.add_argument("--devices", default="", help="Comma-separated phone serials. Default: all in method CSV.")
    parser.add_argument("--models", default="", help="Comma-separated model IDs. Default: all in method CSV.")
    parser.add_argument(
        "--model-path",
        default="",
        help="Override llama model path for the given model ID. Relative paths are resolved under /data/local/tmp/gguf.",
    )
    parser.add_argument(
        "--template-model-id",
        default="llama_3_2_1b_instruct_q4_0",
        help="Known model ID used only to find a runtime config when --model-path is set.",
    )
    parser.add_argument("--prompt-chars", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
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
        help="Minimum llama.cpp decode token ratio required for a normal output length.",
    )
    parser.add_argument(
        "--min-output-word-ratio",
        type=float,
        default=0.5,
        help="Fallback minimum generated word ratio when llama.cpp decode token counts are unavailable.",
    )
    parser.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Backend mapping LABEL=llama_device. Default: CPU=none, GPU=GPUOpenCL, NPU=HTP0.",
    )
    parser.add_argument(
        "--include-config",
        action="append",
        default=[],
        help="Extra p=3 config ID=LABEL+LABEL+LABEL, e.g. extra=CPU+GPU+NPU.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/llama_cli_p3_load_balance_compare_<timestamp>.",
    )
    return parser.parse_args()


def parse_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_backends(items: list[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_BACKENDS)
    backends: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --backend LABEL=llama_device, got {item!r}")
        label, device = item.split("=", 1)
        backends[label.strip()] = device.strip()
    return backends


def parse_extra_configs(items: list[str]) -> list[tuple[str, list[str]]]:
    configs: list[tuple[str, list[str]]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected --include-config ID=LABEL+LABEL+LABEL, got {item!r}")
        config_id, labels_text = item.split("=", 1)
        labels = [label.strip() for label in labels_text.split("+") if label.strip()]
        if len(labels) != 3:
            raise ValueError(f"Expected exactly 3 labels in {item!r}")
        configs.append((config_id.strip(), labels))
    return configs


def available_devices_models(method_csv: Path, source_method: str) -> tuple[list[str], list[str]]:
    rows = [
        row
        for row in parse_csv(method_csv)
        if row.get("method_id") == source_method and row.get("status_note") == "ok"
    ]
    return sorted({row["device"] for row in rows}), sorted({row["model_id"] for row in rows})


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def normalize_model_path(path: str) -> str:
    if not path:
        return ""
    if path.startswith("/"):
        return path
    return f"/data/local/tmp/gguf/{path}"


def config_for_slot(base_cfg: dict[str, Any], case_id: str, slot_index: int, label: str) -> dict[str, Any]:
    cfg = dict(base_cfg)
    base_remote = str(cfg.get("remote_prompt_dir") or "/data/local/tmp/wfgen_llama_prompts").rstrip("/")
    cfg["remote_prompt_dir"] = f"{base_remote}_p3load_{safe_name(case_id)}_s{slot_index}_{safe_name(label)}"
    return cfg


def row_success(row: dict[str, Any]) -> bool:
    return (
        row.get("status") == "ok"
        and not row.get("stdout_error")
        and str(row.get("returncode", "0")) in ("", "0")
        and row.get("output_length_ok") is True
    )


def classify_failure(row: dict[str, Any]) -> str:
    if row_success(row):
        return ""
    text = " ".join(
        str(row.get(key, ""))
        for key in ("error", "stdout_tail", "returncode")
    ).lower()
    if "device not found" in text or "no devices/emulators" in text or "offline" in text:
        return "adb_device_lost"
    if '"returncode": 255' in text and '"stdout": ""' in text and '"stderr": ""' in text:
        return "adb_or_device_lost"
    if "timed out" in text or "timeoutexpired" in text or "timeout" in text:
        return "timeout"
    if "failed to initialize" in text or "error:" in text or row.get("stdout_error"):
        return "llama_error"
    if row.get("output_length_ok") is False:
        return f"output_length:{row.get('output_length_error') or 'invalid'}"
    returncode = str(row.get("returncode", ""))
    if returncode not in ("", "0"):
        return "nonzero_exit"
    if row.get("status") != "ok":
        return "exception"
    return "unknown_failure"


def compact_error(row: dict[str, Any]) -> str:
    error = str(row.get("error") or "")
    stdout_tail = str(row.get("stdout_tail") or "")
    length = ""
    if row.get("output_length_ok") is False:
        length = (
            f"output_length_error={row.get('output_length_error')}; "
            f"decode_tokens={row.get('decode_tokens')}; "
            f"min_decode_tokens={row.get('min_decode_tokens')}; "
            f"output_chars={row.get('output_chars')}; "
            f"output_words={row.get('output_word_count')}; "
        )
    combined = (length + " " + error + " " + stdout_tail).replace("\\n", " ")
    return combined[:600]


def run_slot(
    *,
    base_cfg: dict[str, Any],
    device_serial: str,
    model_id: str,
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
    failure_type = classify_failure(row)
    row.update(
        {
            "device_serial": device_serial,
            "model_id": model_id,
            "u_label": label,
            "case_id": case_id,
            "slot_index": slot_index,
            "failure_type": failure_type,
            "error_compact": compact_error(row) if failure_type else "",
        }
    )
    return row


def run_p3_config(
    *,
    base_cfg: dict[str, Any],
    device_serial: str,
    model_id: str,
    config_id: str,
    labels: list[str],
    backends: dict[str, str],
    kind: str,
    repeat: int,
    args: argparse.Namespace,
    run_token: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    devices = [backends[label] for label in labels]
    case_id = f"{run_token}_{device_serial}_{model_id}_{config_id}_r{repeat}"
    print(
        f"[{device_serial}] {model_id} {config_id} labels={'+'.join(labels)} "
        f"decode={args.max_tokens} repeat={repeat}",
        flush=True,
    )
    started = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                run_slot,
                base_cfg=base_cfg,
                device_serial=device_serial,
                model_id=model_id,
                label=label,
                llama_device=llama_device,
                prompt_chars=args.prompt_chars,
                max_tokens=args.max_tokens,
                repeat=repeat,
                args=args,
                case_id=case_id,
                slot_index=index,
            )
            for index, (label, llama_device) in enumerate(zip(labels, devices))
        ]
        details = [future.result() for future in futures]
    wall_ms = (time.time() - started) * 1000.0
    ok_slots = sum(1 for row in details if row_success(row))
    length_ok_slots = sum(1 for row in details if row.get("output_length_ok") is True)
    failure_types = sorted({row["failure_type"] for row in details if row["failure_type"]})
    status = "ok" if ok_slots == 3 else "error"
    decode_tokens = [
        int(row["decode_tokens"])
        for row in details
        if str(row.get("decode_tokens", "")).isdigit()
    ]
    generation_tps = [
        float(row["generation_tps"])
        for row in details
        if row_success(row) and row.get("generation_tps") not in ("", None)
    ]
    output_words = [
        int(row["output_word_count"])
        for row in details
        if str(row.get("output_word_count", "")).isdigit()
    ]
    row = {
        "device_serial": device_serial,
        "model_id": model_id,
        "kind": kind,
        "config_id": config_id,
        "config_labels": "+".join(labels),
        "config_devices": "+".join(devices),
        "p": 3,
        "prompt_chars_target": args.prompt_chars,
        "max_tokens": args.max_tokens,
        "repeat": repeat,
        "status": status,
        "wall_ms": round(wall_ms, 3),
        "component_max_ms": round(max(float(row["wall_ms"]) for row in details), 3),
        "component_sum_ms": round(sum(float(row["wall_ms"]) for row in details), 3),
        "ok_slots": ok_slots,
        "error_slots": 3 - ok_slots,
        "output_length_ok_slots": length_ok_slots,
        "aggregate_generation_tps": round(sum(generation_tps), 3) if ok_slots == 3 and len(generation_tps) == 3 else "",
        "mean_slot_generation_tps": round(mean(generation_tps), 3) if generation_tps else "",
        "slot_generation_tps": "+".join(str(value) for value in generation_tps),
        "min_decode_tokens_observed": min(decode_tokens) if decode_tokens else "",
        "mean_decode_tokens_observed": round(mean(decode_tokens), 3) if decode_tokens else "",
        "min_output_words_observed": min(output_words) if output_words else "",
        "mean_output_words_observed": round(mean(output_words), 3) if output_words else "",
        "failure_types": "+".join(failure_types),
        "error_compact": " | ".join(row["error_compact"] for row in details if row["error_compact"])[:1200],
    }
    print(
        f"      status={status} wall={wall_ms / 1000.0:.2f}s "
        f"ok_slots={ok_slots}/3 failure={row['failure_types'] or '-'}",
        flush=True,
    )
    return row, details


def summarize(groups: list[dict[str, Any]], backends: dict[str, str]) -> list[dict[str, Any]]:
    keys = sorted({(row["device_serial"], row["model_id"], row["config_id"]) for row in groups})
    summary: list[dict[str, Any]] = []
    for device_serial, model_id, config_id in keys:
        items = [
            row
            for row in groups
            if row["device_serial"] == device_serial and row["model_id"] == model_id and row["config_id"] == config_id
        ]
        ok = [row for row in items if row["status"] == "ok"]
        wall = [float(row["wall_ms"]) for row in ok]
        throughput = [
            float(row["aggregate_generation_tps"])
            for row in ok
            if row.get("aggregate_generation_tps") not in ("", None)
        ]
        first = items[0]
        summary.append(
            {
                "device_serial": device_serial,
                "model_id": model_id,
                "kind": first["kind"],
                "config_id": config_id,
                "config_labels": first["config_labels"],
                "runs": len(items),
                "ok_runs": len(ok),
                "error_runs": len(items) - len(ok),
                "wall_ms_mean": round(mean(wall), 3) if wall else "",
                "wall_s_mean": round(mean(wall) / 1000.0, 3) if wall else "",
                "aggregate_generation_tps_mean": round(mean(throughput), 3) if throughput else "",
                "output_length_ok_runs": sum(1 for row in ok if row.get("output_length_ok_slots") == 3),
                "min_decode_tokens_observed": min(
                    int(row["min_decode_tokens_observed"])
                    for row in ok
                    if str(row.get("min_decode_tokens_observed", "")).isdigit()
                )
                if any(str(row.get("min_decode_tokens_observed", "")).isdigit() for row in ok)
                else "",
                "mean_decode_tokens_observed": round(
                    mean(float(row["mean_decode_tokens_observed"]) for row in ok if row.get("mean_decode_tokens_observed") != ""),
                    3,
                )
                if any(row.get("mean_decode_tokens_observed") != "" for row in ok)
                else "",
                "failure_types": "+".join(sorted({row["failure_types"] for row in items if row["failure_types"]})),
                "error_compact": " | ".join(row["error_compact"] for row in items if row["error_compact"])[:1200],
            }
        )
    return summary


def compare_best(summary_rows: list[dict[str, Any]], backends: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(row["device_serial"], row["model_id"]) for row in summary_rows})
    for device_serial, model_id in keys:
        rows = [row for row in summary_rows if row["device_serial"] == device_serial and row["model_id"] == model_id]
        single_rows = {row["config_id"]: row for row in rows if row["kind"] == "single_u"}
        multi_ok = [
            row
            for row in rows
            if row["kind"] == "multi_u" and row["wall_ms_mean"] != ""
        ]
        best_multi = min(multi_ok, key=lambda row: float(row["wall_ms_mean"])) if multi_ok else None
        best_single_ok = [
            row
            for row in single_rows.values()
            if row["wall_ms_mean"] != ""
        ]
        best_single = min(best_single_ok, key=lambda row: float(row["wall_ms_mean"])) if best_single_ok else None
        record: dict[str, Any] = {
            "device_serial": device_serial,
            "model_id": model_id,
            "best_multi_config": best_multi["config_labels"] if best_multi else "",
            "best_multi_wall_s": best_multi["wall_s_mean"] if best_multi else "",
            "best_single_config": best_single["config_labels"] if best_single else "",
            "best_single_wall_s": best_single["wall_s_mean"] if best_single else "",
            "speedup_vs_best_single": (
                round(float(best_single["wall_ms_mean"]) / float(best_multi["wall_ms_mean"]), 4)
                if best_single and best_multi
                else ""
            ),
        }
        for label in backends:
            row = single_rows.get(f"single_{label}")
            record[f"single_{label}_wall_s"] = row["wall_s_mean"] if row else ""
            record[f"single_{label}_status"] = "ok" if row and row["wall_ms_mean"] != "" else "error"
            record[f"single_{label}_failure"] = row["failure_types"] if row else "missing"
            record[f"speedup_vs_single_{label}"] = (
                round(float(row["wall_ms_mean"]) / float(best_multi["wall_ms_mean"]), 4)
                if row and row["wall_ms_mean"] != "" and best_multi
                else ""
            )
        out.append(record)
    return out


def write_markdown(path: Path, comparisons: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# p=3 load balance comparison",
        "",
        "| Device | Model | Best single-U | Best single s | Best multi-U | Best multi s | Speedup | CPU p3 | GPU p3 | NPU p3 |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['device_serial']} | {row['model_id']} | {row['best_single_config']} | "
            f"{row['best_single_wall_s']} | {row['best_multi_config']} | {row['best_multi_wall_s']} | "
            f"{row['speedup_vs_best_single']} | {row.get('single_CPU_wall_s', '')} | "
            f"{row.get('single_GPU_wall_s', '')} | {row.get('single_NPU_wall_s', '')} |"
        )

    failures = [
        row
        for row in summary_rows
        if row["error_runs"] and row["error_runs"] != "0"
    ]
    lines.extend(["", "## Failures", ""])
    if failures:
        lines.append("| Device | Model | Config | Failure | Error |")
        lines.append("|---|---|---|---|---|")
        for row in failures:
            lines.append(
                f"| {row['device_serial']} | {row['model_id']} | {row['config_labels']} | "
                f"{row['failure_types']} | {row['error_compact']} |"
            )
    else:
        lines.append("No failed p=3 groups.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    backends = parse_backends(args.backend)
    missing = {label for _, labels in [*SINGLE_U_CONFIGS, *MULTI_U_CONFIGS, *parse_extra_configs(args.include_config)] for label in labels} - set(backends)
    if missing:
        raise ValueError(f"Missing backend mapping for labels: {sorted(missing)}")

    all_devices, all_models = available_devices_models(args.method_csv, args.source_method)
    devices = parse_list(args.devices) or all_devices
    models = parse_list(args.models) or all_models
    if args.runtime_config and (len(devices) != 1 or len(models) != 1):
        raise ValueError("--runtime-config can only be used with exactly one device and one model")
    if args.model_path and len(models) != 1:
        raise ValueError("--model-path can only be used with exactly one model ID per run")
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / f"llama_cli_p3_load_balance_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [(config_id, labels, "single_u") for config_id, labels in SINGLE_U_CONFIGS]
    configs.extend((config_id, labels, "multi_u") for config_id, labels in MULTI_U_CONFIGS)
    configs.extend((config_id, labels, "multi_u") for config_id, labels in parse_extra_configs(args.include_config))

    run_token = str(int(time.time() * 1000))
    group_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    runtime_configs: dict[str, str] = {}

    for device_serial in devices:
        for model_id in models:
            ns = argparse.Namespace(**vars(args))
            ns.device = device_serial
            ns.model_id = args.template_model_id if args.model_path else model_id
            runtime_config = infer_runtime_config(ns)
            runtime_configs[f"{device_serial}/{model_id}"] = str(runtime_config)
            base_cfg = base_llama_config(runtime_config)
            if args.model_path:
                base_cfg["model_path"] = normalize_model_path(args.model_path)
            for repeat in range(args.repeats):
                for config_id, labels, kind in configs:
                    row, details = run_p3_config(
                        base_cfg=base_cfg,
                        device_serial=device_serial,
                        model_id=model_id,
                        config_id=config_id,
                        labels=labels,
                        backends=backends,
                        kind=kind,
                        repeat=repeat,
                        args=args,
                        run_token=run_token,
                    )
                    group_rows.append(row)
                    detail_rows.extend(details)

    summary_rows = summarize(group_rows, backends)
    comparison_rows = compare_best(summary_rows, backends)

    group_csv = output_dir / "p3_load_balance_groups.csv"
    detail_csv = output_dir / "p3_load_balance_slot_details.csv"
    summary_csv = output_dir / "p3_load_balance_summary.csv"
    comparison_csv = output_dir / "p3_load_balance_best_comparison.csv"
    summary_json = output_dir / "p3_load_balance_summary.json"
    summary_md = output_dir / "P3_LOAD_BALANCE_SUMMARY.md"

    write_csv(group_csv, group_rows)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(comparison_csv, comparison_rows)
    summary_json.write_text(
        json.dumps(
            {
                "runtime_configs": runtime_configs,
                "backends": backends,
                "group_csv": str(group_csv),
                "detail_csv": str(detail_csv),
                "summary_csv": str(summary_csv),
                "comparison_csv": str(comparison_csv),
                "comparison": comparison_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(summary_md, comparison_rows, summary_rows)

    print()
    print(f"wrote {group_csv}")
    print(f"wrote {detail_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {comparison_csv}")
    print(f"wrote {summary_md}")
    print()
    print("| Device | Model | Best single-U | Best single s | Best multi-U | Best multi s | Speedup |")
    print("|---|---|---|---:|---|---:|---:|")
    for row in comparison_rows:
        print(
            f"| {row['device_serial']} | {row['model_id']} | {row['best_single_config']} | "
            f"{row['best_single_wall_s']} | {row['best_multi_config']} | {row['best_multi_wall_s']} | "
            f"{row['speedup_vs_best_single']} |"
        )


if __name__ == "__main__":
    main()
