"""Generate hardware, LLM, and framework comparison figures.

All comparisons are split by workflow. Within each workflow, the plotted
dimension is compared directly and the other available dimensions are averaged
out: hardware averages over model/framework combinations, model averages over
hardware/framework combinations, and framework averages over hardware/model
combinations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_characterization_figures"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import figure_style
except ImportError:
    import figure_style


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "studies" / "characterization"
CREATOR = "Paper_MOBench characterization figure generator"
STUDY_FIGSIZE = (5.35, 1.80)
STUDY_FONT = {
    "base": 8.2,
    "title": 8.4,
    "label": 8.2,
    "tick": 7.5,
    "legend": 7.2,
}
STUDY_LEGEND_FONT = STUDY_FONT["legend"]
GITHUB_ROOT = REPO_ROOT.parent
DEFAULT_WFGEN_ROOT = REPO_ROOT / "unbundled_raw"
FIGURE_DATA_ROOT = DEFAULT_WFGEN_ROOT / "figure_data"
FIGURE_DATA_ALIASES = {
    "framework_characterization_final_20260622": "fig13-framework_characterization_final_20260622",
    "framework_characterization_llama_cpp_10wf_stable_20260622": (
        "fig13-framework_characterization_llama_cpp_10wf_stable_20260622"
    ),
    "ort_genie_same_model_phi35_10wf_p1_20260622": (
        "fig14-ort_genie_same_model_phi35_10wf_p1_20260622"
    ),
    "ort_genie_same_model_phi35_10wf_p2_20260622": (
        "fig14-ort_genie_same_model_phi35_10wf_p2_20260622"
    ),
    "framework_characterization_phi35_p1_10wf_20260622": (
        "fig14-framework_characterization_phi35_p1_10wf_20260622"
    ),
    "framework_characterization_phi35_fig14_selected_20260627": (
        "fig14-framework_characterization_phi35_fig14_selected_20260627"
    ),
    "framework_characterization_phi35_wave_cpu_npu_p2_10wf_20260627": (
        "fig14-framework_characterization_phi35_wave_cpu_npu_p2_10wf_20260627"
    ),
    "framework_characterization_phi35_wave_cpu_npu_p3_10wf_20260627": (
        "fig14-framework_characterization_phi35_wave_cpu_npu_p3_10wf_20260627"
    ),
    "framework_characterization_phi35_wave_cpu_npu_p3_8elite_retry_20260627": (
        "fig14-framework_characterization_phi35_wave_cpu_npu_p3_8elite_retry_20260627"
    ),
    "framework_characterization_phi35_single_npu_p1_10wf_20260627": (
        "fig14-framework_characterization_phi35_single_npu_p1_10wf_20260627"
    ),
    "framework_characterization_phi35_single_npu_p1_8gen3_10wf_20260627": (
        "fig14-framework_characterization_phi35_single_npu_p1_8gen3_10wf_20260627"
    ),
    "llama_cpp_wave_opt_20260623": "fig15-llama_cpp_wave_opt_20260623",
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623": (
        "fig6_9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623": (
        "fig9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623"
    ),
}

DEVICE_LABELS = {
    "elite": "8 Elite",
    "gen3": "8 Gen3",
}

MODEL_LABELS = {
    "qwen3_0_6b_q4_k_m": "Qwen3 0.6B Q4",
    "qwen3_0_6b_bf16": "Qwen3 0.6B BF16",
    "qwen3_0_6b_q6_k": "Qwen3 0.6B Q6",
    "qwen3_0_6b_q8_0": "Qwen3 0.6B Q8",
    "hunyuan_0_5b_instruct_q6_k": "Hunyuan 0.5B Q6",
    "llama_3_2_1b_instruct_q4_0": "Llama 3.2 1B Q4",
    "llama_3_2_3b_instruct_q4_0": "Llama 3.2 3B Q4",
    "gemma_3_1b_it_q4_k_s": "Gemma 3 1B Q4",
    "qwen1_5_1_8b_chat_q4_0": "Qwen1.5 1.8B Q4",
    "phi_3_5_mini_instruct_q4_k_m": "Phi-3.5 mini",
}

MODEL_TICK_LABELS = {
    "qwen3_0_6b_q4_k_m": "Qwen3\nQ4",
    "qwen3_0_6b_bf16": "Qwen3\nBF16",
    "qwen3_0_6b_q6_k": "Qwen3\nQ6",
    "qwen3_0_6b_q8_0": "Qwen3\nQ8",
    "hunyuan_0_5b_instruct_q6_k": "Hunyuan\nQ6",
    "llama_3_2_1b_instruct_q4_0": "Llama3.2\nQ4",
    "llama_3_2_3b_instruct_q4_0": "Llama3.2\n3B Q4",
    "gemma_3_1b_it_q4_k_s": "Gemma3\nQ4",
    "qwen1_5_1_8b_chat_q4_0": "Qwen1.5\nQ4",
    "phi_3_5_mini_instruct_q4_k_m": "Phi-3.5\nmini",
}

MODEL_LEGEND_LABELS = {
    "qwen3_0_6b_q4_k_m": "Qwen3-Q4",
    "qwen3_0_6b_bf16": "Qwen3-BF16",
    "qwen3_0_6b_q6_k": "Qwen3-Q6",
    "qwen3_0_6b_q8_0": "Qwen3-Q8",
    "hunyuan_0_5b_instruct_q6_k": "Hunyuan-Q6",
    "llama_3_2_1b_instruct_q4_0": "Llama1B-Q4",
    "llama_3_2_3b_instruct_q4_0": "Llama3B-Q4",
    "qwen1_5_1_8b_chat_q4_0": "Qwen1.5-Q4",
    "phi_3_5_mini_instruct_q4_k_m": "Phi-3.5",
}

FRAMEWORK_LABELS = {
    "genie_npu_phi35": "Genie NPU",
    "ort_genai_cpu_phi35": "ORT GenAI CPU",
    "llama_cpp": "llama.cpp selected",
}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures for hardware, LLM, and framework characterization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wfgen-root", type=Path, default=None, help="WFGen repository root.")
    parser.add_argument("--combined-summary", type=Path, default=None, help="combined_summary.csv path.")
    parser.add_argument("--model-device-stats", type=Path, default=None, help="Legacy model_device_stats.csv path.")
    parser.add_argument(
        "--model-wave-opt-cases",
        type=Path,
        default=None,
        help="MOBench wave-opt case CSV to use for the model-characterization panel.",
    )
    parser.add_argument(
        "--framework-summary",
        type=Path,
        action="append",
        default=None,
        help="same-model framework summary.csv path. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output figure directory.")
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional PNG preview directory.")
    args = parser.parse_args()

    from refresh_evaluation_sources import SNAPSHOT as evaluation_snapshot, read
    if evaluation_snapshot.exists() and not any((args.wfgen_root, args.combined_summary,
            args.model_device_stats, args.model_wave_opt_cases, args.framework_summary)):
        metrics = read(evaluation_snapshot)['characterization']
        plot_hardware_by_workflow(metrics, args.output_dir, args.preview_dir)
        plot_model_by_workflow(metrics, args.output_dir, args.preview_dir)
        plot_framework_by_workflow(metrics, args.output_dir, args.preview_dir)
        write_provenance(metrics, args.output_dir)
        return

    wfgen_root = locate_wfgen_root(args.wfgen_root)
    combined_summary = (
        args.combined_summary
        or wfgen_root / "figure_data" / "fig13-framework_characterization_final_20260622" / "combined_summary.csv"
    )
    model_device_stats = (
        args.model_device_stats
        or wfgen_root / "figure_data" / "fig13-framework_characterization_final_20260622" / "model_device_stats.csv"
    )
    model_wave_opt_cases = (
        args.model_wave_opt_cases
        or wfgen_root
        / "figure_data"
        / "fig14-model_characterization_wave_opt_8models_nogemma_20260629"
        / "wave_opt_cases.csv"
    )
    framework_summaries = args.framework_summary or default_framework_summaries(wfgen_root)
    output_dir = args.output_dir.resolve()
    preview_dir = args.preview_dir.resolve() if args.preview_dir else None

    metrics = build_metrics(
        combined_summary.resolve(),
        model_device_stats.resolve(),
        model_wave_opt_cases.resolve() if model_wave_opt_cases.exists() else None,
        [path.resolve() for path in framework_summaries],
    )
    plot_hardware_by_workflow(metrics, output_dir, preview_dir)
    plot_model_by_workflow(metrics, output_dir, preview_dir)
    plot_framework_by_workflow(metrics, output_dir, preview_dir)
    write_provenance(metrics, output_dir)

    print(f"Wrote {output_dir / 'characterization_hardware_by_workflow.pdf'}")
    print(f"Wrote {output_dir / 'characterization_model_by_workflow.pdf'}")
    print(f"Wrote {output_dir / 'characterization_framework_by_workflow.pdf'}")
    print(f"Wrote {output_dir / 'characterization_comparison_provenance.json'}")


def locate_wfgen_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not (root / "wfgen_deploy.py").exists():
            raise FileNotFoundError(f"WFGen root does not look valid: {root}")
        return root

    candidates = [
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        DEFAULT_WFGEN_ROOT,
        REPO_ROOT.parent / "WFGen",
        REPO_ROOT.parent,
    ]
    for candidate in candidates:
        if (candidate / "wfgen_deploy.py").exists() and (candidate / "figure_data").exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate WFGen root. Pass --wfgen-root.")


def default_framework_summaries(wfgen_root: Path) -> list[Path]:
    """Use the complete Phi-3.5 framework-characterization sources for Fig. 14."""
    summaries = [
        wfgen_root / "figure_data" / "fig14-ort_genie_same_model_phi35_10wf_p1_20260622" / "summary.csv",
        wfgen_root / "figure_data" / "fig14-ort_genie_same_model_phi35_10wf_p2_20260622" / "summary.csv",
        wfgen_root / "figure_data" / "fig14-framework_characterization_phi35_fig14_selected_20260627" / "summary.csv",
    ]
    missing = [path for path in summaries if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing default framework summary source(s):\n{missing_text}")
    return summaries


def resolve_figure_data_path(path_value: str | Path) -> Path:
    path_text = str(path_value)
    path = Path(path_text)
    parts = (
        PureWindowsPath(path_text).parts
        if "\\" in path_text or ":" in path_text
        else path.parts
    )
    if "results" in parts:
        result_idx = len(parts) - 1 - parts[::-1].index("results")
        rel_parts = parts[result_idx + 1 :]
        if rel_parts:
            alias = FIGURE_DATA_ALIASES.get(rel_parts[0])
            if alias:
                candidate = FIGURE_DATA_ROOT.joinpath(alias, *rel_parts[1:])
                if candidate.exists():
                    return candidate
            candidate = FIGURE_DATA_ROOT.joinpath(*rel_parts)
            if candidate.exists():
                return candidate
    return path


def build_metrics(
    combined_summary: Path,
    model_device_stats: Path,
    model_wave_opt_cases: Path | None,
    framework_summaries: list[Path],
) -> dict[str, Any]:
    hardware_rows = read_combined_summary(combined_summary)
    model_device_rows = read_model_device_stats(model_device_stats)
    model_source_rows = (
        read_wave_opt_cases_as_workflow_rows(model_wave_opt_cases)
        if model_wave_opt_cases is not None
        else list(hardware_rows)
    )
    framework_rows = read_framework_summaries(framework_summaries)
    if not hardware_rows:
        raise ValueError(f"No completed characterization rows found in {combined_summary}")
    if not model_source_rows:
        raise ValueError(f"No model characterization rows found in {model_wave_opt_cases or combined_summary}")
    if not model_device_rows:
        raise ValueError(f"No model/device rows found in {model_device_stats}")
    if not framework_rows:
        raise ValueError(f"No framework rows found in {framework_summaries}")

    model_order = sorted_model_ids_from_workflow_rows(model_source_rows)
    device_order = sorted_device_ids(hardware_rows)
    framework_order = sorted_framework_ids(framework_rows)

    return {
        "sources": {
            "combined_summary": str(combined_summary),
            "model_device_stats": str(model_device_stats),
            "model_source": str(model_wave_opt_cases or combined_summary),
            "model_source_kind": "odabench_wave_opt_cases" if model_wave_opt_cases is not None else "combined_summary",
            "framework_summary": str(framework_summaries[0]),
            "framework_summaries": [str(path) for path in framework_summaries],
        },
        "model_order": model_order,
        "device_order": device_order,
        "framework_order": framework_order,
        "hardware_rows": hardware_rows,
        "model_device_rows": model_device_rows,
        "model_source_rows": model_source_rows,
        "framework_rows": framework_rows,
        "hardware_by_workflow": build_hardware_by_workflow(hardware_rows, device_order),
        "model_by_workflow": build_model_by_workflow(model_source_rows, model_order),
        "framework_by_workflow": build_framework_by_workflow(framework_rows, framework_order),
    }


def read_combined_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "completed":
                continue
            wall_s = optional_float(row.get("wall_s"))
            if wall_s is None:
                wall_ms = optional_float(row.get("wall_ms"))
                wall_s = None if wall_ms is None else wall_ms / 1000.0
            if wall_s is None:
                continue
            rows.append(
                {
                    "device": row["device"],
                    "device_label": display_device(row["device"], None),
                    "framework_id": row.get("framework_id") or "unknown",
                    "framework_label": display_framework(row.get("framework_id") or "unknown", row.get("framework_label")),
                    "model_id": row.get("model_id") or "unknown",
                    "model_label": display_model(row.get("model_id") or "unknown", row.get("model_name")),
                    "workflow_index": int(float(row.get("workflow_index") or 0)),
                    "workflow_id": row.get("workflow_id") or "",
                    "wall_s": wall_s,
                }
            )
    return rows


def read_wave_opt_cases_as_workflow_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for case in csv.DictReader(handle):
            if case.get("method_id") != "wave_opt":
                continue
            run_dir = resolve_figure_data_path(case["run_dir"])
            manifest_path = run_dir / "execution" / "EXECUTION_MANIFEST.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing wave-opt execution manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for run in manifest.get("runs", []):
                if run.get("status") != "completed":
                    continue
                wall_ms = optional_float(run.get("total_wall_ms"))
                if wall_ms is None:
                    continue
                workflow_index = int(float(run.get("workflow_index") or 0))
                rows.append(
                    {
                        "device": case["device"],
                        "device_label": display_device(case["device"], case.get("device_label")),
                        "framework_id": case.get("framework_id") or "llama_cpp_wave_cpu_npu",
                        "framework_label": display_framework(
                            case.get("framework_id") or "llama_cpp_wave_cpu_npu",
                            case.get("framework_label"),
                        ),
                        "model_id": case["model_id"],
                        "model_label": display_model(case["model_id"], case.get("model_label")),
                        "workflow_index": workflow_index,
                        "workflow_id": run.get("workflow_id") or "",
                        "wall_s": wall_ms / 1000.0,
                        "selected_parallel": optional_float(case.get("selected_parallel")),
                        "selected_method_id": case.get("selected_method_id") or "",
                        "selected_method_label": case.get("selected_method_label") or "",
                        "run_dir": str(run_dir),
                        "result_json": str(resolve_figure_data_path(run.get("result_json") or "")),
                        "source_summary": str(path),
                    }
                )
    return rows


def read_model_device_stats(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed = int(float(row.get("completed") or 0))
            workflows = int(float(row.get("workflows") or 0))
            rows.append(
                {
                    "device": row["device"],
                    "device_label": display_device(row["device"], row.get("device_label")),
                    "model_id": row["model_id"],
                    "model_label": display_model(row["model_id"], row.get("model_label")),
                    "workflows": workflows,
                    "completed": completed,
                    "failed": int(float(row.get("failed") or 0)),
                    "parallel": int(float(row.get("parallel") or 0)),
                    "median_wall_s": required_float(row, "median_wall_s", path),
                    "mean_wall_s": required_float(row, "mean_wall_s", path),
                    "total_wall_s": required_float(row, "total_wall_s", path),
                    "max_wall_s": optional_float(row.get("max_wall_s")),
                    "max_thermal_c": optional_float(row.get("max_thermal_c")),
                    "max_memory_mb": optional_float(row.get("max_memory_mb")),
                }
            )
    return rows


def read_framework_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_framework_summary(path))
    return rows


def read_framework_summary(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "completed":
                continue
            wall_ms = required_float(row, "wall_ms", path)
            rows.append(
                {
                    "device": row["device"],
                    "device_label": display_device(row["device"], None),
                    "framework_id": row["framework_id"],
                    "framework_label": display_framework(row["framework_id"], row.get("framework_label")),
                    "model_id": row.get("model_id") or "",
                    "model_name": row.get("model_name") or "",
                    "workflow_index": int(float(row.get("workflow_index") or 0)),
                    "workflow_id": row.get("workflow_id") or "",
                    "max_parallel_nodes": int(float(row.get("max_parallel_nodes") or 0)),
                    "wall_s": wall_ms / 1000.0,
                    "llm_inference_wall_s": optional_float(row.get("llm_inference_wall_ms_sum"), scale=1 / 1000.0),
                    "cpu_avg_percent": optional_float(row.get("cpu_avg_percent")),
                    "thermal_max_c": optional_float(row.get("thermal_max_c")),
                    "placements": row.get("placements") or "",
                    "source_summary": str(resolve_figure_data_path(row.get("source_summary") or str(path))),
                    "run_dir": str(resolve_figure_data_path(row.get("run_dir") or "")),
                    "result_json": str(resolve_figure_data_path(row.get("result_json") or "")),
                }
            )
    return rows


def sorted_model_ids(rows: list[dict[str, Any]]) -> list[str]:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row["median_wall_s"])
    return sorted(by_model, key=lambda model_id: statistics.mean(by_model[model_id]))


def sorted_model_ids_from_workflow_rows(rows: list[dict[str, Any]]) -> list[str]:
    by_model: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(row["model_id"], []).append(row["wall_s"])
    return sorted(by_model, key=lambda model_id: statistics.mean(by_model[model_id]))


def sorted_device_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["device"] not in seen:
            seen.append(row["device"])
    return seen


def sorted_framework_ids(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        if row["framework_id"] not in seen:
            seen.append(row["framework_id"])
    return seen


def build_hardware_by_workflow(rows: list[dict[str, Any]], device_order: list[str]) -> list[dict[str, Any]]:
    by_combo: dict[tuple[str, int, str, str], list[float]] = {}
    workflow_labels: dict[int, str] = {}
    for row in rows:
        workflow_index = int(row["workflow_index"])
        workflow_labels.setdefault(
            workflow_index,
            workflow_short_label(str(row.get("workflow_id") or ""), workflow_index),
        )
        by_combo.setdefault(
            (row["device"], workflow_index, row["framework_id"], row["model_id"]),
            [],
        ).append(row["wall_s"])

    combo_means_by_device_workflow: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (device, workflow_index, framework_id, model_id), values in by_combo.items():
        combo_means_by_device_workflow.setdefault((device, workflow_index), []).append(
            {
                "framework_id": framework_id,
                "model_id": model_id,
                "mean_wall_s": statistics.mean(values),
                "sample_count": len(values),
            }
        )

    out = []
    for workflow_index in sorted(workflow_labels):
        values = []
        for device in device_order:
            combos = combo_means_by_device_workflow.get((device, workflow_index), [])
            if not combos:
                continue
            combo_means = [combo["mean_wall_s"] for combo in combos]
            values.append(
                {
                    "device": device,
                    "device_label": display_device(device, None),
                    "avg_wall_s": statistics.mean(combo_means),
                    "median_combo_wall_s": statistics.median(combo_means),
                    "combo_count": len(combos),
                    "sample_count": sum(int(combo["sample_count"]) for combo in combos),
                    "model_count": len({combo["model_id"] for combo in combos}),
                    "framework_count": len({combo["framework_id"] for combo in combos}),
                    "combo_means": sorted(combos, key=lambda combo: (combo["framework_id"], combo["model_id"])),
                }
            )
        if values:
            out.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_label": workflow_labels[workflow_index],
                    "values": values,
                }
            )
    return out


def build_model_by_workflow(rows: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    by_combo: dict[tuple[int, str, str, str], list[float]] = {}
    workflow_labels: dict[int, str] = {}
    for row in rows:
        workflow_index = int(row["workflow_index"])
        workflow_labels.setdefault(
            workflow_index,
            workflow_short_label(str(row.get("workflow_id") or ""), workflow_index),
        )
        by_combo.setdefault(
            (workflow_index, row["model_id"], row["device"], row["framework_id"]),
            [],
        ).append(row["wall_s"])

    combo_means_by_workflow_model: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for (workflow_index, model_id, device, framework_id), values in by_combo.items():
        combo_means_by_workflow_model.setdefault((workflow_index, model_id), []).append(
            {
                "device": device,
                "framework_id": framework_id,
                "mean_wall_s": statistics.mean(values),
                "sample_count": len(values),
            }
        )

    out = []
    for workflow_index in sorted(workflow_labels):
        values = []
        for model_id in model_order:
            combos = combo_means_by_workflow_model.get((workflow_index, model_id), [])
            if not combos:
                continue
            combo_means = [combo["mean_wall_s"] for combo in combos]
            values.append(
                {
                    "model_id": model_id,
                    "model_label": display_model(model_id, None),
                    "avg_wall_s": statistics.mean(combo_means),
                    "median_combo_wall_s": statistics.median(combo_means),
                    "combo_count": len(combos),
                    "sample_count": sum(int(combo["sample_count"]) for combo in combos),
                    "device_count": len({combo["device"] for combo in combos}),
                    "framework_count": len({combo["framework_id"] for combo in combos}),
                    "combo_means": sorted(combos, key=lambda combo: (combo["device"], combo["framework_id"])),
                }
            )
        if values:
            out.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_label": workflow_labels[workflow_index],
                    "values": values,
                }
            )
    return out


def build_framework_by_workflow(rows: list[dict[str, Any]], framework_order: list[str]) -> list[dict[str, Any]]:
    by_combo: dict[tuple[int, str, str, str], list[float]] = {}
    workflow_labels: dict[int, str] = {}
    for row in rows:
        workflow_index = int(row["workflow_index"])
        workflow_labels.setdefault(
            workflow_index,
            workflow_short_label(str(row.get("workflow_id") or ""), workflow_index),
        )
        by_combo.setdefault(
            (workflow_index, row["framework_id"], row["device"], row["model_id"]),
            [],
        ).append(row["wall_s"])

    combo_means_by_workflow_framework: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for (workflow_index, framework_id, device, model_id), values in by_combo.items():
        combo_means_by_workflow_framework.setdefault((workflow_index, framework_id), []).append(
            {
                "device": device,
                "model_id": model_id,
                "mean_wall_s": statistics.mean(values),
                "sample_count": len(values),
            }
        )

    out = []
    for workflow_index in sorted(workflow_labels):
        values = []
        for framework_id in framework_order:
            combos = combo_means_by_workflow_framework.get((workflow_index, framework_id), [])
            if not combos:
                continue
            combo_means = [combo["mean_wall_s"] for combo in combos]
            values.append(
                {
                    "framework_id": framework_id,
                    "framework_label": display_framework(framework_id, None),
                    "avg_wall_s": statistics.mean(combo_means),
                    "median_combo_wall_s": statistics.median(combo_means),
                    "combo_count": len(combos),
                    "sample_count": sum(int(combo["sample_count"]) for combo in combos),
                    "device_count": len({combo["device"] for combo in combos}),
                    "model_count": len({combo["model_id"] for combo in combos}),
                    "combo_means": sorted(combos, key=lambda combo: (combo["device"], combo["model_id"])),
                }
            )
        if values:
            out.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_label": workflow_labels[workflow_index],
                    "values": values,
                }
            )
    return out


def plot_hardware_by_workflow(metrics: dict[str, Any], output_dir: Path, preview_dir: Path | None) -> None:
    configure_style()
    rows = metrics["hardware_by_workflow"]
    if not rows:
        raise ValueError("No hardware workflow averages to plot.")
    device_order = metrics["device_order"]

    fig, ax = plt.subplots(figsize=STUDY_FIGSIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.80)

    x = list(range(len(rows)))
    width = min(0.30, 0.72 / max(1, len(device_order)))
    offsets = centered_offsets(len(device_order), width)
    max_value = 0.0
    for index, device in enumerate(device_order):
        values = []
        positions = []
        for row_index, row in enumerate(rows):
            point = next((item for item in row["values"] if item["device"] == device), None)
            if point is None:
                continue
            positions.append(x[row_index] + offsets[index])
            values.append(point["avg_wall_s"])
        if values:
            max_value = max(max_value, max(values))
            ax.bar(
                positions,
                values,
                width=width,
                color=figure_style.bar_color(device),
                edgecolor=figure_style.BAR_EDGE,
                linewidth=figure_style.BAR_EDGE_WIDTH,
                label=display_device(device, None),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([row["workflow_label"] for row in rows], rotation=0, fontsize=STUDY_FONT["tick"])
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Avg latency (s)")
    ax.set_ylim(0, max_value * 1.14)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    study_box_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=max(1, len(labels)),
        prop={"size": STUDY_LEGEND_FONT},
    )
    save_figure(fig, output_dir / "characterization_hardware_by_workflow", preview_dir, exact_canvas=True)

def plot_model_by_workflow(metrics: dict[str, Any], output_dir: Path, preview_dir: Path | None) -> None:
    configure_style()
    rows = metrics["model_by_workflow"]
    model_order = metrics["model_order"]
    if not rows:
        raise ValueError("No model workflow averages to plot.")

    fig, ax = plt.subplots(figsize=STUDY_FIGSIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.80 if len(model_order) <= 3 else 0.74)

    x = list(range(len(rows)))
    max_value = 0.0
    for index, model_id in enumerate(model_order):
        values = []
        positions = []
        for row_index, row in enumerate(rows):
            point = next((item for item in row["values"] if item["model_id"] == model_id), None)
            if point is None:
                continue
            positions.append(x[row_index])
            values.append(point["avg_wall_s"])
        if not values:
            continue
        max_value = max(max_value, max(values))
        style = model_series_style(model_id)
        ax.plot(
            positions,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markeredgewidth=style["markeredgewidth"],
            markeredgecolor=style["markeredgecolor"],
            label=display_model_legend(model_id),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([row["workflow_label"] for row in rows], fontsize=STUDY_FONT["tick"])
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Avg latency (s)")
    ax.set_ylim(0, max_value * 1.14)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    study_box_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    ncol = min(4, len(labels))
    handles, labels = row_major_legend_order(handles, labels, ncol)
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=ncol,
        columnspacing=0.35,
        handlelength=1.15,
        handletextpad=0.18,
        labelspacing=0.16,
        prop={"size": STUDY_LEGEND_FONT},
    )
    save_figure(fig, output_dir / "characterization_model_by_workflow", preview_dir, exact_canvas=True)


def plot_framework_by_workflow(metrics: dict[str, Any], output_dir: Path, preview_dir: Path | None) -> None:
    configure_style()
    rows = metrics["framework_by_workflow"]
    framework_order = metrics["framework_order"]
    if not rows:
        raise ValueError("No framework workflow averages to plot.")

    fig, ax = plt.subplots(figsize=STUDY_FIGSIZE)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.80)

    x = list(range(len(rows)))
    width = min(0.30, 0.70 / max(1, len(framework_order)))
    offsets = centered_offsets(len(framework_order), width)
    max_value = 0.0
    for index, framework_id in enumerate(framework_order):
        values = []
        positions = []
        for row_index, row in enumerate(rows):
            point = next((item for item in row["values"] if item["framework_id"] == framework_id), None)
            if point is None:
                continue
            positions.append(x[row_index] + offsets[index])
            values.append(point["avg_wall_s"] if point["avg_wall_s"] is not None else float("nan"))
        if values:
            max_value = max(max_value, max((v for v in values if math.isfinite(v)), default=0.0))
            ax.bar(
                positions,
                values,
                width=width,
                color=figure_style.bar_color(framework_id),
                edgecolor=figure_style.BAR_EDGE,
                linewidth=figure_style.BAR_EDGE_WIDTH,
                label=display_framework(framework_id, None),
            )

    # Failed measurements have no complete latency; keep the original bars and
    # layout, using the paper's existing ERR convention for missing heights.
    for index, framework_id in enumerate(framework_order):
        for position, row in enumerate(rows):
            point = next((v for v in row["values"] if v["framework_id"] == framework_id), None)
            if point is not None and point["avg_wall_s"] is None:
                px = position + offsets[index]
                ax.plot(px, max_value * .025, "x", color="#B83232", markersize=3, markeredgewidth=.8)
                ax.text(px, max_value * .06, point.get("marker", "N/A"), rotation=90,
                        ha="center", va="bottom", fontsize=5.5, color="#B83232")

    ax.set_xticks(x)
    ax.set_xticklabels([row["workflow_label"] for row in rows])
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Avg latency (s)")
    ax.set_ylim(0, max_value * 1.18)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    study_box_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=max(1, len(labels)),
        prop={"size": STUDY_LEGEND_FONT},
    )
    save_figure(fig, output_dir / "characterization_framework_by_workflow", preview_dir, exact_canvas=True)


def write_provenance(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "figures": [
            "characterization_hardware_by_workflow",
            "characterization_model_by_workflow",
            "characterization_framework_by_workflow",
        ],
        "sources": metrics["sources"],
        "model_order": metrics["model_order"],
        "device_order": metrics["device_order"],
        "framework_order": metrics["framework_order"],
        "hardware_by_workflow": metrics["hardware_by_workflow"],
        "model_source_rows": metrics["model_source_rows"],
        "model_by_workflow": metrics["model_by_workflow"],
        "framework_by_workflow": metrics["framework_by_workflow"],
    }
    for key in ("datasets_by_figure", "framework_matched_parallelism", "framework_ort_correction", "framework_offline_selection", "model_characterization_completion"):
        if key in metrics:
            provenance[key] = metrics[key]
    (output_dir / "characterization_comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )


def configure_style() -> None:
    figure_style.configure_matplotlib(plt)
    plt.rcParams.update(
        {
            "font.size": STUDY_FONT["base"],
            "axes.titlesize": STUDY_FONT["title"],
            "axes.labelsize": STUDY_FONT["label"],
            "xtick.labelsize": STUDY_FONT["tick"],
            "ytick.labelsize": STUDY_FONT["tick"],
            "legend.fontsize": STUDY_FONT["legend"],
        }
    )


def study_box_axes(ax: plt.Axes) -> None:
    figure_style.box_axes(ax, grid_axis="none")
    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(figure_style.STYLE["axes_linewidth"])
        ax.spines[side].set_color(figure_style.BASELINE)


def save_figure(fig: plt.Figure, stem: Path, preview_dir: Path | None, *, exact_canvas: bool) -> None:
    figure_style.save_pdf_svg(
        fig,
        stem,
        exact_canvas=exact_canvas,
        creator=CREATOR,
        preview_dir=preview_dir,
        top_margin_points=2.0,
    )
    plt.close(fig)
    strip_trailing_whitespace(stem.with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def centered_offsets(count: int, width: float) -> list[float]:
    if count <= 1:
        return [0.0]
    center = (count - 1) / 2.0
    return [(index - center) * width for index in range(count)]


def workflow_short_label(workflow_id: str, workflow_index: int) -> str:
    match = re.search(r"cloud_(retail|airline)_(\d+)_", workflow_id)
    if not match:
        return f"W{workflow_index}"
    prefix = "R" if match.group(1) == "retail" else "A"
    return f"{prefix}{match.group(2)}"


def display_device(device: str, source_label: str | None) -> str:
    if device in DEVICE_LABELS:
        return DEVICE_LABELS[device]
    label = str(source_label or device).strip()
    if label.startswith(device):
        label = label[len(device) :].strip()
    return label or device


def display_model(model_id: str, source_label: str | None) -> str:
    return MODEL_LABELS.get(model_id) or str(source_label or model_id)


def display_model_legend(model_id: str) -> str:
    return MODEL_LEGEND_LABELS.get(model_id, display_model(model_id, None))


def display_model_tick(model_id: str) -> str:
    return MODEL_TICK_LABELS.get(model_id, display_model(model_id, None))


def model_series_style(model_id: str) -> dict[str, Any]:
    return figure_style.line_style(model_id)


def row_major_legend_order(handles: list[Any], labels: list[str], ncol: int) -> tuple[list[Any], list[str]]:
    if ncol <= 1 or len(handles) <= ncol:
        return handles, labels
    nrows = (len(handles) + ncol - 1) // ncol
    order = [
        row * ncol + col
        for col in range(ncol)
        for row in range(nrows)
        if row * ncol + col < len(handles)
    ]
    return [handles[index] for index in order], [labels[index] for index in order]


def display_framework(framework_id: str, source_label: str | None) -> str:
    return FRAMEWORK_LABELS.get(framework_id) or str(source_label or framework_id)


def required_float(row: dict[str, Any], key: str, path: Path) -> float:
    value = optional_float(row.get(key))
    if value is None:
        raise ValueError(f"Missing numeric column {key!r} in {path}")
    return value


def optional_float(value: Any, *, scale: float = 1.0) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
