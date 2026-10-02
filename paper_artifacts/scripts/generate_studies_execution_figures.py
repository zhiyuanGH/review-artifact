"""Generate execution-stage figures for the MOBench case study.

The main latency figure uses the frozen selected-ten phone-native comparison.
Sequential is round-robin p=1; MOBench selects a zero-failure p=1/2/3 suite per
phone/model. The two max-parallel baselines use p=4. Other figures retain their
separate historical datasets. Use --latency-only to update just this figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from statistics import mean
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_studies_execution_figures"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

try:
    from . import figure_style
except ImportError:
    import figure_style


REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_ROOT = REPO_ROOT.parent
WFGEN_ROOT = REPO_ROOT / "unbundled_raw"
FIGURE_DATA_ROOT = WFGEN_ROOT / "figure_data"
FIGURE_DATA_ALIASES = {
    "llama_cpp_wave_opt_20260623": "fig9_12-llama_cpp_wave_opt_20260623",
    "odabench_ablation_studies_20260624": "fig10_11-odabench_ablation_studies_20260624",
    "framework_characterization_llama_cpp_single_u_seq_p1_10wf_20260623": (
        "fig6_9_12-framework_characterization_llama_cpp_single_u_seq_p1_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_round_robin_p1_10wf_20260623": (
        "fig9_12-framework_characterization_llama_cpp_round_robin_p1_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_round_robin_p2_10wf_20260623": (
        "fig9-framework_characterization_llama_cpp_round_robin_p2_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623": (
        "fig6_9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623": (
        "fig9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623"
    ),
    "framework_characterization_odabench_wo_decider_p2_common_10wf_20260623": (
        "fig11-framework_characterization_odabench_wo_decider_p2_common_10wf_20260623"
    ),
    "framework_characterization_odabench_wo_decider_qwen15_8elite_p2_10wf_20260623": (
        "fig11-framework_characterization_odabench_wo_decider_qwen15_8elite_p2_10wf_20260623"
    ),
    "framework_characterization_odabench_wo_decider_qwen15_8gen3_p3_10wf_20260623": (
        "fig11-framework_characterization_odabench_wo_decider_qwen15_8gen3_p3_10wf_20260623"
    ),
}

MODEL_ORDER = [
    "qwen3_0_6b_q4_k_m",
    "llama_3_2_1b_instruct_q4_0",
    "qwen1_5_1_8b_chat_q4_0",
]
DEVICE_ORDER = ["elite", "gen3"]
QWEN15_DEVICE_ORDER = ["gen3", "elite"]
DEVICE_LABELS = {
    "gen3": "8 Gen 3",
    "elite": "8 Elite",
}
DECIDER_ABLATION_MODEL_ORDER = [
    "llama_3_2_1b_instruct_q4_0",
    "qwen1_5_1_8b_chat_q4_0",
    "qwen3_0_6b_q4_k_m",
]


@dataclass(frozen=True)
class ExecutionFigureConfig:
    data_dir: Path = field(
        default_factory=lambda: FIGURE_DATA_ROOT / "fig9_12-llama_cpp_wave_opt_20260623"
    )
    fig6_qwen15_dir: Path = field(
        default_factory=lambda: FIGURE_DATA_ROOT / "fig6-qwen1_5_1_8b_oracle_20260628"
    )
    ablation_studies_dir: Path = field(
        default_factory=lambda: WFGEN_ROOT
        / "figure_data"
        / "fig10_11-odabench_ablation_studies_20260624"
    )
    decider_p3_same_u_summary_csv: Path = field(
        default_factory=lambda: WFGEN_ROOT
        / "results"
        / "odabench_qwen15_p3_same_u_full_20260628"
        / "summary.csv"
    )
    decider_qwen15_figure_data_dir: Path = field(
        default_factory=lambda: FIGURE_DATA_ROOT
        / "fig10-decider_ablation_qwen15_developer_greedy_20260629"
    )
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "figures" / "studies" / "execution"
    )
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "odabench": figure_style.METHOD_COLORS["odabench"],
            "ablation": figure_style.METHOD_COLORS["ablation"],
            "naive": figure_style.METHOD_COLORS["naive"],
            "seq_cpu": figure_style.BACKEND_COLORS["cpu"],
            "seq_gpu": figure_style.BACKEND_COLORS["gpu"],
            "seq_npu": figure_style.BACKEND_COLORS["npu"],
            "oracle": figure_style.bar_color("oracle"),
            "ink": figure_style.BASE_COLORS["ink"],
            "muted": figure_style.BASE_COLORS["muted"],
            "line": figure_style.BASE_COLORS["line"],
            "grid": figure_style.BASE_COLORS["grid"],
            "white": figure_style.BASE_COLORS["white"],
        }
    )

    @property
    def method_time_csv(self) -> Path:
        return self.data_dir / "method_time_comparison.csv"

    @property
    def fig6_latency_csv(self) -> Path:
        return self.fig6_qwen15_dir / "execution_latency_qwen15_by_workflow.csv"

    @property
    def fig6_latency_cases_csv(self) -> Path:
        return self.fig6_qwen15_dir / "execution_latency_qwen15_cases.csv"

    @property
    def fig6_latency_summary_json(self) -> Path:
        return self.fig6_qwen15_dir / "execution_latency_qwen15_summary.json"

    @property
    def system_performance_csv(self) -> Path:
        return self.data_dir / "system_performance.csv"

    @property
    def wave_opt_csv(self) -> Path:
        return self.data_dir / "wave_opt_cases.csv"

    @property
    def summary_markdown(self) -> Path:
        return self.data_dir / "WAVE_OPT_SUMMARY.md"

    @property
    def ablation_studies_csv(self) -> Path:
        return self.ablation_studies_dir / "odabench_ablation_studies.csv"

    @property
    def ablation_studies_summary_markdown(self) -> Path:
        return self.ablation_studies_dir / "ODABENCH_ABLATION_STUDIES.md"


CONFIG = ExecutionFigureConfig()
STUDY_FIGSIZE = (3.45, 2.35)
STUDY_SINGLE_FIGSIZE = (3.45, 1.70)
STUDY_FONT = {
    "base": 8.2,
    "title": 8.4,
    "label": 8.2,
    "tick": 7.5,
    "legend": 7.0,
}
STUDY_LEGEND_FONT = STUDY_FONT["legend"]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def workflow_short_label(workflow_id: str, index: int) -> str:
    match = re.search(r"cloud_(retail|airline)_(\d+)_", workflow_id)
    if not match:
        return f"W{index}"
    prefix = "R" if match.group(1) == "retail" else "A"
    return f"{prefix}{match.group(2)}"


def as_float(value: Any) -> float:
    if value in (None, "", "nan", "NaN"):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def as_int(value: Any, default: int = 0) -> int:
    number = as_float(value)
    if math.isnan(number):
        return default
    return int(number)


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def mean(values: list[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def case_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    device = row.get("device", "")
    model = row.get("model_id", "")
    device_index = DEVICE_ORDER.index(device) if device in DEVICE_ORDER else 999
    model_index = MODEL_ORDER.index(model) if model in MODEL_ORDER else 999
    return device_index, model_index


def case_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["device"], row["model_id"]


def ablation_case_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    device = row.get("device", "")
    model = row.get("model_id", "")
    device_index = DEVICE_ORDER.index(device) if device in DEVICE_ORDER else 999
    model_index = (
        DECIDER_ABLATION_MODEL_ORDER.index(model)
        if model in DECIDER_ABLATION_MODEL_ORDER
        else 999
    )
    return device_index, model_index


def ablation_case_label(row: dict[str, Any]) -> str:
    model_label = {
        "llama_3_2_1b_instruct_q4_0": "Llama",
        "qwen1_5_1_8b_chat_q4_0": "Qwen1.5",
        "qwen3_0_6b_q4_k_m": "Qwen3",
    }.get(row["model_id"], row.get("model_label", row["model_id"]))
    device_label = "8E" if row["device"] == "elite" else "8G3"
    return f"{model_label}\n{device_label}"


def read_completed_runs(run_dir: str) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = resolve_figure_data_path(run_dir) / "execution" / "EXECUTION_MANIFEST.json"
    data = read_json(manifest_path)
    runs = sorted(data["runs"], key=lambda row: int(row["workflow_index"]))
    incomplete = [row for row in runs if row.get("status") != "completed"]
    if incomplete:
        failed = ", ".join(
            f"{row.get('workflow_index')}:{row.get('status')}" for row in incomplete
        )
        raise ValueError(f"Not all runs completed in {manifest_path}: {failed}")
    return manifest_path, runs


def select_method_rows(rows: list[dict[str, str]], method_id: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("method_id") == method_id]
    selected.sort(key=case_sort_key)
    return selected


def validate_same_cases(methods: dict[str, list[dict[str, str]]]) -> None:
    reference_key = next(iter(methods))
    reference_cases = [case_key(row) for row in methods[reference_key]]
    for method, rows in methods.items():
        cases = [case_key(row) for row in rows]
        if cases != reference_cases:
            raise ValueError(
                f"Case set for {method} does not match {reference_key}: "
                f"{cases} vs {reference_cases}"
            )


def collect_workflow_series(
    case_rows: list[dict[str, str]],
) -> tuple[
    dict[int, list[float]],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
]:
    series: dict[int, list[float]] = {}
    workflow_meta: dict[int, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []

    for case in case_rows:
        manifest_path, runs = read_completed_runs(case["run_dir"])
        sources.append(
            {
                "device": case.get("device"),
                "device_label": case.get("device_label"),
                "model_id": case.get("model_id"),
                "model_label": case.get("model_label"),
                "method_id": case.get("method_id"),
                "method_label": case.get("method_label"),
                "run_dir": str(resolve_figure_data_path(case["run_dir"])),
                "manifest": str(manifest_path),
                "total_wall_min": as_float(case.get("total_wall_min")),
                "status_note": case.get("status_note"),
                "selected_method_id": case.get("selected_method_id", ""),
            }
        )
        for run in runs:
            workflow_index = int(run["workflow_index"])
            workflow_id = run["workflow_id"]
            existing = workflow_meta.get(workflow_index)
            if existing and existing["workflow_id"] != workflow_id:
                raise ValueError(
                    "Workflow ID mismatch at index "
                    f"{workflow_index}: {existing['workflow_id']} vs {workflow_id}"
                )
            workflow_meta[workflow_index] = {
                "workflow_index": workflow_index,
                "workflow_id": workflow_id,
                "name": run.get("name", ""),
                "label": workflow_short_label(workflow_id, workflow_index),
            }
            series.setdefault(workflow_index, []).append(
                float(run["total_wall_ms"]) / 1000.0
            )

    return series, workflow_meta, sources


def build_latency_rows(
    method_rows: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    series_by_method: dict[str, dict[int, list[float]]] = {}
    all_meta: dict[int, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}

    for method, rows in method_rows.items():
        series, workflow_meta, method_sources = collect_workflow_series(rows)
        series_by_method[method] = series
        sources[method] = method_sources
        for workflow_index, meta in workflow_meta.items():
            existing = all_meta.get(workflow_index)
            if existing and existing["workflow_id"] != meta["workflow_id"]:
                raise ValueError(
                    "Workflow metadata mismatch at index "
                    f"{workflow_index}: {existing['workflow_id']} vs {meta['workflow_id']}"
                )
            all_meta[workflow_index] = meta

    rows: list[dict[str, Any]] = []
    for workflow_index in sorted(all_meta):
        row = dict(all_meta[workflow_index])
        for method, series in series_by_method.items():
            values = series.get(workflow_index, [])
            if len(values) != 6:
                raise ValueError(
                    f"Expected six device-model samples for {method} workflow "
                    f"{workflow_index}, got {len(values)}"
                )
            row[f"{method}_s"] = mean(values)
            row[f"{method}_sample_count"] = len(values)
            row[f"{method}_case_values_s"] = values
        row["speedup_vs_naive_x"] = row["naive_s"] / row["odabench_s"]
        row["reduction_vs_naive_pct"] = (
            100.0 * (row["naive_s"] - row["odabench_s"]) / row["naive_s"]
        )
        if "ablation" in series_by_method:
            row["speedup_vs_ablation_x"] = row["ablation_s"] / row["odabench_s"]
            row["reduction_vs_ablation_pct"] = (
                100.0 * (row["ablation_s"] - row["odabench_s"]) / row["ablation_s"]
            )
        rows.append(row)

    return rows, sources


def read_fig6_qwen15_latency_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv_rows(path):
        seq_cpu_s = as_float(row["seq_cpu_avg_s"])
        seq_gpu_s = as_float(row["seq_gpu_avg_s"])
        seq_npu_s = as_float(row["seq_npu_avg_s"])
        oracle_s = as_float(row["oracle_avg_s"])
        rows.append(
            {
                "workflow_index": as_int(row["workflow_index"]),
                "workflow_id": row["workflow_id"],
                "name": "",
                "label": row["workflow_label"],
                "seq_cpu_s": seq_cpu_s,
                "seq_cpu_sample_count": 2,
                "seq_cpu_case_values_s": [
                    as_float(row["seq_cpu_elite_s"]),
                    as_float(row["seq_cpu_gen3_s"]),
                ],
                "seq_cpu_statuses": row["seq_cpu_statuses"],
                "seq_gpu_s": seq_gpu_s,
                "seq_gpu_sample_count": 2,
                "seq_gpu_case_values_s": [
                    as_float(row["seq_gpu_elite_s"]),
                    as_float(row["seq_gpu_gen3_s"]),
                ],
                "seq_gpu_statuses": row["seq_gpu_statuses"],
                "seq_npu_s": seq_npu_s,
                "seq_npu_sample_count": 2,
                "seq_npu_case_values_s": [
                    as_float(row["seq_npu_elite_s"]),
                    as_float(row["seq_npu_gen3_s"]),
                ],
                "seq_npu_statuses": row["seq_npu_statuses"],
                "oracle_s": oracle_s,
                "oracle_sample_count": 2,
                "oracle_case_values_s": [
                    as_float(row["oracle_elite_s"]),
                    as_float(row["oracle_gen3_s"]),
                ],
                "oracle_statuses": row["oracle_statuses"],
                "speedup_seq_cpu_vs_oracle_x": seq_cpu_s / oracle_s,
                "speedup_seq_gpu_vs_oracle_x": seq_gpu_s / oracle_s,
                "speedup_seq_npu_vs_oracle_x": seq_npu_s / oracle_s,
                "reduction_seq_cpu_vs_oracle_pct": 100.0 * (seq_cpu_s - oracle_s) / seq_cpu_s,
                "reduction_seq_gpu_vs_oracle_pct": 100.0 * (seq_gpu_s - oracle_s) / seq_gpu_s,
                "reduction_seq_npu_vs_oracle_pct": 100.0 * (seq_npu_s - oracle_s) / seq_npu_s,
            }
        )
    return rows


def summarize_series(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    total_s = sum(row[f"{key}_s"] for row in rows)
    return {
        "total_s": total_s,
        "avg_s": total_s / len(rows),
    }


def resource_point(row: dict[str, str], method_key: str, label: str) -> dict[str, Any]:
    return {
        "method": method_key,
        "label": label,
        "device": row.get("device"),
        "device_label": row.get("device_label"),
        "model_id": row.get("model_id"),
        "model_label": row.get("model_label"),
        "cpu_util_pct_avg": as_float(row.get("cpu_util_pct_avg")),
        "gpu_busy_pct_avg": as_float(row.get("gpu_busy_pct_avg")),
        "mem_used_mb_max": as_float(row.get("mem_used_mb_max")),
        "thermal_max_c_avg": as_float(row.get("thermal_max_c_avg")),
        "llama_cli_rss_mb_max": as_float(row.get("llama_cli_rss_mb_max")),
        "run_dir": str(resolve_figure_data_path(row.get("run_dir") or "")),
        "resource_summary": str(resolve_figure_data_path(row.get("resource_summary") or "")),
        "perfetto_trace": str(resolve_figure_data_path(row.get("perfetto_trace") or "")),
    }


def summarize_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_metric(key: str) -> dict[str, Any]:
        values = finite_values([point[key] for point in points])
        return {
            "sample_count": len(values),
            "avg": sum(values) / len(values) if values else math.nan,
            "min": min(values) if values else math.nan,
            "max": max(values) if values else math.nan,
        }

    return {
        "case_count": len(points),
        "cpu_util_pct_avg": summarize_metric("cpu_util_pct_avg"),
        "gpu_busy_pct_avg": summarize_metric("gpu_busy_pct_avg"),
        "mem_used_mb_max": summarize_metric("mem_used_mb_max"),
        "thermal_max_c_avg": summarize_metric("thermal_max_c_avg"),
        "llama_cli_rss_mb_max": summarize_metric("llama_cli_rss_mb_max"),
    }


def build_resource_usage(
    system_rows: list[dict[str, str]],
    wave_opt_rows: list[dict[str, str]],
) -> dict[str, Any]:
    naive_rows = select_method_rows(system_rows, "round_robin_p1")
    wave_rows = sorted(wave_opt_rows, key=case_sort_key)
    methods = {
        "naive": {
            "label": "Sequential",
            "points": [
                resource_point(row, "naive", "Sequential") for row in naive_rows
            ],
        },
        "odabench": {
            "label": "Wave-opt",
            "points": [
                resource_point(row, "odabench", "Wave-opt") for row in wave_rows
            ],
        },
    }
    for method in methods.values():
        method["summary"] = summarize_points(method["points"])
    return {
        "methods": methods,
        "plot_filter": {
            "device": "gen3",
            "device_label": "8 Gen 3",
            "reason": "GPU-busy samples are available for 8 Gen 3 but absent for 8 Elite.",
        },
        "sources": {
            "system_performance_csv": str(CONFIG.system_performance_csv),
            "wave_opt_csv": str(CONFIG.wave_opt_csv),
        },
    }


def build_ablation_studies() -> dict[str, Any]:
    source_rows = sorted(
        read_csv_rows(CONFIG.ablation_studies_csv),
        key=ablation_case_sort_key,
    )
    structure_rows: list[dict[str, Any]] = []
    decider_rows: list[dict[str, Any]] = []

    for row in source_rows:
        label = ablation_case_label(row)
        odabench_min = as_float(row["odabench_total_wall_min"])
        blind_p3_min = as_float(row["blind_p3_total_wall_min"])
        wo_decider_min = as_float(row["wo_decider_total_wall_min"])

        blind_failed_workflows = as_int(row["blind_p3_failed_workflows"])
        blind_failed_llm_nodes = as_int(row["blind_p3_failed_llm_nodes"])
        blind_clean = (
            row["blind_p3_status"] == "ok"
            and blind_failed_workflows == 0
            and blind_failed_llm_nodes == 0
        )
        wo_failed_workflows = as_int(row["wo_decider_failed_workflows"])
        wo_failed_llm_nodes = as_int(row["wo_decider_failed_llm_nodes"])
        wo_clean = (
            row["wo_decider_status"] == "ok"
            and wo_failed_workflows == 0
            and wo_failed_llm_nodes == 0
        )

        common = {
            "device": row["device"],
            "device_label": row["device_label"],
            "model_id": row["model_id"],
            "model_label": row["model_label"],
            "label": label,
            "odabench_parallel": as_int(row["odabench_p"]),
            "odabench_status": row["odabench_status"],
            "odabench_total_wall_min": odabench_min,
            "odabench_trace": str(resolve_figure_data_path(row["odabench_trace"])),
        }
        structure_rows.append(
            {
                **common,
                "blind_p3_parallel": as_int(row["blind_p3_p"]),
                "blind_p3_status": row["blind_p3_status"],
                "blind_p3_total_wall_min": blind_p3_min,
                "blind_p3_failed_workflows": blind_failed_workflows,
                "blind_p3_failed_llm_nodes": blind_failed_llm_nodes,
                "blind_p3_over_odabench": as_float(row["blind_p3_over_odabench"]),
                "blind_p3_clean": blind_clean,
                "blind_p3_note": row["blind_p3_note"],
                "blind_p3_trace": str(resolve_figure_data_path(row["blind_p3_trace"])),
            }
        )
        decider_rows.append(
            {
                **common,
                "best_single_u": row["wo_decider_best_single_u"],
                "wo_decider_parallel": as_int(row["wo_decider_p"]),
                "wo_decider_total_wall_min": wo_decider_min,
                "wo_decider_status": row["wo_decider_status"],
                "wo_decider_failed_workflows": wo_failed_workflows,
                "wo_decider_failed_llm_nodes": wo_failed_llm_nodes,
                "wo_decider_clean": wo_clean,
                "wo_decider_over_odabench": (
                    wo_decider_min / odabench_min if wo_clean else math.nan
                ),
                "wo_decider_note": row["wo_decider_note"],
                "wo_decider_trace": str(resolve_figure_data_path(row["wo_decider_trace"])),
            }
        )

    structure_odabench_total = sum(
        row["odabench_total_wall_min"] for row in structure_rows
    )
    structure_blind_total = sum(
        row["blind_p3_total_wall_min"] for row in structure_rows
    )
    clean_decider_rows = [row for row in decider_rows if row["wo_decider_clean"]]
    failed_decider_rows = [row for row in decider_rows if not row["wo_decider_clean"]]
    clean_decider_odabench_total = sum(
        row["odabench_total_wall_min"] for row in clean_decider_rows
    )
    clean_wo_decider_total = sum(
        row["wo_decider_total_wall_min"] for row in clean_decider_rows
    )
    sources = {
        "csv": str(CONFIG.ablation_studies_csv),
        "summary_markdown": str(CONFIG.ablation_studies_summary_markdown),
    }

    return {
        "structure_ablation": {
            "rows": structure_rows,
            "summary": {
                "case_count": len(structure_rows),
                "clean_case_count": sum(row["blind_p3_clean"] for row in structure_rows),
                "odabench_p2_case_count": sum(
                    row["odabench_parallel"] == 2 for row in structure_rows
                ),
                "odabench_p3_case_count": sum(
                    row["odabench_parallel"] == 3 for row in structure_rows
                ),
                "odabench_total_min": structure_odabench_total,
                "blind_p3_total_min": structure_blind_total,
                "blind_p3_over_odabench": (
                    structure_blind_total / structure_odabench_total
                    if structure_odabench_total > 0
                    else math.nan
                ),
                "latency_reduction_vs_blind_p3_pct": (
                    100.0
                    * (structure_blind_total - structure_odabench_total)
                    / structure_blind_total
                    if structure_blind_total > 0
                    else math.nan
                ),
                "min_case_over_odabench": min(
                    row["blind_p3_over_odabench"] for row in structure_rows
                ),
                "max_case_over_odabench": max(
                    row["blind_p3_over_odabench"] for row in structure_rows
                ),
            },
            "sources": sources,
        },
        "decider_ablation": {
            "rows": decider_rows,
            "summary": {
                "case_count": len(decider_rows),
                "clean_case_count": len(clean_decider_rows),
                "failed_case_count": len(failed_decider_rows),
                "failed_workflows": sum(
                    row["wo_decider_failed_workflows"] for row in decider_rows
                ),
                "failed_llm_nodes": sum(
                    row["wo_decider_failed_llm_nodes"] for row in decider_rows
                ),
                "clean_odabench_total_min": clean_decider_odabench_total,
                "clean_wo_decider_total_min": clean_wo_decider_total,
                "clean_wo_decider_over_odabench": (
                    clean_wo_decider_total / clean_decider_odabench_total
                    if clean_decider_odabench_total > 0
                    else math.nan
                ),
            },
            "sources": sources,
        },
    }


def decider_csv_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def write_decider_ablation_figure_data(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir = CONFIG.decider_qwen15_figure_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "decider_ablation_qwen15_developer_greedy.csv"
    fieldnames = [
        "workflow_index",
        "workflow_id",
        "workflow_label",
        "device",
        "device_label",
        "model_id",
        "model_label",
        "method",
        "method_label",
        "latency_s",
        "partial_wall_s",
        "status",
        "marker",
        "developer_greedy_backend",
        "framework_id",
        "placements",
        "llm_failed_count",
        "llm_node_count",
        "failure_reason",
    ]
    with rows_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {key: decider_csv_value(record.get(key, "")) for key in fieldnames}
            )

    summary_out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "qwen1_5_1_8b_chat_q4_0",
        "model_label": "Qwen1.5 1.8B",
        "method_definition": {
            "oracle": "Offline topology-aware placement reference from the Fig. 6 Qwen1.5 full-workflow data.",
            "wo_decider": (
                "Developer-greedy placement: select the isolated fastest single-U "
                "backend for LLM nodes, including parallel waves. For Qwen1.5 on "
                "both devices, this backend is NPU."
            ),
        },
        "metric": "complete_workflow_latency_s",
        "rows_csv": str(rows_csv),
        "sources": summary["sources"],
        "by_device": summary["by_device"],
        "totals": summary["totals"],
        "note": (
            "latency_s is blank for failed w/o-decider workflows. partial_wall_s "
            "records time spent before runtime error and is not plotted as a "
            "completed latency bar."
        ),
    }
    summary_path = output_dir / "decider_ablation_qwen15_developer_greedy_summary.json"
    summary_path.write_text(json.dumps(json_safe(summary_out), indent=2), encoding="utf-8")

    readme_lines = [
        "# Fig. 12 Placement-Oracle Ablation Data",
        "",
        "This directory contains the Qwen1.5-1.8B workflow-level data used for Fig. 12.",
        "",
        f"- Oracle placement source: `{CONFIG.fig6_latency_csv}`",
        f"- p=3 same-U source: `{CONFIG.decider_p3_same_u_summary_csv}`",
        f"- Rows CSV: `{rows_csv}`",
        "",
        "The w/o-decider method models a developer-greedy deployment: pick the "
        "isolated best single-U backend for each LLM node and keep using it in "
        "parallel waves. For Qwen1.5 on both Snapdragon devices, the isolated "
        "best backend is NPU, so w/o decider is the p=3 NPU same-U run.",
        "",
        "| Device | Method | Completed | Completed total (s) | Failed |",
        "|---|---|---:|---:|---:|",
    ]
    for device in QWEN15_DEVICE_ORDER:
        device_summary = summary["by_device"][device]
        readme_lines.append(
            f"| {device_summary['device_label']} | Oracle | "
            f"{device_summary['odabench_completed_count']}/10 | "
            f"{device_summary['odabench_total_s']:.1f} | 0 |"
        )
        readme_lines.append(
            f"| {device_summary['device_label']} | w/o decider | "
            f"{device_summary['wo_decider_completed_count']}/10 | "
            f"{device_summary['wo_decider_completed_total_s']:.1f} | "
            f"{device_summary['wo_decider_failed_count']} |"
        )
    readme_lines.extend(
        [
            "",
            "`ERR` marks failed workflows and should not be interpreted as zero latency.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def build_decider_workflow_ablation() -> dict[str, Any]:
    oracle_rows = read_csv_rows(CONFIG.fig6_latency_csv)
    p3_rows = read_csv_rows(CONFIG.decider_p3_same_u_summary_csv)
    p3_npu_by_key = {
        (row["device"], as_int(row["workflow_index"])): row
        for row in p3_rows
        if row.get("model_id") == "qwen1_5_1_8b_chat_q4_0"
        and row.get("framework_id") == "llama_cpp_npu_p3_same_u"
    }

    rows: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        workflow_index = as_int(oracle["workflow_index"])
        for device in QWEN15_DEVICE_ORDER:
            rows.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_id": oracle["workflow_id"],
                    "workflow_label": oracle["workflow_label"],
                    "device": device,
                    "device_label": DEVICE_LABELS[device],
                    "model_id": "qwen1_5_1_8b_chat_q4_0",
                    "model_label": "Qwen1.5 1.8B",
                    "method": "odabench",
                    "method_label": "Oracle",
                    "latency_s": as_float(oracle[f"oracle_{device}_s"]),
                    "partial_wall_s": as_float(oracle[f"oracle_{device}_s"]),
                    "status": "completed",
                    "marker": "",
                    "developer_greedy_backend": "",
                    "framework_id": "qwen1.5_oracle_wave_20260628",
                    "placements": "",
                    "llm_failed_count": 0,
                    "llm_node_count": "",
                    "failure_reason": "",
                }
            )
            p3 = p3_npu_by_key.get((device, workflow_index))
            if p3 is None:
                raise ValueError(
                    f"Missing p=3 NPU same-U row for {device} workflow {workflow_index}"
                )
            status = p3.get("status", "")
            completed = status == "completed" and as_int(p3.get("llm_failed_count")) == 0
            p3_wall_s = as_float(p3.get("wall_ms")) / 1000.0
            rows.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_id": oracle["workflow_id"],
                    "workflow_label": oracle["workflow_label"],
                    "device": device,
                    "device_label": DEVICE_LABELS[device],
                    "model_id": "qwen1_5_1_8b_chat_q4_0",
                    "model_label": "Qwen1.5 1.8B",
                    "method": "wo_decider",
                    "method_label": "w/o decider",
                    "latency_s": p3_wall_s if completed else math.nan,
                    "partial_wall_s": p3_wall_s,
                    "status": status,
                    "marker": "" if completed else "ERR",
                    "developer_greedy_backend": "NPU",
                    "framework_id": p3.get("framework_id", ""),
                    "placements": p3.get("placements", ""),
                    "llm_failed_count": as_int(p3.get("llm_failed_count")),
                    "llm_node_count": as_int(p3.get("llm_node_count")),
                    "failure_reason": p3.get("failure_reason", ""),
                }
            )

    by_device: dict[str, dict[str, Any]] = {}
    for device in QWEN15_DEVICE_ORDER:
        odabench_rows = [
            row
            for row in rows
            if row["device"] == device and row["method"] == "odabench"
        ]
        wo_rows = [
            row
            for row in rows
            if row["device"] == device and row["method"] == "wo_decider"
        ]
        wo_completed = [row for row in wo_rows if row["status"] == "completed"]
        wo_failed = [row for row in wo_rows if row["status"] != "completed"]
        by_device[device] = {
            "device": device,
            "device_label": DEVICE_LABELS[device],
            "developer_greedy_backend": "NPU",
            "odabench_completed_count": len(odabench_rows),
            "odabench_total_s": sum(row["latency_s"] for row in odabench_rows),
            "wo_decider_completed_count": len(wo_completed),
            "wo_decider_failed_count": len(wo_failed),
            "wo_decider_completed_total_s": sum(row["latency_s"] for row in wo_completed),
            "wo_decider_failed_partial_wall_s": sum(row["partial_wall_s"] for row in wo_failed),
            "failed_workflows": [
                {
                    "workflow_index": row["workflow_index"],
                    "workflow_label": row["workflow_label"],
                    "partial_wall_s": row["partial_wall_s"],
                    "failure_reason": row["failure_reason"],
                }
                for row in wo_failed
            ],
        }

    summary = {
        "workflow_count": len(oracle_rows),
        "workflow_device_count": len(oracle_rows) * len(QWEN15_DEVICE_ORDER),
        "wo_decider_completed_workflow_count": sum(
            item["wo_decider_completed_count"] for item in by_device.values()
        ),
        "wo_decider_failed_workflow_count": sum(
            item["wo_decider_failed_count"] for item in by_device.values()
        ),
        "by_device": by_device,
        "totals": {
            "odabench_total_s": sum(
                item["odabench_total_s"] for item in by_device.values()
            ),
            "wo_decider_completed_total_s": sum(
                item["wo_decider_completed_total_s"] for item in by_device.values()
            ),
            "wo_decider_failed_partial_wall_s": sum(
                item["wo_decider_failed_partial_wall_s"] for item in by_device.values()
            ),
        },
        "sources": {
            "oracle_csv": str(CONFIG.fig6_latency_csv),
            "p3_same_u_summary_csv": str(CONFIG.decider_p3_same_u_summary_csv),
            "figure_data_dir": str(CONFIG.decider_qwen15_figure_data_dir),
        },
    }
    write_decider_ablation_figure_data(rows, summary)
    return {
        "rows": rows,
        "summary": summary,
        "sources": summary["sources"],
    }


def build_metrics() -> dict[str, Any]:
    wave_opt_rows = read_csv_rows(CONFIG.wave_opt_csv)
    system_rows = read_csv_rows(CONFIG.system_performance_csv)
    method_rows = {
        "naive": select_method_rows(system_rows, "round_robin_p1"),
        "odabench": sorted(wave_opt_rows, key=case_sort_key),
    }
    validate_same_cases(method_rows)
    latency_rows, latency_sources = build_latency_rows(method_rows)

    qwen15_rows = read_fig6_qwen15_latency_rows(CONFIG.fig6_latency_csv)
    latency_cases = read_csv_rows(CONFIG.fig6_latency_cases_csv)
    latency_summary = read_json(CONFIG.fig6_latency_summary_json)

    naive = summarize_series(latency_rows, "naive")
    odabench = summarize_series(latency_rows, "odabench")
    workflow_count = len(latency_rows)

    seq_cpu = summarize_series(qwen15_rows, "seq_cpu")
    seq_gpu = summarize_series(qwen15_rows, "seq_gpu")
    seq_npu = summarize_series(qwen15_rows, "seq_npu")
    oracle = summarize_series(qwen15_rows, "oracle")
    best_single_u_key, best_single_u = min(
        [("seq_cpu", seq_cpu), ("seq_gpu", seq_gpu), ("seq_npu", seq_npu)],
        key=lambda item: item[1]["total_s"],
    )

    metrics = {
        "dataset": "fig9_12-llama_cpp_wave_opt_20260623",
        "workflow_count": workflow_count,
        "case_count": 6,
        "rows": latency_rows,
        "latency": {
            "rows": latency_rows,
            "sources": latency_sources,
            "naive": naive,
            "odabench": odabench,
        },
        "naive": naive,
        "odabench": odabench,
        "improvement": {
            "speedup_naive_vs_odabench_x": naive["total_s"] / odabench["total_s"],
            "latency_reduction_naive_vs_odabench_pct": 100.0
            * (naive["total_s"] - odabench["total_s"])
            / naive["total_s"],
        },
        "method_mapping": {
            "naive": "round_robin_p1",
            "odabench": "wave_opt",
        },
        "method_definitions": {
            "naive": (
                "Linear p=1 non-optimized deployment from the comparison set "
                "(round_robin_p1)."
            ),
            "odabench": (
                "Per-device/per-model MOBench wave-opt selection from corrected "
                "p=2 and p=3 runs."
            ),
        },
        "single_u_oracle_reference": {
            "dataset": "fig6-qwen1_5_1_8b_oracle_20260628",
            "model_id": "qwen1_5_1_8b_chat_q4_0",
            "model_label": "Qwen1.5 1.8B",
            "workflow_count": len(qwen15_rows),
            "case_count": 2,
            "rows": qwen15_rows,
            "seq_cpu": seq_cpu,
            "seq_gpu": seq_gpu,
            "seq_npu": seq_npu,
            "oracle": oracle,
            "improvement": {
                "speedup_seq_cpu_vs_oracle_x": seq_cpu["total_s"] / oracle["total_s"],
                "latency_reduction_seq_cpu_vs_oracle_pct": 100.0
                * (seq_cpu["total_s"] - oracle["total_s"])
                / seq_cpu["total_s"],
                "speedup_seq_gpu_vs_oracle_x": seq_gpu["total_s"] / oracle["total_s"],
                "latency_reduction_seq_gpu_vs_oracle_pct": 100.0
                * (seq_gpu["total_s"] - oracle["total_s"])
                / seq_gpu["total_s"],
                "speedup_seq_npu_vs_oracle_x": seq_npu["total_s"] / oracle["total_s"],
                "latency_reduction_seq_npu_vs_oracle_pct": 100.0
                * (seq_npu["total_s"] - oracle["total_s"])
                / seq_npu["total_s"],
                "best_single_u_method": best_single_u_key,
                "speedup_best_single_u_vs_oracle_x": (
                    best_single_u["total_s"] / oracle["total_s"]
                ),
                "latency_reduction_best_single_u_vs_oracle_pct": 100.0
                * (best_single_u["total_s"] - oracle["total_s"])
                / best_single_u["total_s"],
            },
        },
        "sources": {
            "fig6_latency_csv": str(CONFIG.fig6_latency_csv),
            "fig6_latency_cases_csv": str(CONFIG.fig6_latency_cases_csv),
            "fig6_latency_summary_json": str(CONFIG.fig6_latency_summary_json),
            "fig6_latency_cases": latency_cases,
            "fig6_latency_summary": latency_summary,
            "method_time_csv": str(CONFIG.method_time_csv),
            "system_performance_csv": str(CONFIG.system_performance_csv),
            "wave_opt_csv": str(CONFIG.wave_opt_csv),
            "summary_markdown": str(CONFIG.summary_markdown),
        },
    }
    ablation_studies = build_ablation_studies()
    metrics["resource_usage"] = build_resource_usage(system_rows, wave_opt_rows)
    metrics["structure_ablation"] = ablation_studies["structure_ablation"]
    metrics["decider_ablation"] = build_decider_workflow_ablation()
    return metrics


def plot_latency_pair(
    metrics: dict[str, Any],
    *,
    left_key: str,
    left_label: str,
    left_style: str,
    right_key: str,
    right_label: str,
    right_style: str,
    output_name: str,
    preview_dir: Path | None = None,
) -> None:
    configure_style()
    colors = CONFIG.colors
    rows = metrics["rows"]
    n = len(rows)
    x = list(range(n))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=STUDY_SINGLE_FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.21, top=0.79)

    left_values = [row[f"{left_key}_s"] for row in rows]
    right_values = [row[f"{right_key}_s"] for row in rows]

    ax.bar(
        [pos - bar_w / 2 for pos in x],
        left_values,
        width=bar_w,
        color=figure_style.bar_color(left_key),
        edgecolor=figure_style.BAR_EDGE,
        linewidth=figure_style.BAR_EDGE_WIDTH,
        label=left_label,
    )
    ax.bar(
        [pos + bar_w / 2 for pos in x],
        right_values,
        width=bar_w,
        color=figure_style.bar_color(right_key),
        edgecolor=figure_style.BAR_EDGE,
        linewidth=figure_style.BAR_EDGE_WIDTH,
        label=right_label,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Execution latency (s)")
    figure_style.box_axes(ax, grid_axis="y")
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, max(left_values + right_values) * 1.08)

    ax.tick_params(axis="both", length=2.5)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend(
        fig,
        handles,
        labels,
        ncol=2,
        anchor=(0.60, 0.99),
        columnspacing=0.9,
        handlelength=0.9,
        handletextpad=0.3,
        prop={"size": STUDY_LEGEND_FONT},
    )

    save_figure(fig, CONFIG.output_dir / output_name, preview_dir)


def build_phone_latency(run: Path) -> dict[str, Any]:
    """Freeze derived figure data while leaving raw phone measurements local."""
    indices = [120, 7, 33, 100, 186, 213, 239, 48, 152, 179]
    candidates = ("naive", "wave2", "wave3", "max_rr", "max_wave")
    parallelism = {"naive": 1, "wave2": 2, "wave3": 3, "max_rr": 4, "max_wave": 4}
    methods = {key: [] for key in ("naive", "odabench", "max_rr", "max_wave")}
    sources, selection = {}, []
    for device in DEVICE_ORDER:
        path = run / device / "summary.json"
        summary = read_json(path)
        if not summary["complete"] or len(summary["rows"]) != 150:
            raise ValueError(f"Incomplete phone suite: {device}")
        raw = summary["rows"]
        keys = {(r["model_id"], r["config"], r["source_index"]) for r in raw}
        expected = {(m, c, i) for m in MODEL_ORDER
                    for c in candidates for i in indices}
        if len(keys) != 150 or keys != expected:
            raise ValueError("Duplicate or missing phone/model/config/workflow cases")
        sources[device] = {"path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for model in MODEL_ORDER:
            for index in indices:
                options = [r for r in raw if r["model_id"] == model
                           and r["source_index"] == index and r["config"] in candidates]
                feasible = [r for r in options if r["success"]]
                if not feasible:
                    raise ValueError(f"No successful candidate for {device}/{model}/{index}")
                chosen = min(feasible, key=lambda r: (r["wall_ms"], candidates.index(r["config"])))
                methods["odabench"].append(chosen)
                selection.append({"device": device, "model_id": model,
                                  "source_index": index, "workflow_id": chosen["workflow_id"],
                                  "configuration": chosen["config"], "p": parallelism[chosen["config"]],
                                  "wall_ms": chosen["wall_ms"], "successful": 1, "total": 1,
                                  "candidate_count": len(options), "successful_candidates": len(feasible)})
        for key in ("naive", "max_rr", "max_wave"):
            methods[key].extend(r for r in raw if r["config"] == key)
    rows = []
    for position, index in enumerate(indices):
        first = next(r for r in methods["naive"] if r["source_index"] == index)
        row = {"workflow_index": position, "source_index": index, "workflow_id": first["workflow_id"],
               "label": workflow_short_label(first["workflow_id"], index)}
        for key, runs in methods.items():
            group = [r for r in runs if r["source_index"] == index]
            assert len(group) == 6
            row[f"{key}_sample_count"] = len(group)
            row[f"{key}_successful"] = sum(r["success"] for r in group)
            row[f"{key}_case_values_s"] = [r["wall_ms"] / 1000 for r in group]
            row[f"{key}_case_successes"] = [r["success"] for r in group]
            row[f"{key}_attempt_s"] = mean(row[f"{key}_case_values_s"])
            row[f"{key}_s"] = row[f"{key}_attempt_s"] if all(r["success"] for r in group) else None
        row["speedup_vs_naive_x"] = row["naive_s"] / row["odabench_s"]
        row["reduction_vs_naive_pct"] = (1 - row["odabench_s"] / row["naive_s"]) * 100
        rows.append(row)
    naive, ours = summarize_series(rows, "naive"), summarize_series(rows, "odabench")
    definitions = {
        "naive": "p=1; LLM CPU/GPU/NPU round robin within each workflow.",
        "max_rr": "p=4; the same round-robin placement and dependency-wave barriers.",
        "max_wave": "p=4; model-specific backend priority order restarted in each wave.",
        "odabench": "Per phone/model/workflow, choose the lowest-latency successful measured attempt among Sequential p=1, wave p=2, wave p=3, max_rr p=4, max_wave p=4. Same-data offline selection; search cost excluded.",
    }
    return {"dataset": run.name, "workflow_count": 10, "case_count": 6, "rows": rows,
            "naive": naive, "odabench": ours,
            "latency": {"rows": rows, "sources": sources, "naive": naive, "odabench": ours,
                        "selection": selection, "selection_unit": "phone/model/workflow",
                        "candidate_configurations": list(candidates),
                        "success_counts": {k: {"successful": sum(r["success"] for r in rs), "total": len(rs)} for k, rs in methods.items()},
                        "aggregation": "Arithmetic mean over the same two phones and three models. Any failed case suppresses that group's complete-latency bar; ERR reports successful/total.",
                        "scope": "Observed success after offline selection on these same workflows, not independent validation of an unseen-workload guarantee."},
            "improvement": {"speedup_naive_vs_odabench_x": naive["total_s"] / ours["total_s"],
                            "latency_reduction_naive_vs_odabench_pct": (1-ours["total_s"]/naive["total_s"])*100},
            "method_mapping": {"naive": "round_robin_p1", "odabench": "per_workflow_successful_min_p1234",
                               "max_rr": "round_robin_p4", "max_wave": "wave_aware_p4"},
            "method_definitions": definitions}


def plot_execution_latency(
    metrics: dict[str, Any], preview_dir: Path | None = None
) -> None:
    configure_style()
    colors = CONFIG.colors
    rows = metrics["latency"]["rows"]
    n = len(rows)
    x = list(range(n))
    parallel = "max_rr_s" in rows[0]
    bar_w = 0.18 if parallel else 0.32
    methods = [("naive", "Sequential"), ("odabench", "MOBench")]
    if parallel:
        methods.extend([("max_rr", "Max-parallel + RR"), ("max_wave", "HeRo")])
    specs = [(key, label, (i-(len(methods)-1)/2)*bar_w) for i, (key, label) in enumerate(methods)]

    # Draw at the single-column page width instead of enlarging a column plot.
    fig, ax = plt.subplots(figsize=(5.35, 1.65))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.25, top=0.81)

    all_values: list[float] = []
    for key, label, offset in specs:
        values = [float("nan") if row[f"{key}_s"] is None else row[f"{key}_s"] for row in rows]
        all_values.extend(value for value in values if math.isfinite(value))
        ax.bar(
            [pos + offset for pos in x],
            values,
            width=bar_w,
            color=figure_style.bar_color(key),
            edgecolor=figure_style.BAR_EDGE,
            linewidth=figure_style.BAR_EDGE_WIDTH,
            label=label,
        )

    ymax = max(all_values) * 1.10
    if parallel:
        for key, _, offset in specs:
            for position, row in enumerate(rows):
                if row[f"{key}_s"] is None:
                    ax.plot(position+offset, ymax*.025, marker="x", color="#B83232", markersize=3, markeredgewidth=.8)
                    error_y = ymax * (.32 if key == "max_rr" else .58)
                    ax.text(position+offset, error_y, f"ERR {row[f'{key}_successful']}/6", color="#B83232",
                            rotation=90, ha="center", va="bottom", fontsize=6.3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows], fontsize=STUDY_FONT["tick"])
    ax.set_xlabel("Workflow")
    ax.set_ylabel("Execution latency (s)")
    study_box_axes(ax)
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, ymax)

    ax.tick_params(axis="both", length=2.5)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=len(methods),
        columnspacing=0.8,
        handlelength=0.9,
        handletextpad=0.3,
        prop={"size": STUDY_LEGEND_FONT},
    )

    save_figure(fig, CONFIG.output_dir / "execution_latency", preview_dir,
                exact_canvas=True, top_margin_points=2.0)


def plot_structure_ablation(
    metrics: dict[str, Any], preview_dir: Path | None = None
) -> None:
    configure_style()
    colors = CONFIG.colors
    rows = metrics["structure_ablation"]["rows"]
    n = len(rows)
    x = list(range(n))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=STUDY_SINGLE_FIGSIZE)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.29, top=0.78)

    odabench_values = [row["odabench_total_wall_min"] for row in rows]
    blind_p3_values = [row["blind_p3_total_wall_min"] for row in rows]

    ax.bar(
        [pos - bar_w / 2 for pos in x],
        odabench_values,
        width=bar_w,
        color=figure_style.bar_color("odabench"),
        edgecolor=figure_style.BAR_EDGE,
        linewidth=figure_style.BAR_EDGE_WIDTH,
        label="MOBench",
        zorder=2,
    )
    ax.bar(
        [pos + bar_w / 2 for pos in x],
        blind_p3_values,
        width=bar_w,
        color=figure_style.bar_color("blind_p3"),
        edgecolor=figure_style.BAR_EDGE,
        linewidth=figure_style.BAR_EDGE_WIDTH,
        label="blind p=3",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows])
    ax.set_xlabel("Device-model case")
    ax.set_ylabel("Total latency (min)")
    ax.set_xlim(-0.55, n - 0.45)
    ax.set_ylim(0, max(odabench_values + blind_p3_values) * 1.12)
    study_box_axes(ax)
    ax.tick_params(axis="both", length=2.5)

    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend(
        fig,
        handles,
        labels,
        ncol=2,
        anchor=(0.60, 0.99),
        columnspacing=0.9,
        handlelength=0.9,
        handletextpad=0.3,
        prop={"size": STUDY_LEGEND_FONT},
    )

    save_figure(
        fig,
        CONFIG.output_dir / "execution_structure_ablation",
        preview_dir,
        exact_canvas=True,
    )


def plot_decider_ablation(
    metrics: dict[str, Any], preview_dir: Path | None = None
) -> None:
    configure_style()
    colors = CONFIG.colors
    rows = metrics["decider_ablation"]["rows"]
    workflow_labels = [
        row["workflow_label"]
        for row in rows
        if row["device"] == QWEN15_DEVICE_ORDER[0] and row["method"] == "odabench"
    ]
    workflow_indices = [
        row["workflow_index"]
        for row in rows
        if row["device"] == QWEN15_DEVICE_ORDER[0] and row["method"] == "odabench"
    ]
    x = list(range(len(workflow_labels)))
    bar_w = 0.26
    completed_values = [
        row["latency_s"]
        for row in rows
        if row["status"] == "completed" and not math.isnan(row["latency_s"])
    ]
    ymax = max(completed_values) * 1.18
    marker_y = ymax * 0.055
    fail_color = figure_style.DESIGN_COLORS["red"]

    fig, axes = plt.subplots(2, 1, figsize=STUDY_FIGSIZE, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.10, top=0.84, hspace=0.36)

    for ax, device in zip(axes, QWEN15_DEVICE_ORDER):
        for method, label, offset, color in [
            ("odabench", "Oracle", -bar_w / 2, figure_style.bar_color("oracle")),
            ("wo_decider", "w/o decider", bar_w / 2, figure_style.bar_color("wo_decider")),
        ]:
            method_rows = [
                row
                for row in rows
                if row["device"] == device and row["method"] == method
            ]
            method_by_workflow = {
                row["workflow_index"]: row for row in method_rows
            }
            ordered_rows = [method_by_workflow[index] for index in workflow_indices]
            values = [
                0.0 if math.isnan(row["latency_s"]) else row["latency_s"]
                for row in ordered_rows
            ]
            ax.bar(
                [pos + offset for pos in x],
                values,
                width=bar_w,
                color=color,
                edgecolor=figure_style.BAR_EDGE,
                linewidth=figure_style.BAR_EDGE_WIDTH,
                label=label,
                zorder=2,
            )
            for pos, row in zip(x, ordered_rows):
                if row["status"] == "completed":
                    continue
                ax.scatter(
                    pos + offset,
                    marker_y,
                    marker="X",
                    s=28,
                    facecolor=fail_color,
                    edgecolor=colors["white"],
                    linewidth=0.4,
                    zorder=4,
                )

        ax.set_title(DEVICE_LABELS[device], pad=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(workflow_labels)
        ax.set_xlim(-0.58, len(workflow_labels) - 0.42)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Latency (s)")
        study_box_axes(ax)
        ax.tick_params(axis="both", length=2.5, labelbottom=True)

    handles: list[Any] = [
        Patch(facecolor=figure_style.bar_color("oracle"), edgecolor=figure_style.BAR_EDGE, linewidth=figure_style.BAR_EDGE_WIDTH, label="Oracle"),
        Patch(
            facecolor=figure_style.bar_color("wo_decider"),
            edgecolor=figure_style.BAR_EDGE,
            linewidth=figure_style.BAR_EDGE_WIDTH,
            label="w/o decider",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            markersize=4.0,
            linestyle="",
            markerfacecolor=fail_color,
            markeredgecolor=colors["white"],
            label="ERR",
        ),
    ]
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        [handle.get_label() for handle in handles],
        ncol=3,
        columnspacing=0.7,
        handlelength=0.75,
        handletextpad=0.25,
        prop={"size": STUDY_LEGEND_FONT},
    )

    save_figure(
        fig,
        CONFIG.output_dir / "execution_decider_ablation",
        preview_dir,
        exact_canvas=True,
        top_margin_points=2.0,
    )


RESOURCE_TIMELINE_DEVICE = "gen3"
RESOURCE_TIMELINE_MODEL = "qwen1_5_1_8b_chat_q4_0"
RESOURCE_TIMELINE_WORKFLOW = 8
RESOURCE_TIMELINE_REPEAT_ROOT = (
    WFGEN_ROOT / "results" / "fig11_qwen15_a200_resource_repeats_8gen3_20260629"
)
RESOURCE_TIMELINE_REPEAT_METHODS = {
    "naive": {
        "glob": "naive_r*",
        "framework": "llama_cpp_round_robin",
        "label": "Sequential",
        "max_parallel_nodes": 1,
    },
    "odabench": {
        "glob": "odabench_r*",
        "framework": "llama_cpp_wave_cpu_npu",
        "label": "MOBench",
        "max_parallel_nodes": 3,
    },
}
RESOURCE_TIMELINE_CALLOUT_COLOR = "#ff4a4a"
RESOURCE_TIMELINE_CALLOUTS = [
    {
        "panel": 0,
        "box": (5.0, 45.0, 48.0, 57.0),
        "label": "high parallel",
        "label_xy": (59.0, 18.0),
    },
    {
        "panel": 1,
        "box": (5.0, 0.0, 48.0, 43.0),
        "label": "tasks lack GPU affinity",
        "label_xy": (59.0, 31.0),
    },
    {
        "panel": 2,
        "box": (5.0, 5.0, 48.0, 6.0),
        "label": "high parallel",
        "label_xy": (59.0, 8.9),
    },
]


def clip_percent(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return max(0.0, min(100.0, value))


def parse_gpu_busy(row: dict[str, str]) -> float:
    gpu = clip_percent(as_float(row.get("gpu_busy_percent")))
    if not math.isnan(gpu):
        return gpu
    if row.get("gpu_busy_counter") in {"0", "0.0"} and row.get("gpu_total_counter") in {
        "0",
        "0.0",
    }:
        return 0.0
    return math.nan


def read_workflow_resource_timeline(
    point: dict[str, Any],
    workflow_index: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    resource_dir = Path(point["resource_summary"]).parent
    return read_workflow_resource_timeline_from_resource_dir(resource_dir, workflow_index)


def read_workflow_resource_timeline_from_resource_dir(
    resource_dir: Path,
    workflow_index: int,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    samples_csv = resource_dir / "resource_samples.csv"
    by_workflow = read_json(resource_dir / "resource_by_workflow.json")
    window = by_workflow[str(workflow_index)]
    start_s = float(window["started_epoch_s"])
    finish_s = float(window["finished_epoch_s"])
    rows = read_csv_rows(samples_csv)
    rows.sort(key=lambda row: as_float(row.get("timestamp_epoch_s")))
    samples: list[dict[str, float]] = []
    for row in rows:
        timestamp = as_float(row.get("timestamp_epoch_s"))
        if math.isnan(timestamp) or timestamp < start_s or timestamp > finish_s:
            continue
        mem_mb = as_float(row.get("mem_used_mb"))
        samples.append(
            {
                "elapsed_s": timestamp - start_s,
                "cpu_pct": clip_percent(as_float(row.get("cpu_util_percent"))),
                "gpu_pct": parse_gpu_busy(row),
                "mem_gb": mem_mb / 1024.0 if not math.isnan(mem_mb) else math.nan,
            }
        )
    return (
        {
            "workflow_index": workflow_index,
            "workflow_id": window.get("workflow_id", ""),
            "workflow_label": workflow_short_label(window.get("workflow_id", ""), workflow_index),
            "started_epoch_s": start_s,
            "finished_epoch_s": finish_s,
            "duration_s": finish_s - start_s,
            "resource_samples_csv": str(samples_csv),
        },
        samples,
    )


def finite_stat(values: np.ndarray) -> tuple[float, float, float, float, int]:
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    if count == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if count > 1 else 0.0
    return mean, std, mean - std, mean + std, count


def resample_resource_runs(
    runs: list[dict[str, Any]],
    metric: str,
    *,
    clamp_percent: bool,
) -> dict[str, list[float]]:
    max_duration = max(run["workflow"]["duration_s"] for run in runs)
    x_grid = np.linspace(0.0, max_duration, 120)
    interpolated: list[np.ndarray] = []
    for run in runs:
        points = [
            (sample["elapsed_s"], sample[metric])
            for sample in run["samples"]
            if math.isfinite(sample.get(metric, math.nan))
        ]
        if len(points) < 2:
            interpolated.append(np.full_like(x_grid, np.nan, dtype=float))
            continue
        x = np.array([point[0] for point in points], dtype=float)
        y = np.array([point[1] for point in points], dtype=float)
        order = np.argsort(x)
        values = np.interp(x_grid, x[order], y[order], left=np.nan, right=np.nan)
        interpolated.append(values)

    stack = np.vstack(interpolated)
    means: list[float] = []
    stds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    counts: list[int] = []
    for column in stack.T:
        mean, std, lower, upper, count = finite_stat(column)
        if clamp_percent and math.isfinite(lower):
            lower = clip_percent(lower)
        if clamp_percent and math.isfinite(upper):
            upper = clip_percent(upper)
        means.append(mean)
        stds.append(std)
        lowers.append(lower)
        uppers.append(upper)
        counts.append(count)
    return {
        "elapsed_s": [float(value) for value in x_grid],
        "mean": means,
        "std": stds,
        "lower": lowers,
        "upper": uppers,
        "sample_count": counts,
    }


def build_repeated_workflow_resource_timeline() -> dict[str, Any] | None:
    if not RESOURCE_TIMELINE_REPEAT_ROOT.exists():
        return None
    methods: dict[str, Any] = {}
    workflow_meta: dict[str, Any] | None = None
    for method_id, method_config in RESOURCE_TIMELINE_REPEAT_METHODS.items():
        runs: list[dict[str, Any]] = []
        for repeat_dir in sorted(RESOURCE_TIMELINE_REPEAT_ROOT.glob(method_config["glob"])):
            resource_dir = (
                repeat_dir
                / RESOURCE_TIMELINE_DEVICE
                / method_config["framework"]
                / RESOURCE_TIMELINE_MODEL
                / "execution"
                / "resources"
            )
            summary_csv = repeat_dir / "summary.csv"
            if not (resource_dir / "resource_samples.csv").exists():
                continue
            current_workflow, samples = read_workflow_resource_timeline_from_resource_dir(
                resource_dir,
                RESOURCE_TIMELINE_WORKFLOW,
            )
            workflow_meta = current_workflow if workflow_meta is None else workflow_meta
            summary_row = read_csv_rows(summary_csv)[0] if summary_csv.exists() else {}
            runs.append(
                {
                    "repeat": repeat_dir.name,
                    "run_dir": str(repeat_dir),
                    "summary_csv": str(summary_csv),
                    "status": summary_row.get("status", ""),
                    "wall_s": as_float(summary_row.get("wall_ms")) / 1000.0,
                    "resource_samples_csv": str(resource_dir / "resource_samples.csv"),
                    "resource_by_workflow": str(resource_dir / "resource_by_workflow.json"),
                    "perfetto_trace": str(resource_dir / "wfgen_resource_trace.perfetto-trace"),
                    "workflow": current_workflow,
                    "samples": samples,
                }
            )
        if len(runs) < 2:
            return None
        metrics = {
            "cpu_pct": resample_resource_runs(runs, "cpu_pct", clamp_percent=True),
            "gpu_pct": resample_resource_runs(runs, "gpu_pct", clamp_percent=True),
            "mem_gb": resample_resource_runs(runs, "mem_gb", clamp_percent=False),
        }
        methods[method_id] = {
            "label": method_config["label"],
            "framework": method_config["framework"],
            "max_parallel_nodes": method_config["max_parallel_nodes"],
            "repeat_count": len(runs),
            "duration_s_mean": sum(run["workflow"]["duration_s"] for run in runs)
            / len(runs),
            "duration_s_min": min(run["workflow"]["duration_s"] for run in runs),
            "duration_s_max": max(run["workflow"]["duration_s"] for run in runs),
            "runs": runs,
            "metrics": metrics,
        }
    if workflow_meta is None:
        return None
    return {
        "mode": "repeated_workflow",
        "repeat_root": str(RESOURCE_TIMELINE_REPEAT_ROOT),
        "repeat_stat": "mean +/- 1 std",
        "device": RESOURCE_TIMELINE_DEVICE,
        "device_label": DEVICE_LABELS[RESOURCE_TIMELINE_DEVICE],
        "model_id": RESOURCE_TIMELINE_MODEL,
        "model_label": "Qwen1.5 1.8B",
        "workflow_index": RESOURCE_TIMELINE_WORKFLOW,
        "workflow_id": workflow_meta["workflow_id"],
        "workflow_label": workflow_meta["workflow_label"],
        "methods": methods,
        "note": (
            "Each method is repeated on the same A200/Qwen1.5 workflow. Lines show "
            "the interpolated mean over repeats and shaded bands show one sample "
            "standard deviation. GPU counter samples with zero busy and total "
            "increments are treated as idle samples."
        ),
    }


def write_resource_timeline_outputs(
    timeline: dict[str, Any],
    output_base: Path,
) -> None:
    csv_path = output_base.with_name(output_base.name + "_timeline.csv")
    json_path = output_base.with_name(output_base.name + "_timeline_summary.json")
    if timeline.get("mode") == "single_workflow":
        fields = [
            "method",
            "model_id",
            "workflow_index",
            "workflow_label",
            "elapsed_s",
            "cpu_pct",
            "gpu_pct",
            "mem_gb",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method_id, method in timeline["methods"].items():
                for sample in method["samples"]:
                    writer.writerow(
                        {
                            "method": method_id,
                            "model_id": timeline["model_id"],
                            "workflow_index": timeline["workflow_index"],
                            "workflow_label": timeline["workflow_label"],
                            "elapsed_s": sample["elapsed_s"],
                            "cpu_pct": sample["cpu_pct"],
                            "gpu_pct": sample["gpu_pct"],
                            "mem_gb": sample["mem_gb"],
                        }
                    )
        json_path.write_text(json.dumps(json_safe(timeline), indent=2), encoding="utf-8")
        return

    if timeline.get("mode") == "repeated_workflow":
        fields = [
            "method",
            "metric",
            "elapsed_s",
            "mean",
            "std",
            "lower",
            "upper",
            "sample_count",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method_id, method in timeline["methods"].items():
                for metric, values in method["metrics"].items():
                    for index, elapsed_s in enumerate(values["elapsed_s"]):
                        writer.writerow(
                            {
                                "method": method_id,
                                "metric": metric,
                                "elapsed_s": elapsed_s,
                                "mean": values["mean"][index],
                                "std": values["std"][index],
                                "lower": values["lower"][index],
                                "upper": values["upper"][index],
                                "sample_count": values["sample_count"][index],
                            }
                        )
        json_path.write_text(json.dumps(json_safe(timeline), indent=2), encoding="utf-8")
        return

    fields = ["method", "metric", "x_pct", "mean", "min", "max"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method_id, method in timeline["methods"].items():
            for metric, values in method["metrics"].items():
                for index, x_pct in enumerate(timeline["x_pct"]):
                    writer.writerow(
                        {
                            "method": method_id,
                            "metric": metric,
                            "x_pct": x_pct,
                            "mean": values["mean"][index],
                            "min": values["min"][index],
                            "max": values["max"][index],
                        }
                    )
    json_path.write_text(json.dumps(json_safe(timeline), indent=2), encoding="utf-8")


def build_single_workflow_resource_timeline(resources: dict[str, Any]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    workflow_meta: dict[str, Any] | None = None
    for method_id in ("naive", "odabench"):
        point = next(
            point
            for point in resources[method_id]["points"]
            if point.get("device") == RESOURCE_TIMELINE_DEVICE
            and point.get("model_id") == RESOURCE_TIMELINE_MODEL
        )
        current_workflow, samples = read_workflow_resource_timeline(
            point,
            RESOURCE_TIMELINE_WORKFLOW,
        )
        workflow_meta = current_workflow if workflow_meta is None else workflow_meta
        methods[method_id] = {
            "label": "Sequential" if method_id == "naive" else "MOBench",
            "point": {
                "device": point["device"],
                "device_label": point["device_label"],
                "model_id": point["model_id"],
                "model_label": point["model_label"],
                "resource_summary": point["resource_summary"],
                "perfetto_trace": point["perfetto_trace"],
            },
            "workflow": current_workflow,
            "samples": samples,
        }
    assert workflow_meta is not None
    return {
        "mode": "single_workflow",
        "device": RESOURCE_TIMELINE_DEVICE,
        "device_label": DEVICE_LABELS[RESOURCE_TIMELINE_DEVICE],
        "model_id": RESOURCE_TIMELINE_MODEL,
        "model_label": "Qwen1.5 1.8B",
        "workflow_index": RESOURCE_TIMELINE_WORKFLOW,
        "workflow_id": workflow_meta["workflow_id"],
        "workflow_label": workflow_meta["workflow_label"],
        "methods": methods,
        "note": (
            "This figure intentionally uses one matched workflow trace instead of "
            "averaging across workflows, because workflow DAG shapes and execution "
            "phases differ. The selected A200/Qwen1.5 case is drawn from Fig. 11's "
            "existing resource provenance on Snapdragon 8 Gen 3."
        ),
    }


def plot_execution_resource_usage(
    metrics: dict[str, Any], preview_dir: Path | None = None
) -> None:
    configure_style()
    colors = CONFIG.colors
    timeline = build_repeated_workflow_resource_timeline()
    if timeline is None:
        timeline = build_single_workflow_resource_timeline(metrics["resource_usage"]["methods"])
    output_base = CONFIG.output_dir / "execution_resource_usage"
    write_resource_timeline_outputs(timeline, output_base)

    fig, axes = plt.subplots(3, 1, figsize=(3.45, 2.12), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.15, top=0.82, hspace=0.16)

    method_styles = {
        "naive": {
            "label": "Sequential",
            "color": colors["naive"],
            "linestyle": "-",
        },
        "odabench": {
            "label": "MOBench",
            "color": colors["odabench"],
            "linestyle": "-",
        },
    }
    panels = [
        ("CPU (%)", "cpu_pct", (-3, 103)),
        ("GPU (%)", "gpu_pct", (-3, 103)),
        ("Mem. (GB)", "mem_gb", None),
    ]

    handles: list[Any] = []
    labels: list[str] = []
    for ax, (ylabel, metric, ylim) in zip(axes, panels):
        for method_id in ("naive", "odabench"):
            style = method_styles[method_id]
            method = timeline["methods"][method_id]
            if timeline["mode"] == "repeated_workflow":
                values = method["metrics"][metric]
                x = values["elapsed_s"]
                y = values["mean"]
                ax.fill_between(
                    x,
                    values["lower"],
                    values["upper"],
                    color=style["color"],
                    alpha=0.16,
                    linewidth=0,
                    zorder=2,
                )
                marker = ""
                markersize = 0.0
            else:
                samples = method["samples"]
                x = [sample["elapsed_s"] for sample in samples]
                y = [sample[metric] for sample in samples]
                marker = "o"
                markersize = 1.7
            line = ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                marker=marker,
                markersize=markersize,
                markeredgewidth=0,
                clip_on=True,
                zorder=3,
            )[0]
            if ax is axes[0]:
                handles.append(line)
                labels.append(style["label"])
        ax.set_ylabel(ylabel, labelpad=1.4)
        figure_style.box_axes(ax, grid_axis="both")
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_yticks([0, 50, 100])
        else:
            if timeline["mode"] == "repeated_workflow":
                all_values = [
                    value
                    for method_id in ("naive", "odabench")
                    for bound in ("lower", "upper")
                    for value in timeline["methods"][method_id]["metrics"][metric][bound]
                    if math.isfinite(value)
                ]
            else:
                all_values = [
                    sample[metric]
                    for method_id in ("naive", "odabench")
                    for sample in timeline["methods"][method_id]["samples"]
                    if not math.isnan(sample[metric])
                ]
            ax.set_ylim(max(0.0, min(all_values) * 0.92), max(all_values) * 1.06)
            ax.set_yticks([5, 8, 11])

    draw_resource_timeline_callouts(axes)

    if timeline["mode"] == "repeated_workflow":
        max_elapsed = max(
            timeline["methods"][method_id]["duration_s_max"]
            for method_id in ("naive", "odabench")
        )
    else:
        max_elapsed = max(
            sample["elapsed_s"]
            for method_id in ("naive", "odabench")
            for sample in timeline["methods"][method_id]["samples"]
        )
    axes[-1].set_xlim(0, max_elapsed * 1.02)
    axes[-1].set_xlabel("Elapsed time (s)", labelpad=1.3)
    figure_style.boxed_legend(
        fig,
        handles,
        labels,
        ncol=2,
        anchor=(0.50, 0.965),
        columnspacing=0.7,
        handlelength=1.0,
        handletextpad=0.32,
        labelspacing=0.18,
        prop={"size": STUDY_LEGEND_FONT},
    )

    save_figure(fig, output_base, preview_dir, exact_canvas=True)


def draw_resource_timeline_callouts(axes: np.ndarray) -> None:
    for callout in RESOURCE_TIMELINE_CALLOUTS:
        ax = axes[callout["panel"]]
        x, y, width, height = callout["box"]
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=RESOURCE_TIMELINE_CALLOUT_COLOR,
                linewidth=1.7,
                zorder=4,
                clip_on=False,
            )
        )
        ax.text(
            *callout["label_xy"],
            callout["label"],
            color=RESOURCE_TIMELINE_CALLOUT_COLOR,
            fontsize=7.2,
            ha="left",
            va="center",
            zorder=5,
            clip_on=False,
        )


def save_figure(
    fig: plt.Figure,
    stem: Path,
    preview_dir: Path | None = None,
    *,
    exact_canvas: bool = False,
    top_margin_points: float | None = None,
) -> None:
    figure_style.save_pdf_svg(
        fig,
        stem,
        exact_canvas=exact_canvas,
        top_margin_points=top_margin_points,
        creator="Paper_MOBench execution figure generator",
        preview_dir=preview_dir,
    )
    plt.close(fig)
    strip_trailing_whitespace(stem.with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_provenance(metrics: dict[str, Any]) -> None:
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    path = CONFIG.output_dir / "execution_latency_provenance.json"
    path.write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG previews used in visual QA.",
    )
    parser.add_argument("--latency-only", action="store_true", help="Regenerate only the main latency figure and its provenance.")
    parser.add_argument("--latency-run", type=Path, help="Freeze new latency data from completed selected-ten phone measurements.")
    args = parser.parse_args()

    from refresh_evaluation_sources import SNAPSHOT as evaluation_snapshot, read, render
    if evaluation_snapshot.exists() and not args.latency_run:
        data = read(evaluation_snapshot)
        if args.latency_only:
            plot_execution_latency(data['latency'], preview_dir=args.preview_dir)
        else:
            render(data, args.preview_dir, only='execution')
        return

    snapshot = CONFIG.output_dir / "execution_latency_selected10.json"
    metrics = read_json(CONFIG.output_dir / "execution_latency_provenance.json") if args.latency_only else build_metrics()
    if args.latency_run:
        update = build_phone_latency(args.latency_run.resolve())
        snapshot.write_text(json.dumps(update, indent=2) + "\n", encoding="utf-8")
    elif snapshot.exists():
        update = read_json(snapshot)
    else:
        update = None
    if update:
        legacy_dataset = metrics.get("datasets_by_figure", {}).get("other_execution_figures", metrics["dataset"])
        metrics.update(update)
        metrics["datasets_by_figure"] = {"execution_latency": update["dataset"], "other_execution_figures": legacy_dataset}
    plot_execution_latency(metrics, preview_dir=args.preview_dir)
    if args.latency_only:
        write_provenance(metrics)
        print(f"Wrote {CONFIG.output_dir / 'execution_latency.pdf'}; other execution figures unchanged")
        return
    plot_structure_ablation(metrics, preview_dir=args.preview_dir)
    plot_decider_ablation(metrics, preview_dir=args.preview_dir)
    plot_execution_resource_usage(metrics, preview_dir=args.preview_dir)
    write_provenance(metrics)

    print(f"Wrote {CONFIG.output_dir / 'execution_latency.pdf'}")
    print(f"Wrote {CONFIG.output_dir / 'execution_structure_ablation.pdf'}")
    print(f"Wrote {CONFIG.output_dir / 'execution_decider_ablation.pdf'}")
    print(f"Wrote {CONFIG.output_dir / 'execution_resource_usage.pdf'}")
    print(
        "Execution latency Sequential vs MOBench: "
        f"{metrics['improvement']['speedup_naive_vs_odabench_x']:.2f}x faster, "
        f"{metrics['improvement']['latency_reduction_naive_vs_odabench_pct']:.1f}% lower latency; "
        f"blind p=3 is "
        f"{metrics['structure_ablation']['summary']['blind_p3_over_odabench']:.2f}x "
        "MOBench; "
        f"w/o decider failed "
        f"{metrics['decider_ablation']['summary']['wo_decider_failed_workflow_count']}/"
        f"{metrics['decider_ablation']['summary']['workflow_device_count']} "
        "workflow-device cases"
    )


if __name__ == "__main__":
    main()
