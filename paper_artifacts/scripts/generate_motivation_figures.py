"""Generate evidence figures for the motivation section.

The script reads WFGen experiment outputs, keeps only completed/comparable runs for
plots, and writes a small provenance file so the figure data can be audited.
"""

from __future__ import annotations

import json
import os
import csv
import tempfile
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_motivation_figures"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

try:
    from . import figure_style
except ImportError:
    import figure_style


REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_INPUT_DIR = REPO_ROOT / "data"
GITHUB_ROOT = REPO_ROOT.parent
WFGEN_ROOT = REPO_ROOT / "unbundled_raw"
FIGURE_DATA_ALIASES = {
    "tau_bench_all_samples_qwen_cloud_20260626": "fig3-tau_bench_all_samples_qwen_cloud_20260626",
    "tau_bench_all_samples_full_qwen_cloud_20260626": "fig3-tau_bench_all_samples_full_qwen_cloud_20260626",
    "tau_bench_all_samples_call_chain_qwen_20260626_r2": "fig3-tau_bench_all_samples_call_chain_qwen_20260626_r2",
    "codegen_llm_benchmark_all10_final_20260610": "fig4-codegen_llm_benchmark_all10_final_20260610",
    "deployment_benchmark_qwen3_4b_q4_0_20260608": "fig5-deployment_benchmark_qwen3_4b_q4_0_20260608",
    "my_run": "fig5-my_run",
    "llama_cpp_wave_opt_20260623": "fig6-llama_cpp_wave_opt_20260623",
    "framework_characterization_llama_cpp_single_u_seq_p1_10wf_20260623": (
        "fig6_9_12-framework_characterization_llama_cpp_single_u_seq_p1_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623": (
        "fig6_9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p2_10wf_20260623"
    ),
    "framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623": (
        "fig9_10_11_12_15-framework_characterization_llama_cpp_wave_cpu_npu_nogemma_p3_10wf_20260623"
    ),
}


def resolve_repo_relative(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (REPO_ROOT / expanded).resolve()


def candidate_wfgen_results() -> tuple[Path, ...]:
    """Known layouts for the WFGen result artifacts used by these figures."""
    return (
        Path("..") / "WFGen" / "figure_data",
        Path("..") / "figure_data",
        Path("..") / "WFGen" / "wfgen" / "figure_data",
    )


def default_wfgen_results() -> Path:
    return REPO_ROOT / "unbundled_raw" / "figure_data"


def path_for_record(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return Path(os.path.relpath(path, REPO_ROOT)).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class MotivationFigureConfig:
    wfgen_results: Path = field(default_factory=default_wfgen_results)
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "figures" / "motivation")
    planning_run_name: str = "fig3-tau_bench_all_samples_full_qwen_cloud_20260626"
    planning_codegen_run_name: str = "fig4-tau_bench_all_samples_call_chain_qwen_20260626_r2-codegen_qwen_20260627"
    codegen_run_name: str = "fig4-codegen_llm_benchmark_all10_final_20260610"
    topology_run_name: str = "fig3-tau_bench_all_samples_full_qwen_cloud_20260626"
    topology_swimlane_run_name: str = "fig3-tau_bench_all_samples_full_qwen_cloud_20260626"
    topology_swimlane_workflow_id: str = "tau_bench_cloud_retail_0_train_0"
    offloading_eval_run_name: str = "fig6-llama_cpp_wave_opt_20260623"
    colors: dict[str, str] = field(
        default_factory=lambda: dict(figure_style.DATA_COLORS)
    )
    font: dict[str, Any] = field(
        default_factory=lambda: dict(figure_style.FONT)
    )
    style: dict[str, float] = field(
        default_factory=lambda: dict(figure_style.STYLE)
    )
    figure_sizes: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "planning_latency": (2.60, 1.40),
            "planning_breakdown": (2.60, 1.40),
            "codegen_latency": (2.60, 1.40),
            "codegen_tokens": (2.60, 1.40),
            "topology_parallelism": (1.87, 1.43),
            "topology_swimlane": (3.30, 1.43),
            "offloading_latency": (2.05, 1.38),
            "offloading_resource": (1.85, 1.38),
        }
    )
    output_files: tuple[str, ...] = (
        "overview/background.pdf",
        "overview/background.svg",
        "planning/planning_cost_latency.pdf",
        "planning/planning_cost_latency.svg",
        "planning/planning_cost_breakdown.pdf",
        "planning/planning_cost_breakdown.svg",
        "codegen/codegen_latency.pdf",
        "codegen/codegen_latency.svg",
        "codegen/codegen_tokens.pdf",
        "codegen/codegen_tokens.svg",
        "topology/topology_parallelism.pdf",
        "topology/topology_parallelism.svg",
        "topology/topology_swimlane.pdf",
        "topology/topology_swimlane.svg",
        "offloading/offloading_latency.pdf",
        "offloading/offloading_latency.svg",
        "offloading/offloading_resource_usage.pdf",
        "offloading/offloading_resource_usage.svg",
        "provenance/source_classification.json",
    )
    savefig_pad_inches: float = 0.015
    tight_layout_pad: float = 0.15


CONFIG = MotivationFigureConfig()
MOTIVATION_FONT = {
    "base": 8.2,
    "title": 8.4,
    "label": 8.2,
    "tick": 7.5,
    "legend": 6.8,
    "annotation": 7.0,
}
WFGEN_RESULTS = resolve_repo_relative(CONFIG.wfgen_results)
OUT_DIR = CONFIG.output_dir
COLORS = CONFIG.colors
BACKGROUND_COLORS = {
    "ink": figure_style.MOTIVATION_BACKGROUND_COLORS["ink"],
    "gray_dark": figure_style.MOTIVATION_BACKGROUND_COLORS["grayDark"],
    "grid": figure_style.MOTIVATION_BACKGROUND_COLORS["grid"],
    "blue": figure_style.MOTIVATION_BACKGROUND_COLORS["blue"],
    "teal": figure_style.MOTIVATION_BACKGROUND_COLORS["teal"],
    "teal_light": figure_style.MOTIVATION_BACKGROUND_COLORS["tealLight"],
    "blue_light": figure_style.MOTIVATION_BACKGROUND_COLORS["blueLight"],
    "light_gray": figure_style.MOTIVATION_BACKGROUND_COLORS["lightGray"],
    "memory": figure_style.MOTIVATION_BACKGROUND_COLORS["memory"],
    "gray": figure_style.MOTIVATION_BACKGROUND_COLORS["gray"],
    "orange": figure_style.MOTIVATION_BACKGROUND_COLORS["orange"],
    "rose": figure_style.MOTIVATION_BACKGROUND_COLORS["rose"],
    "white": figure_style.MOTIVATION_BACKGROUND_COLORS["white"],
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_planning_request_attempts(
    run_dir: Path, samples: list[dict[str, Any]]
) -> dict[str, Any]:
    log_path = run_dir / "logs" / "planning_run.log"
    if not log_path.exists():
        return {
            "available": False,
            "log_path": path_for_record(log_path),
            "reason": "missing planning_run.log",
        }

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    send_pattern = re.compile(
        r"sending workflow-generation request "
        r"\[sample ([0-9]+)/280 selection_index=([0-9]+)\]"
    )
    response_pattern = re.compile(
        r"response received "
        r"\[sample ([0-9]+)/280 selection_index=([0-9]+)\]"
    )
    sends = [
        (int(match.group(1)), int(match.group(2)))
        for match in send_pattern.finditer(log_text)
    ]
    responses = [
        (int(match.group(1)), int(match.group(2)))
        for match in response_pattern.finditer(log_text)
    ]
    send_counts = Counter(sends)
    response_counts = Counter(responses)
    samples_by_selection = {sample["selection_index"]: sample for sample in samples}
    samples_by_number = {sample["sample_index"] + 1: sample for sample in samples}

    extra_attempts: list[dict[str, Any]] = []
    for sample_key in sorted(set(send_counts) | set(response_counts)):
        extra_count = send_counts[sample_key] - response_counts[sample_key]
        if extra_count <= 0:
            continue
        sample_number, selection_index = sample_key
        sample = samples_by_selection.get(selection_index) or samples_by_number.get(sample_number)
        if sample is None:
            extra_attempts.append(
                {
                    "sample_number": sample_number,
                    "selection_index": selection_index,
                    "extra_sent_request_count": extra_count,
                    "estimated_prompt_tokens": None,
                    "estimated_completion_tokens": None,
                    "estimated_total_tokens": None,
                    "estimated_wall_ms": None,
                }
            )
            continue
        extra_attempts.append(
            {
                "sample_number": sample_number,
                "sample_index": sample["sample_index"],
                "selection_index": selection_index,
                "extra_sent_request_count": extra_count,
                "estimated_prompt_tokens": sample["prompt_tokens"] * extra_count,
                "estimated_completion_tokens": sample["completion_tokens"] * extra_count,
                "estimated_total_tokens": sample["total_tokens"] * extra_count,
                "estimated_wall_ms": sample["qwen_wall_ms"] * extra_count,
                "estimation_basis": "same-sample successful response usage",
            }
        )

    successful_totals = {
        "prompt_tokens": int(sum(sample["prompt_tokens"] for sample in samples)),
        "completion_tokens": int(sum(sample["completion_tokens"] for sample in samples)),
        "total_tokens": int(sum(sample["total_tokens"] for sample in samples)),
        "wall_ms": float(sum(sample["qwen_wall_ms"] for sample in samples)),
    }
    estimated_extra = {
        "prompt_tokens": int(
            sum(item["estimated_prompt_tokens"] or 0 for item in extra_attempts)
        ),
        "completion_tokens": int(
            sum(item["estimated_completion_tokens"] or 0 for item in extra_attempts)
        ),
        "total_tokens": int(
            sum(item["estimated_total_tokens"] or 0 for item in extra_attempts)
        ),
        "wall_ms": float(
            sum(item["estimated_wall_ms"] or 0.0 for item in extra_attempts)
        ),
    }
    prompt_only_lower_bound = {
        "prompt_tokens": successful_totals["prompt_tokens"] + estimated_extra["prompt_tokens"],
        "completion_tokens": successful_totals["completion_tokens"],
        "total_tokens": successful_totals["total_tokens"] + estimated_extra["prompt_tokens"],
        "wall_ms": successful_totals["wall_ms"],
    }
    same_sample_proxy_totals = {
        "prompt_tokens": successful_totals["prompt_tokens"] + estimated_extra["prompt_tokens"],
        "completion_tokens": successful_totals["completion_tokens"] + estimated_extra["completion_tokens"],
        "total_tokens": successful_totals["total_tokens"] + estimated_extra["total_tokens"],
        "wall_ms": successful_totals["wall_ms"] + estimated_extra["wall_ms"],
        "avg_total_tokens_per_workflow": (
            successful_totals["total_tokens"] + estimated_extra["total_tokens"]
        )
        / len(samples),
        "avg_wall_s_per_workflow": (
            successful_totals["wall_ms"] + estimated_extra["wall_ms"]
        )
        / len(samples)
        / 1000.0,
    }

    return {
        "available": True,
        "log_path": path_for_record(log_path),
        "sent_request_count": len(sends),
        "usage_bearing_response_count": len(responses),
        "unlogged_sent_request_count": int(sum(item["extra_sent_request_count"] for item in extra_attempts)),
        "extra_sent_requests_without_local_usage": extra_attempts,
        "successful_response_totals": successful_totals,
        "prompt_only_lower_bound_totals": prompt_only_lower_bound,
        "same_sample_response_proxy_totals": same_sample_proxy_totals,
        "note": (
            "Extra sent planning requests do not have local usage records because "
            "the run was interrupted or restarted before a response was logged. "
            "Exact charged usage requires provider-side billing logs."
        ),
    }


def resolve_results_path(path_value: str | Path) -> Path:
    path_text = str(path_value)
    path = Path(path_text)
    if path.exists():
        return path

    if "\\" in path_text or ":" in path_text:
        parts = PureWindowsPath(path_text).parts
    else:
        parts = path.parts

    for marker in ("results", "figure_data"):
        if marker not in parts:
            continue
        result_idx = len(parts) - 1 - parts[::-1].index(marker)
        rel_parts = parts[result_idx + 1 :]
        if rel_parts:
            alias = FIGURE_DATA_ALIASES.get(rel_parts[0], rel_parts[0])
            candidate = WFGEN_RESULTS.joinpath(alias, *rel_parts[1:])
            if candidate.exists():
                return candidate

    return path


def savefig(fig: plt.Figure, name: str, exact_canvas: bool = False) -> None:
    figure_style.save_pdf_svg(
        fig,
        OUT_DIR / name,
        exact_canvas=exact_canvas,
        tight_layout_pad=CONFIG.tight_layout_pad,
        pad_inches=CONFIG.savefig_pad_inches,
        creator="Paper_MOBench motivation figure generator",
    )
    plt.close(fig)
    strip_trailing_whitespace((OUT_DIR / name).with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def configure_style(config: MotivationFigureConfig = CONFIG) -> None:
    figure_style.configure_matplotlib(plt)
    config.font.update(MOTIVATION_FONT)
    plt.rcParams.update(
        {
            "font.size": MOTIVATION_FONT["base"],
            "axes.titlesize": MOTIVATION_FONT["title"],
            "axes.labelsize": MOTIVATION_FONT["label"],
            "xtick.labelsize": MOTIVATION_FONT["tick"],
            "ytick.labelsize": MOTIVATION_FONT["tick"],
            "legend.fontsize": MOTIVATION_FONT["legend"],
        }
    )


def get_planning_run() -> dict[str, Any]:
    frozen = FROZEN_INPUT_DIR / "motivation_planning.json"
    if frozen.exists():
        data = read_json(frozen)
        for key in ['run_dir', 'usage_csv', 'prompt_breakdown_path']:
            data[key] = Path(data[key])
        return data
    run_dir = WFGEN_RESULTS / CONFIG.planning_run_name
    usage_csv = run_dir / "breakdown" / "per_sample_usage.csv"
    prompt_breakdown_path = run_dir / "breakdown" / "planning_prompt_breakdown_summary.json"
    final_breakdown = read_json(run_dir / "breakdown" / "final_breakdown.json")
    manifest = read_json(run_dir / "planning" / "manifest.json")
    usage_rows = sorted(read_csv_rows(usage_csv), key=lambda row: int(row["sample_index"]))
    if final_breakdown.get("sample_count") != 280 or len(usage_rows) != 280:
        raise ValueError("Expected 280 planning samples")
    if any(row.get("status") != "ok" for row in usage_rows):
        raise ValueError("Planning run contains non-ok rows")

    samples: list[dict[str, Any]] = []
    for row in usage_rows:
        samples.append(
            {
                "sample_index": int(row["sample_index"]),
                "selection_index": int(row["selection_index"]),
                "domain": row["domain"],
                "category": row["category"],
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "total_tokens": int(row["total_tokens"]),
                "qwen_wall_ms": float(row["wall_ms"]),
                "node_count": int(row.get("node_count") or 0),
                "prompt_chars": int(row.get("prompt_chars") or 0),
                "raw_response": row.get("raw_response", ""),
            }
        )

    return {
        "run_dir": run_dir,
        "usage_csv": usage_csv,
        "final_breakdown": final_breakdown,
        "manifest": manifest,
        "prompt_breakdown_path": prompt_breakdown_path,
        "prompt_breakdown_summary": read_json(prompt_breakdown_path),
        "samples": samples,
        "sample_count": len(samples),
        "request_attempt_audit": audit_planning_request_attempts(run_dir, samples),
    }


def get_planning_codegen_run() -> dict[str, Any]:
    frozen = FROZEN_INPUT_DIR / "motivation_codegen.json"
    if frozen.exists():
        data = read_json(frozen)
        for key in ['run_dir', 'breakdown_path', 'per_sample_path', 'benchmark_path']:
            data[key] = Path(data[key])
        return data
    run_dir = WFGEN_RESULTS / CONFIG.planning_codegen_run_name
    breakdown_path = run_dir / "breakdown" / "codegen_breakdown.json"
    per_sample_path = run_dir / "breakdown" / "per_sample_codegen.csv"
    benchmark_path = run_dir / "codegen_llm_benchmark.json"
    first_pass_path = run_dir / "codegen_llm_benchmark_first_pass.json"
    breakdown = read_json(breakdown_path)
    benchmark = read_json(benchmark_path)
    first_pass = read_json(first_pass_path)
    per_sample_rows = sorted(read_csv_rows(per_sample_path), key=lambda row: int(row["prompt_index"]))
    final_rows = sorted(benchmark.get("results", []), key=lambda row: int(row["prompt_index"]))
    if len(per_sample_rows) != 280 or len(final_rows) != 280 or breakdown.get("sample_count") != 280:
        raise ValueError("Expected 280 code-generation samples")
    if benchmark.get("status") != "ok" or any(row.get("status") != "ok" for row in final_rows):
        raise ValueError("Code-generation final artifacts contain non-ok rows")
    per_sample_by_index = {int(row["prompt_index"]): row for row in per_sample_rows}

    samples: list[dict[str, Any]] = []
    for row in final_rows:
        prompt_index = int(row["prompt_index"])
        attempt_row = per_sample_by_index[prompt_index]
        samples.append(
            {
                "prompt_index": prompt_index,
                "selection_index": int(attempt_row["selection_index"]),
                "domain": attempt_row["domain"],
                "first_status": attempt_row["first_status"],
                "final_status": row["status"],
                "compile_attempts": int(attempt_row["compile_attempts"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "total_tokens": int(row["total_tokens"]),
                "wall_ms": float(row["wall_ms"]),
                "first_prompt_tokens": int(attempt_row["first_prompt_tokens"]),
                "first_completion_tokens": int(attempt_row["first_completion_tokens"]),
                "first_total_tokens": int(attempt_row["first_total_tokens"]),
                "total_prompt_tokens_including_retries": int(attempt_row["total_prompt_tokens_including_retries"]),
                "total_completion_tokens_including_retries": int(attempt_row["total_completion_tokens_including_retries"]),
                "total_tokens_including_retries": int(attempt_row["total_tokens_including_retries"]),
                "first_wall_ms": float(attempt_row["first_wall_ms"]),
                "total_wall_ms_including_retries": float(attempt_row["total_wall_ms_including_retries"]),
            }
        )

    return {
        "run_dir": run_dir,
        "breakdown": breakdown,
        "benchmark": benchmark,
        "first_pass": first_pass,
        "samples": samples,
        "sample_count": len(samples),
        "breakdown_path": breakdown_path,
        "per_sample_path": per_sample_path,
        "benchmark_path": benchmark_path,
    }


def box_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    figure_style.box_axes(ax, grid_axis=grid_axis)


def add_panel_note(ax: plt.Axes, text: str, x: float, y: float) -> None:
    figure_style.add_panel_note(ax, text, x, y)


def draw_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    facecolor: str,
    edgecolor: str | None = None,
    textcolor: str | None = None,
    fontsize: float | None = None,
    weight: str = "normal",
) -> FancyBboxPatch:
    edgecolor = edgecolor or COLORS["white"]
    textcolor = textcolor or COLORS["dark"]
    fontsize = fontsize or CONFIG.font["annotation"]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.55,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=textcolor,
        linespacing=1.1,
        zorder=3,
    )
    return patch


def draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["blue"],
    linewidth: float = 1.0,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=linewidth,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def plot_overview_background() -> dict[str, Any]:
    colors = BACKGROUND_COLORS
    fig, ax = plt.subplots(figsize=(3.55, 2.42))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.98)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    layer_specs = [
        ("Application", 0.83, 0.14),
        ("Framework", 0.63, 0.14),
        ("Runtime", 0.43, 0.14),
        ("Hardware", 0.12, 0.20),
    ]
    for layer, y, height in layer_specs:
        ax.text(
            0.035,
            y + height / 2,
            layer,
            ha="left",
            va="center",
            fontsize=CONFIG.font["annotation"],
            fontweight="bold",
            color=colors["gray_dark"],
        )
        ax.plot(
            [0.18, 0.96],
            [y - 0.018, y - 0.018],
            color=colors["grid"],
            linewidth=0.5,
            zorder=1,
        )

    app_x = [0.22, 0.36, 0.50, 0.64, 0.78]
    app_labels = ["LLM", "APP", "LLM", "APP", "LLM"]
    app_colors = [colors["blue"], colors["teal"], colors["blue"], colors["teal"], colors["blue"]]
    for x, label, color in zip(app_x, app_labels, app_colors):
        draw_round_box(
            ax,
            x,
            0.86,
            0.095,
            0.085,
            label,
            color,
            textcolor=colors["white"],
            fontsize=CONFIG.font["annotation"],
            weight="bold",
        )
    for left, right in zip(app_x[:-1], app_x[1:]):
        draw_arrow(ax, (left + 0.095, 0.902), (right, 0.902), colors["gray_dark"], 0.75)

    framework = [
        (0.22, "qnn-net-run", colors["teal_light"]),
        (0.39, "LiteRT", colors["teal_light"]),
        (0.56, "Llama.cpp\nllm.npu", colors["blue_light"]),
        (0.75, "Genie", colors["blue_light"]),
    ]
    for x, label, color in framework:
        draw_round_box(ax, x, 0.66, 0.14, 0.09, label, color, edgecolor=colors["white"])

    runtime = [
        (0.22, "NNAPI", colors["light_gray"]),
        (0.39, "QNN", colors["blue_light"]),
        (0.56, "NeuroPilot", colors["light_gray"]),
        (0.75, "CoreML", colors["light_gray"]),
    ]
    for x, label, color in runtime:
        draw_round_box(ax, x, 0.46, 0.14, 0.09, label, color, edgecolor=colors["white"])

    draw_round_box(
        ax,
        0.24,
        0.26,
        0.60,
        0.075,
        "Unified memory (LPDDR5x) / shared address space",
        colors["memory"],
        edgecolor=colors["light_gray"],
        fontsize=CONFIG.font["annotation"] - 0.4,
    )
    hardware = [
        (0.24, "CPU", colors["gray"]),
        (0.45, "GPU", colors["orange"]),
        (0.66, "NPU", colors["blue"]),
    ]
    for x, label, color in hardware:
        draw_round_box(
            ax,
            x,
            0.14,
            0.16,
            0.095,
            label,
            color,
            textcolor=colors["white"],
            fontsize=CONFIG.font["annotation"] + 0.2,
            weight="bold",
        )

    path_color = colors["rose"]
    draw_arrow(ax, (0.827, 0.86), (0.82, 0.75), path_color, 1.1)
    draw_arrow(ax, (0.75, 0.66), (0.46, 0.55), path_color, 1.1, rad=0.08)
    draw_arrow(ax, (0.46, 0.46), (0.74, 0.235), path_color, 1.1, rad=-0.10)
    ax.text(
        0.69,
        0.405,
        "task execution path\n(Qualcomm SoC example)",
        ha="left",
        va="center",
        fontsize=CONFIG.font["annotation"] - 0.2,
        color=path_color,
        linespacing=1.15,
    )

    savefig(fig, "overview/background", exact_canvas=True)
    return {
        "type": "schematic",
        "layers": ["Application", "Framework", "Runtime", "Hardware"],
        "highlighted_path": "LLM -> Genie -> QNN -> NPU",
    }


def plot_planning_cost() -> dict[str, Any]:
    planning = get_planning_run()

    plot_planning_latency(planning)
    token_breakdown = plot_planning_breakdown(planning)
    samples = planning["samples"]
    latency_s = np.array([sample["qwen_wall_ms"] / 1000.0 for sample in samples])
    prompt_tokens = np.array([sample["prompt_tokens"] for sample in samples])
    completion_tokens = np.array([sample["completion_tokens"] for sample in samples])
    total_tokens = np.array([sample["total_tokens"] for sample in samples])

    totals = {
        "sample_count": planning["sample_count"],
        "workflow_count": planning["final_breakdown"]["workflow_count"],
        "prompt_builder": planning["manifest"]["prompt_builder"],
        "status_counts": planning["final_breakdown"]["status_counts"],
        "domain_counts": planning["final_breakdown"]["domain_counts"],
        "avg_prompt_tokens": float(np.mean(prompt_tokens)),
        "avg_completion_tokens": float(np.mean(completion_tokens)),
        "avg_total_tokens": float(np.mean(total_tokens)),
        "total_prompt_tokens": int(np.sum(prompt_tokens)),
        "total_completion_tokens": int(np.sum(completion_tokens)),
        "total_tokens": int(np.sum(total_tokens)),
        "avg_latency_s": float(np.mean(latency_s)),
        "median_latency_s": float(np.median(latency_s)),
        "total_latency_s": float(np.sum(latency_s)),
        "min_latency_s": float(np.min(latency_s)),
        "max_latency_s": float(np.max(latency_s)),
        "latency_breakdown_available": False,
        "request_attempt_audit": planning["request_attempt_audit"],
        "planning_token_breakdown_avg": token_breakdown,
    }
    return totals


def plot_planning_latency(planning: dict[str, Any]) -> None:
    planning_latency_s = np.array([s["qwen_wall_ms"] / 1000.0 for s in planning["samples"]])
    x_max = np.ceil(float(np.max(planning_latency_s)) / 50.0) * 50.0

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["planning_latency"])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.78)

    series = [
        ("Planning", planning_latency_s, COLORS["cdf_planning"]),
    ]
    for label, values, color in series:
        latency_sorted = np.sort(values)
        cdf = np.arange(1, len(latency_sorted) + 1) / len(latency_sorted)
        step_x = np.r_[latency_sorted[0], latency_sorted]
        step_y = np.r_[0.0, cdf]
        ax.step(
            step_x,
            step_y,
            where="post",
            color=color,
            linewidth=CONFIG.style["line_width"],
            label=f"{label} ({np.mean(values):.1f}s)",
            zorder=3,
        )
        ax.axvline(
            float(np.mean(values)),
            color=color,
            linestyle=(0, (2, 2)),
            linewidth=0.65,
            alpha=0.75,
            zorder=2,
        )

    ax.set_xlabel("LLM-call latency (s)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, x_max + 1, 100))
    ax.set_yticks([0, 0.5, 1.0])
    box_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.18,
    )
    savefig(fig, "planning/planning_cost_latency", exact_canvas=True)


def plot_planning_breakdown(planning: dict[str, Any]) -> dict[str, float]:
    summary = planning["prompt_breakdown_summary"]
    averages = summary["averages"]
    groups = [
        (
            "Tools",
            averages["tool_catalog_prompt_tokens_est"],
            COLORS["stack_tools"],
        ),
        (
            "Schema",
            averages["workflow_schema_in_user_prompt_prompt_tokens_est"],
            COLORS["stack_schema"],
        ),
        (
            "Sample",
            (
                averages["sample_payload_prompt_tokens_est"]
                + averages["user_request_summary_prompt_tokens_est"]
                + averages["task_header_prompt_tokens_est"]
            ),
            COLORS["stack_sample"],
        ),
        (
            "Rules",
            averages["generation_rules_prompt_tokens_est"] + averages["fixed_json_framing_prompt_tokens_est"],
            COLORS["stack_rules"],
        ),
        (
            "Output",
            planning["final_breakdown"]["usage"]["avg_completion_tokens"],
            COLORS["stack_output"],
        ),
    ]
    groups = sorted(groups, key=lambda item: item[1], reverse=True)
    total_tokens = sum(value for _, value, _ in groups)
    avg_breakdown = {
        "source": path_for_record(planning["prompt_breakdown_path"]),
        "avg_user_prompt_chars": float(averages["user_prompt_chars"]),
        "avg_prompt_tokens": float(averages["prompt_tokens"]),
        "avg_completion_tokens": float(planning["final_breakdown"]["usage"]["avg_completion_tokens"]),
        "avg_total_tokens": float(planning["final_breakdown"]["usage"]["avg_total_tokens"]),
        "avg_system_prompt_chars": float(averages["system_prompt_chars"]),
        "avg_response_format_schema_chars": float(averages["response_format_schema_file_chars"]),
        "note": "Prompt component token values are estimates from character shares; output tokens are service usage.",
    }
    for label, value, _ in groups:
        key = label.lower().replace("/", "_")
        avg_breakdown[f"{key}_tokens"] = float(value)
        avg_breakdown[f"{key}_share"] = float(value / total_tokens)

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["planning_breakdown"])
    fig.subplots_adjust(left=0.21, right=0.98, bottom=0.27, top=0.70)
    x = np.array([0])
    bottom = np.zeros_like(x, dtype=float)
    for label, value, color in groups:
        value_k = value / 1000.0
        ax.bar(
            x,
            [value_k],
            width=0.54,
            bottom=bottom,
            color=color,
            edgecolor=COLORS["white"],
            linewidth=CONFIG.style["bar_edge_width"],
            label=label,
            zorder=3,
        )
        if value / total_tokens > 0.18:
            ax.text(
                x[0],
                bottom[0] + value_k / 2.0,
                f"{value / total_tokens * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=CONFIG.font["annotation"] - 0.2,
                color=COLORS["white"],
                fontweight="bold",
                zorder=4,
            )
        bottom += value_k

    ax.set_xlabel("280 workflows")
    ax.set_ylabel("Avg tokens /\nworkflow (K)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Planning"])
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(0, np.ceil((total_tokens / 1000.0) / 2.5) * 2.5 + 0.5)
    ax.set_yticks([0, 5, 10])
    box_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=3,
        columnspacing=0.55,
        handlelength=0.9,
        handletextpad=0.25,
        labelspacing=0.16,
    )

    savefig(fig, "planning/planning_cost_breakdown", exact_canvas=True)
    return avg_breakdown


def get_codegen_benchmark() -> dict[str, Any]:
    benchmark_path = (
        WFGEN_RESULTS
        / CONFIG.codegen_run_name
        / "codegen_llm_benchmark.json"
    )
    data = read_json(benchmark_path)
    rows = sorted(data.get("results", []), key=lambda item: int(item.get("prompt_index", 0)))
    if data.get("status") != "ok":
        raise ValueError("Code-generation benchmark is not ok")
    if len(rows) != 10:
        raise ValueError("Expected 10 code-generation benchmark rows")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError("Code-generation benchmark contains non-ok rows")
    data["results"] = rows
    return data


def plot_codegen_cost() -> dict[str, Any]:
    codegen = get_planning_codegen_run()
    rows = codegen["samples"]
    plot_codegen_latency(rows)
    plot_codegen_tokens(rows)

    latency_s = np.array([float(row["total_wall_ms_including_retries"]) / 1000.0 for row in rows])
    prompt_tokens = np.array([int(row["total_prompt_tokens_including_retries"]) for row in rows])
    completion_tokens = np.array([int(row["total_completion_tokens_including_retries"]) for row in rows])
    total_tokens = np.array([int(row["total_tokens_including_retries"]) for row in rows])
    return {
        "workflow_count": len(rows),
        "model": codegen["breakdown"].get("model", ""),
        "source": path_for_record(codegen["run_dir"]),
        "cost_accounting": "includes failed first-pass attempts and retry requests",
        "first_pass_compile_success_rate": codegen["breakdown"]["first_pass"]["compile_success_rate"],
        "final_compile_success_rate": codegen["breakdown"]["final_compiled_artifacts"]["compile_success_rate"],
        "retry_request_count": codegen["breakdown"]["retry"]["request_count"],
        "retry_token_overhead": codegen["breakdown"]["retry"]["total_tokens"],
        "retry_latency_s_overhead": codegen["breakdown"]["retry"]["wall_ms_sum"] / 1000.0,
        "avg_latency_s": float(np.mean(latency_s)),
        "total_latency_s": float(np.sum(latency_s)),
        "min_latency_s": float(np.min(latency_s)),
        "max_latency_s": float(np.max(latency_s)),
        "avg_prompt_tokens": float(np.mean(prompt_tokens)),
        "avg_completion_tokens": float(np.mean(completion_tokens)),
        "avg_total_tokens": float(np.mean(total_tokens)),
        "total_prompt_tokens": int(np.sum(prompt_tokens)),
        "total_completion_tokens": int(np.sum(completion_tokens)),
        "total_tokens": int(np.sum(total_tokens)),
        "compile_status": "syntax-compiled, not executed",
    }


def plot_codegen_latency(rows: list[dict[str, Any]]) -> None:
    latency_s = np.array([float(row["total_wall_ms_including_retries"]) / 1000.0 for row in rows])
    avg_latency = float(np.mean(latency_s))
    x_max = np.ceil(float(np.max(latency_s)) / 50.0) * 50.0

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["codegen_latency"])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.78)
    latency_sorted = np.sort(latency_s)
    cdf = np.arange(1, len(latency_sorted) + 1) / len(latency_sorted)
    step_x = np.r_[latency_sorted[0], latency_sorted]
    step_y = np.r_[0.0, cdf]
    ax.step(
        step_x,
        step_y,
        where="post",
        color=COLORS["cdf_codegen"],
        linewidth=CONFIG.style["line_width"],
        label=f"Codegen+retry ({avg_latency:.1f}s)",
        zorder=3,
    )
    ax.axvline(
        avg_latency,
        color=COLORS["cdf_codegen"],
        linestyle=(0, (2, 2)),
        linewidth=0.65,
        alpha=0.75,
        zorder=2,
    )
    ax.set_xlabel("LLM-call latency (s)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, x_max + 1, 100))
    ax.set_yticks([0, 0.5, 1.0])
    box_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=1,
        handlelength=1.2,
        handletextpad=0.35,
        labelspacing=0.18,
    )

    savefig(fig, "codegen/codegen_latency", exact_canvas=True)


def plot_codegen_tokens(rows: list[dict[str, Any]]) -> None:
    prompt_tokens = np.array(
        [int(row["total_prompt_tokens_including_retries"]) for row in rows],
        dtype=float,
    ) / 1000.0
    completion_tokens = np.array(
        [int(row["total_completion_tokens_including_retries"]) for row in rows],
        dtype=float,
    ) / 1000.0
    total_tokens = prompt_tokens + completion_tokens
    total_avg = float(np.mean(total_tokens))
    groups = [
        ("Prompt", prompt_tokens, COLORS["stack_sample"]),
        ("Code", completion_tokens, COLORS["stack_rules"]),
        (f"Total ({total_avg:.2f}K)", total_tokens, COLORS["cdf_total"]),
    ]

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["codegen_tokens"])
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.27, top=0.78)
    for label, values, color in groups:
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        step_x = np.r_[sorted_values[0], sorted_values]
        step_y = np.r_[0.0, cdf]
        ax.step(
            step_x,
            step_y,
            where="post",
            color=color,
            linewidth=CONFIG.style["line_width"],
            label=label,
            zorder=4,
        )

    ax.axvline(
        total_avg,
        color=COLORS["cdf_total"],
        linestyle=(0, (2, 2)),
        linewidth=0.65,
        alpha=0.75,
        zorder=2,
    )
    x_max = np.ceil(float(np.max(total_tokens)) / 5.0) * 5.0
    ax.set_xlabel("Tokens / workflow (K)")
    ax.set_ylabel("CDF")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.04)
    ax.set_xticks(np.arange(0, x_max + 0.1, 10))
    ax.set_yticks([0, 0.5, 1.0])
    box_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=3,
        columnspacing=0.35,
        handlelength=0.72,
        handletextpad=0.20,
        labelspacing=0.14,
        prop={"size": CONFIG.font["legend"] - 0.25},
    )

    savefig(fig, "codegen/codegen_tokens", exact_canvas=True)


def get_workflow_topology() -> list[dict[str, Any]]:
    frozen = FROZEN_INPUT_DIR / "motivation_topology.json"
    if frozen.exists():
        return read_json(frozen)["rows"]
    data = read_json(WFGEN_RESULTS / CONFIG.topology_run_name / "planning" / "workflow.json")
    workflows = data.get("workflows") or data.get("workflow_document", {}).get("workflows") or []
    rows = []
    for idx, workflow in enumerate(workflows, start=1):
        waves = workflow.get("execution_hints", {}).get("parallel_waves", [])
        wave_widths = [len(w.get("node_ids", [])) for w in waves]
        nodes = len(workflow.get("nodes", []))
        wave_count = len(waves)
        reduction = (nodes - wave_count) / nodes if nodes else 0.0
        rows.append(
            {
                "task": idx,
                "workflow_id": workflow["workflow_id"],
                "nodes": nodes,
                "waves": wave_count,
                "max_width": max(wave_widths) if wave_widths else 0,
                "reduction": reduction,
                "reduction_pct": reduction * 100.0,
            }
        )
    if not rows:
        raise ValueError("Expected at least one workflow topology row")
    return rows


def plot_topology_parallelism() -> dict[str, Any]:
    rows = get_workflow_topology()
    nodes = np.array([r["nodes"] for r in rows])
    waves = np.array([r["waves"] for r in rows])
    max_widths = np.array([r["max_width"] for r in rows])
    reductions_pct = np.array([r["reduction_pct"] for r in rows])
    sorted_reductions = np.sort(reductions_pct)
    cdf = np.arange(1, len(sorted_reductions) + 1) / len(sorted_reductions) * 100.0
    mean_reduction = float(np.mean(reductions_pct))
    median_reduction = float(np.median(reductions_pct))
    weighted_reduction = float((np.sum(nodes) - np.sum(waves)) / np.sum(nodes) * 100.0)

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["topology_parallelism"])
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.24, top=0.92)
    ax.step(
        sorted_reductions,
        cdf,
        where="post",
        color=COLORS["blue"],
        linewidth=CONFIG.style["line_width"],
        label="Workflow CDF",
        zorder=3,
    )
    ax.axvline(
        mean_reduction,
        color=COLORS["gray_dark"],
        linestyle="--",
        linewidth=CONFIG.style["line_width"],
        zorder=4,
    )
    ax.text(
        mean_reduction + 1.2,
        8,
        f"mean\n{mean_reduction:.1f}%",
        ha="left",
        va="bottom",
        fontsize=CONFIG.font["annotation"] - 0.6,
        color=COLORS["gray_dark"],
    )
    ax.set_xlabel("Reduction (%)")
    ax.set_ylabel("CDF (%)")
    ax.set_xlim(-1, 76)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_yticks([0, 50, 100])
    box_axes(ax)
    savefig(fig, "topology/topology_parallelism", exact_canvas=True)
    swimlane = plot_topology_swimlane()

    return {
        "avg_nodes": float(np.mean(nodes)),
        "avg_waves": float(np.mean(waves)),
        "avg_node_to_wave_ratio": float(np.mean(nodes / waves)),
        "avg_reduction_pct": mean_reduction,
        "median_reduction_pct": median_reduction,
        "weighted_total_reduction_pct": weighted_reduction,
        "parallel_workflows": int(np.sum(max_widths > 1)),
        "workflow_count": len(rows),
        "swimlane_example": swimlane,
    }


def get_swimlane_workflow() -> dict[str, Any]:
    frozen = FROZEN_INPUT_DIR / "motivation_topology.json"
    if frozen.exists():
        return read_json(frozen)["swimlane"]
    data = read_json(WFGEN_RESULTS / CONFIG.topology_swimlane_run_name / "planning" / "workflow.json")
    workflows = data.get("workflows", [])
    for workflow in workflows:
        if workflow.get("workflow_id") == CONFIG.topology_swimlane_workflow_id:
            return workflow
    for workflow in workflows:
        waves = workflow.get("execution_hints", {}).get("parallel_waves", [])
        max_width = max((len(wave.get("node_ids", [])) for wave in waves), default=0)
        if max_width > 1 and len(workflow.get("nodes", [])) <= 6:
            return workflow
    raise ValueError("Could not find a workflow for the topology swimlane example")


def plot_topology_swimlane() -> dict[str, Any]:
    workflow = get_swimlane_workflow()
    nodes = {node["node_id"]: node for node in workflow.get("nodes", [])}
    waves = workflow.get("execution_hints", {}).get("parallel_waves", [])
    if not waves:
        raise ValueError("Selected workflow has no parallel_waves")
    sequence = [node_id for wave in waves for node_id in wave.get("node_ids", [])]
    wave_ids = [wave.get("node_ids", []) for wave in waves]

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["topology_swimlane"])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    width, height = np.array(CONFIG.figure_sizes["topology_swimlane"]) * 72
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    # Use points so text, strokes, and spacing stay legible at paper size.
    margin = 4.0
    node_numbers = {node_id: index + 1 for index, node_id in enumerate(sequence)}
    tool_style = ("#EDF4FA", COLORS["blue"])
    llm_style = ("#EFF6F0", "#52765A")

    def node_style(node_id: str) -> tuple[str, str]:
        return llm_style if nodes[node_id].get("kind") == "llm" else tool_style

    def node_box(x: float, y: float, box_width: float, box_height: float, node_id: str, *, token: bool = False) -> None:
        fill, outline = node_style(node_id)
        ax.add_patch(FancyBboxPatch(
            (x, y), box_width, box_height,
            boxstyle="round,pad=0,rounding_size=1.8",
            linewidth=0.65, edgecolor=outline, facecolor=fill, zorder=3,
        ))
        number = node_numbers[node_id]
        if token:
            label = str(number)
        elif node_id.startswith("check_after_"):
            upstream = node_id.removeprefix("check_after_")
            upstream_label = swimlane_label(upstream, nodes)
            subject = {"QueryUser": "user", "QueryOrder": "order"}.get(upstream_label, "result")
            label = f"{number}. Check\n{subject}"
        else:
            label = f"{number}. {swimlane_display_label(swimlane_label(node_id, nodes))}"
            lines = label.splitlines()
            label = "\n".join([lines[0], *[line.lower() for line in lines[1:]]])
        ax.text(x + box_width / 2, y + box_height / 2, label,
                ha="center", va="center", fontsize=8.0 if token else 7.2,
                fontweight="bold" if token else "normal", color=COLORS["ink"],
                linespacing=1.0, zorder=4)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=4.1,
            linewidth=0.7, color=COLORS["gray_dark"],
            shrinkA=0.6, shrinkB=0.6, zorder=2,
        ))

    # A compact color key also applies to the numbered sequential tokens.
    for x, label, style in [(width / 2 - 34, "Tool", tool_style),
                            (width / 2 + 7, "LLM", llm_style)]:
        ax.add_patch(FancyBboxPatch(
            (x, height - 8), 7, 4.5,
            boxstyle="round,pad=0,rounding_size=0.6",
            linewidth=0.6, facecolor=style[0], edgecolor=style[1],
        ))
        ax.text(x + 10, height - 5.8, label, ha="left", va="center",
                fontsize=7.0, color=COLORS["ink"])

    ax.text(margin, 86, "Sequential", ha="left", va="center",
            fontsize=8.2, fontweight="bold", color=COLORS["ink"])
    ax.text(width - margin, 86, f"{len(sequence)} steps", ha="right", va="center",
            fontsize=7.0, color=COLORS["gray_dark"])
    token_width = 14.0
    token_x = np.linspace(margin, width - margin - token_width, len(sequence))
    for index, (node_id, x) in enumerate(zip(sequence, token_x)):
        node_box(float(x), 65, token_width, 14, node_id, token=True)
        if index:
            arrow((float(token_x[index - 1]) + token_width, 72), (float(x), 72))

    ax.plot([margin, width - margin], [59, 59], color=COLORS["light_gray"], linewidth=0.6)
    ax.text(margin, 52, "Dependency-safe parallel", ha="left", va="center",
            fontsize=8.0, fontweight="bold", color=COLORS["ink"])
    ax.text(width - margin, 52, f"{len(waves)} waves", ha="right", va="center",
            fontsize=7.0, color=COLORS["gray_dark"])

    wave_w = 47.0
    wave_x = np.linspace(margin, width - margin - wave_w, len(wave_ids))
    bottom_centers: dict[str, tuple[float, float]] = {}
    for wave_index, node_ids in enumerate(wave_ids):
        x = float(wave_x[wave_index])
        ax.text(x + wave_w / 2, 43, f"Wave {wave_index + 1}", ha="center", va="center",
                fontsize=6.3, color=COLORS["gray_dark"])
        centers = [20.5] if len(node_ids) == 1 else np.linspace(30.5, 10.5, len(node_ids))
        for node_id, center_y in zip(node_ids, centers):
            node_box(x, float(center_y) - 8.5, wave_w, 17, node_id)
            bottom_centers[node_id] = (x + wave_w / 2, float(center_y))

    for source, target in workflow_edges_for_swimlane(workflow, set(sequence)):
        sx, sy = bottom_centers[source]
        tx, ty = bottom_centers[target]
        arrow((sx + wave_w / 2, sy), (tx - wave_w / 2, ty))

    savefig(fig, "topology/topology_swimlane", exact_canvas=True)

    return {
        "path": f"{CONFIG.topology_swimlane_run_name}/planning/workflow.json",
        "workflow_id": workflow.get("workflow_id", ""),
        "workflow_name": workflow.get("name", ""),
        "sequential_depth": len(sequence),
        "dependency_wave_depth": len(waves),
        "wave_widths": [len(ids) for ids in wave_ids],
    }


def lane_bg(ax: plt.Axes, x: float, y: float, width: float, height: float, title: str, note: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.55,
            edgecolor=COLORS["light_gray"],
            facecolor=COLORS["surface"],
            zorder=0,
        )
    )
    ax.text(x + 0.02, y + height - 0.030, title, ha="left", va="center", fontsize=CONFIG.font["annotation"], color=COLORS["dark"], weight="bold")
    ax.text(x + 0.02, y + 0.04, note, ha="left", va="center", fontsize=CONFIG.font["annotation"] - 0.2, color=COLORS["gray_dark"])


def draw_swimlane_box(ax: plt.Axes, x: float, y: float, width: float, height: float, label: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            linewidth=0.45,
            edgecolor=COLORS["white"],
            facecolor=color,
            zorder=3,
        )
    )
    display_label = swimlane_display_label(label)
    fontsize = CONFIG.font["annotation"] - (2.2 if "\n" in display_label else 1.2)
    ax.text(
        x + width / 2,
        y + height / 2,
        display_label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["dark"],
        linespacing=0.88,
        zorder=4,
    )


def draw_swimlane_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    alpha: float = 0.55,
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=5.5,
        linewidth=0.55,
        color=COLORS["gray_dark"],
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)


def wave_box_positions(count: int, low: float, high: float) -> list[float]:
    if count <= 1:
        return [(low + high) / 2 - 0.041]
    return list(np.linspace(high - 0.058, low, count))


def swimlane_label(node_id: str, nodes: dict[str, dict[str, Any]]) -> str:
    node = nodes.get(node_id, {})
    label = str(node.get("label") or node.get("entrypoint") or node_id)
    if node_id.startswith("check_after_"):
        return "LLM"
    lower = label.lower()
    entrypoint = str(node.get("entrypoint") or "").lower()
    combined = f"{lower} {entrypoint}"
    if "find_user" in combined or "query_user" in combined:
        return "QueryUser"
    if "get_order" in combined or "query_order" in combined:
        return "QueryOrder"
    if "cancel" in combined and "order" in combined:
        return "CancelOrder"
    if "return" in combined and "order" in combined:
        return "ReturnOrder"
    if "exchange" in combined and "order" in combined:
        return "ExchangeOrder"
    if "user" in combined:
        return "QueryUser"
    if "product" in combined:
        return "QueryProduct"
    if "address" in combined:
        return "QueryAddress"
    if "payment" in combined:
        return "QueryPayment"
    if "modify" in combined or "update" in combined:
        if "reservation" in combined:
            return "UpdateReservation"
        return "UpdateOrder"
    if "order" in combined or "reservation" in combined:
        return "QueryOrder"
    return camel_label(label)


def swimlane_display_label(label: str) -> str:
    for prefix in ("Query", "Cancel", "Return", "Exchange", "Update"):
        if label.startswith(prefix) and len(label) > len(prefix):
            return f"{prefix}\n{label.removeprefix(prefix)}"
    return label


def camel_label(raw_label: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", raw_label) if part]
    if not parts:
        return raw_label
    return "".join(part[:1].upper() + part[1:] for part in parts[:2])


def swimlane_color(node_id: str, nodes: dict[str, dict[str, Any]]) -> str:
    node = nodes.get(node_id, {})
    if node.get("kind") == "llm":
        return COLORS["component_green"]
    return COLORS["blue_light"]


def workflow_edges_for_swimlane(workflow: dict[str, Any], allowed_nodes: set[str]) -> list[tuple[str, str]]:
    """Show the DAG's precedence relation without redundant transitive edges."""
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in workflow.get("edges", []):
        source = edge.get("from_node")
        target = edge.get("to_node")
        if source in allowed_nodes and target in allowed_nodes and (source, target) not in seen:
            edges.append((source, target))
            seen.add((source, target))
    successors = {node_id: set() for node_id in allowed_nodes}
    for source, target in edges:
        successors[source].add(target)

    def alternate_path(source: str, target: str) -> bool:
        pending = list(successors[source] - {target})
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current not in visited:
                visited.add(current)
                pending.extend(successors[current])
        return False

    return [(source, target) for source, target in edges if not alternate_path(source, target)]


def ranking_by_model(summary_name: str) -> dict[str, dict[str, float]]:
    summary = read_json(WFGEN_RESULTS / summary_name / "deployment_benchmark_summary.json")
    runs: dict[str, dict[str, float]] = {}
    for entry in summary["ranking"]:
        if entry.get("completed_workflows") != 10:
            raise ValueError(f"Incomplete ranking entry in {summary_name}: {entry}")
        model = entry["model_name"].replace(".gguf", "")
        algorithm = entry["algorithm"].removeprefix("cli-")
        runs.setdefault(model, {})[algorithm] = entry["total_wall_ms"] / 1000.0

    for record in summary["manifest"]["records"]:
        run_dir = resolve_results_path(record["run_dir"])
        run = read_json(run_dir / "benchmark_run_record.json")
        if record.get("status") != "completed" or run.get("status") != "completed":
            raise ValueError(f"Run was not completed: {run_dir}")
        if run.get("errors"):
            raise ValueError(f"Completed run has errors: {run_dir}")
    return runs


def completed_offloading_records(
    summary_name: str,
    model_keys: list[str],
    policy_keys: list[str],
) -> dict[tuple[str, str], Path]:
    summary = read_json(WFGEN_RESULTS / summary_name / "deployment_benchmark_summary.json")
    selected: dict[tuple[str, str], Path] = {}
    wanted_models = set(model_keys)
    wanted_policies = set(policy_keys)
    for record in summary["manifest"]["records"]:
        model = record["model_name"].removesuffix(".gguf")
        policy = record["algorithm"].removeprefix("cli-")
        if model not in wanted_models or policy not in wanted_policies:
            continue

        run_dir = resolve_results_path(record["run_dir"])
        run = read_json(run_dir / "benchmark_run_record.json")
        failures = run.get("execution_failures", {})
        if record.get("status") != "completed" or run.get("status") != "completed":
            raise ValueError(f"Selected offloading run was not completed: {run_dir}")
        if record.get("errors") or run.get("errors") or failures.get("failed_workflows"):
            raise ValueError(f"Selected offloading run has final errors: {run_dir}")
        selected[(model, policy)] = run_dir

    expected = {(model, policy) for model in model_keys for policy in policy_keys}
    missing = expected.difference(selected)
    if missing:
        raise ValueError(f"Missing selected offloading runs: {sorted(missing)}")
    return selected


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def workflow_metric_avg(
    workflow: dict[str, Any],
    metric_name: str,
    *,
    default: float | None = None,
) -> float:
    metric = workflow.get("metrics", {}).get(metric_name)
    if metric and metric.get("count", 0) > 0:
        value = parse_optional_float(metric.get("avg"))
        if value is not None:
            return value
    if default is not None:
        return default
    raise ValueError(f"Missing workflow metric {metric_name}: {workflow.get('name', '<unknown>')}")


def workflow_sort_key(index: str) -> tuple[int, str]:
    return (0, f"{int(index):04d}") if index.isdigit() else (1, index)


def resource_usage_workflow_points(run_dir: Path) -> list[dict[str, Any]]:
    by_workflow = read_json(run_dir / "execution" / "resources" / "resource_by_workflow.json")
    points: list[dict[str, Any]] = []
    for workflow_index in sorted(by_workflow, key=workflow_sort_key):
        workflow = by_workflow[workflow_index]
        try:
            cpu_percent = workflow_metric_avg(workflow, "cpu_util_percent")
        except ValueError:
            continue
        points.append(
            {
                "workflow_index": int(workflow_index) if workflow_index.isdigit() else workflow_index,
                "workflow_name": workflow.get("name", ""),
                "cpu_percent": cpu_percent,
                "gpu_busy_percent": workflow_metric_avg(
                    workflow,
                    "gpu_busy_percent",
                    default=0.0,
                ),
            }
        )
    if not points:
        raise ValueError(f"Missing workflow-level resource points: {run_dir}")
    return points


def summarize_resource_workflow_points(
    resource_usage: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for model, policy_points in resource_usage.items():
        summary[model] = {}
        for policy, points in policy_points.items():
            cpu_values = [point["cpu_percent"] for point in points]
            gpu_values = [point["gpu_busy_percent"] for point in points]
            model_counts = [
                point["model_count"]
                for point in points
                if "model_count" in point
            ]
            summary[model][policy] = {
                "workflow_count": len(points),
                "cpu_percent_range": [min(cpu_values), max(cpu_values)],
                "gpu_busy_percent_range": [min(gpu_values), max(gpu_values)],
            }
            if model_counts:
                summary[model][policy]["model_count_range"] = [
                    min(model_counts),
                    max(model_counts),
                ]
                summary[model][policy]["workflows_with_missing_model_samples"] = sum(
                    1 for point in points if point.get("missing_model_ids")
                )
    return summary


OFFLOADING_EVAL_MODEL_LABELS = {
    "qwen3_0_6b_q4_k_m": "Qwen3 0.6B",
    "llama_3_2_1b_instruct_q4_0": "Llama 3.2 1B",
    "qwen1_5_1_8b_chat_q4_0": "Qwen1.5 1.8B",
}
OFFLOADING_EVAL_SELECTED_MODEL_ID = "llama_3_2_1b_instruct_q4_0"
OFFLOADING_EVAL_SELECTED_LABEL = "Llama 3.2\n1B"
OFFLOADING_SINGLE_U_METHOD_IDS = ("seq_cpu_p1", "seq_gpu_p1", "seq_npu_p1")
OFFLOADING_EVAL_POLICIES = [
    (
        "cpu-only",
        "seq_cpu_p1",
        "CPU",
        figure_style.bar_color("p3_cpu"),
        figure_style.SOLID_BAR_HATCH,
        figure_style.METHOD_STYLES["baseline"]["marker"],
    ),
    (
        "gpu-only",
        "seq_gpu_p1",
        "GPU",
        figure_style.bar_color("p3_gpu"),
        figure_style.SOLID_BAR_HATCH,
        figure_style.METHOD_STYLES["baseline_2"]["marker"],
    ),
    (
        "npu-only",
        "seq_npu_p1",
        "NPU",
        figure_style.bar_color("p3_npu"),
        figure_style.SOLID_BAR_HATCH,
        "^",
    ),
    (
        "oracle",
        "wave_opt",
        "Oracle",
        figure_style.bar_color("oracle"),
        figure_style.SOLID_BAR_HATCH,
        figure_style.METHOD_STYLES["ours"]["marker"],
    ),
]
OFFLOADING_EVAL_RESOURCE_DEVICE = "gen3"


def offloading_eval_csv() -> Path:
    return WFGEN_RESULTS / CONFIG.offloading_eval_run_name / "method_time_comparison.csv"


def find_offloading_eval_row(
    rows: list[dict[str, str]],
    *,
    device: str,
    model_id: str,
    method_id: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("device") == device
        and row.get("model_id") == model_id
        and row.get("method_id") == method_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one offloading eval row for {device}/{model_id}/{method_id}, "
            f"got {len(matches)}"
        )
    row = matches[0]
    if row.get("status_note") != "ok":
        raise ValueError(
            f"Offloading eval row is not clean: {device}/{model_id}/{method_id}: "
            f"{row.get('status_note')}"
        )
    return row


def build_offloading_eval_latency(
    rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    devices = sorted({row["device"] for row in rows if row.get("device")})
    latency_seconds: dict[str, dict[str, float]] = {OFFLOADING_EVAL_SELECTED_LABEL: {}}
    latency_details: dict[str, dict[str, Any]] = {OFFLOADING_EVAL_SELECTED_LABEL: {}}
    for policy_key, method_id, _, _, _, _ in OFFLOADING_EVAL_POLICIES:
        per_device: dict[str, float] = {}
        for device in devices:
            row = find_offloading_eval_row(
                rows,
                device=device,
                model_id=OFFLOADING_EVAL_SELECTED_MODEL_ID,
                method_id=method_id,
            )
            per_device[device] = float(row["total_wall_ms"]) / 1000.0
        mean_seconds = float(np.mean(list(per_device.values())))
        latency_seconds[OFFLOADING_EVAL_SELECTED_LABEL][policy_key] = mean_seconds
        latency_details[OFFLOADING_EVAL_SELECTED_LABEL][policy_key] = {
            "mean_seconds": mean_seconds,
            "sample_count": len(per_device),
            "model_count": 1,
            "device_count": len(devices),
            "selected_model_id": OFFLOADING_EVAL_SELECTED_MODEL_ID,
            "selected_model_label": OFFLOADING_EVAL_MODEL_LABELS[
                OFFLOADING_EVAL_SELECTED_MODEL_ID
            ],
            "per_device_seconds": per_device,
        }
    return latency_seconds, latency_details


def build_offloading_oracle_selection(rows: list[dict[str, str]]) -> dict[str, Any]:
    devices = sorted({row["device"] for row in rows if row.get("device")})
    ranking: list[dict[str, Any]] = []
    for model_id, model_label in OFFLOADING_EVAL_MODEL_LABELS.items():
        per_device: dict[str, Any] = {}
        for device in devices:
            best_single: dict[str, Any] | None = None
            for method_id in OFFLOADING_SINGLE_U_METHOD_IDS:
                row = find_offloading_eval_row(
                    rows,
                    device=device,
                    model_id=model_id,
                    method_id=method_id,
                )
                wall_seconds = float(row["total_wall_ms"]) / 1000.0
                candidate = {
                    "method_id": method_id,
                    "method_label": row["method_label"],
                    "wall_seconds": wall_seconds,
                }
                if best_single is None or wall_seconds < best_single["wall_seconds"]:
                    best_single = candidate
            oracle_row = find_offloading_eval_row(
                rows,
                device=device,
                model_id=model_id,
                method_id="wave_opt",
            )
            oracle_seconds = float(oracle_row["total_wall_ms"]) / 1000.0
            assert best_single is not None
            per_device[device] = {
                "device_label": oracle_row["device_label"],
                "best_single": best_single,
                "oracle_seconds": oracle_seconds,
                "speedup": best_single["wall_seconds"] / oracle_seconds,
                "latency_reduction_pct": (
                    (best_single["wall_seconds"] - oracle_seconds)
                    / best_single["wall_seconds"]
                    * 100.0
                ),
                "selected_parallel": oracle_row.get("selected_parallel", ""),
                "selected_method_id": oracle_row.get("selected_method_id", ""),
            }

        best_single_mean = float(
            np.mean(
                [
                    device_summary["best_single"]["wall_seconds"]
                    for device_summary in per_device.values()
                ]
            )
        )
        oracle_mean = float(
            np.mean(
                [
                    device_summary["oracle_seconds"]
                    for device_summary in per_device.values()
                ]
            )
        )
        ranking.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "best_single_mean_seconds": best_single_mean,
                "oracle_mean_seconds": oracle_mean,
                "speedup": best_single_mean / oracle_mean,
                "latency_reduction_pct": (
                    (best_single_mean - oracle_mean) / best_single_mean * 100.0
                ),
                "per_device": per_device,
            }
        )

    ranking.sort(key=lambda entry: entry["latency_reduction_pct"], reverse=True)
    selected = next(
        entry
        for entry in ranking
        if entry["model_id"] == OFFLOADING_EVAL_SELECTED_MODEL_ID
    )
    return {
        "selection_metric": (
            "largest latency reduction versus the fastest single-U policy, "
            "averaged over the two evaluation devices"
        ),
        "selected_model_id": OFFLOADING_EVAL_SELECTED_MODEL_ID,
        "selected_model_label": OFFLOADING_EVAL_MODEL_LABELS[
            OFFLOADING_EVAL_SELECTED_MODEL_ID
        ],
        "selected_matches_largest_gain": ranking[0]["model_id"]
        == OFFLOADING_EVAL_SELECTED_MODEL_ID,
        "selected": selected,
        "ranking": ranking,
    }


def resource_usage_workflow_points_from_row(row: dict[str, str]) -> list[dict[str, Any]]:
    by_workflow_path = resolve_results_path(row["resource_by_workflow"])
    if not by_workflow_path.exists():
        by_workflow_path = resolve_results_path(row["run_dir"]) / "execution" / "resources" / "resource_by_workflow.json"
    by_workflow = read_json(by_workflow_path)
    points: list[dict[str, Any]] = []
    for workflow_index in sorted(by_workflow, key=workflow_sort_key):
        workflow = by_workflow[workflow_index]
        try:
            cpu_percent = workflow_metric_avg(workflow, "cpu_util_percent")
        except ValueError:
            continue
        points.append(
            {
                "workflow_index": int(workflow_index) if workflow_index.isdigit() else workflow_index,
                "workflow_name": workflow.get("name", ""),
                "cpu_percent": cpu_percent,
                "gpu_busy_percent": workflow_metric_avg(
                    workflow,
                    "gpu_busy_percent",
                    default=0.0,
                ),
            }
        )
    if not points:
        raise ValueError(f"Missing workflow-level resource points: {by_workflow_path}")
    return points


def average_resource_points_by_workflow(
    per_model_points: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    expected_model_ids = set(per_model_points)
    for model_id, points in per_model_points.items():
        for point in points:
            grouped.setdefault(point["workflow_index"], []).append({**point, "model_id": model_id})

    averaged: list[dict[str, Any]] = []
    for workflow_index in sorted(grouped, key=lambda index: workflow_sort_key(str(index))):
        points = grouped[workflow_index]
        model_ids = [point["model_id"] for point in points]
        averaged.append(
            {
                "workflow_index": workflow_index,
                "workflow_name": points[0].get("workflow_name", ""),
                "cpu_percent": float(np.mean([point["cpu_percent"] for point in points])),
                "gpu_busy_percent": float(np.mean([point["gpu_busy_percent"] for point in points])),
                "model_count": len(model_ids),
                "model_ids": model_ids,
                "missing_model_ids": sorted(expected_model_ids.difference(model_ids)),
            }
        )
    if not averaged:
        raise ValueError("Missing workflow-level resource points for all-model average")
    return averaged


def build_offloading_eval_resource_usage(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    resource_usage: dict[str, dict[str, list[dict[str, Any]]]] = {
        OFFLOADING_EVAL_SELECTED_LABEL: {}
    }
    for policy_key, method_id, _, _, _, _ in OFFLOADING_EVAL_POLICIES:
        per_model_points: dict[str, list[dict[str, Any]]] = {}
        row = find_offloading_eval_row(
            rows,
            device=OFFLOADING_EVAL_RESOURCE_DEVICE,
            model_id=OFFLOADING_EVAL_SELECTED_MODEL_ID,
            method_id=method_id,
        )
        per_model_points[OFFLOADING_EVAL_SELECTED_MODEL_ID] = (
            resource_usage_workflow_points_from_row(row)
        )
        resource_usage[OFFLOADING_EVAL_SELECTED_LABEL][policy_key] = (
            average_resource_points_by_workflow(per_model_points)
        )
    return resource_usage


def plot_offloading_latency(model_runs: dict[str, dict[str, float]]) -> None:
    policies = [
        (policy_key, legend_label, color, hatch)
        for policy_key, _, legend_label, color, hatch, _ in OFFLOADING_EVAL_POLICIES
    ]
    labels = list(model_runs.keys())
    x = np.arange(len(labels))
    width = 0.18
    offsets = (np.arange(len(policies)) - (len(policies) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["offloading_latency"])
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.24, top=0.86)
    max_minutes = 0.0
    for offset, (policy_key, legend_label, color, hatch) in zip(offsets, policies):
        values = [runs[policy_key] / 60.0 for runs in model_runs.values()]
        max_minutes = max(max_minutes, max(values))
        ax.bar(
            x + offset,
            values,
            width,
            label=legend_label,
            color=color,
            edgecolor=figure_style.BAR_EDGE,
            linewidth=figure_style.BAR_EDGE_WIDTH,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("E2E latency (min)", labelpad=1)
    y_max = max(30.0, np.ceil(max_minutes / 5.0) * 5.0)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 0.1, 10))
    box_axes(ax)
    handles, legend_labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend(
        fig,
        handles,
        legend_labels,
        ncol=4,
        anchor=(0.56, 0.99),
        columnspacing=0.55,
        handlelength=0.75,
        handletextpad=0.25,
        prop={"size": CONFIG.font["legend"] - 0.6},
    )
    savefig(fig, "offloading/offloading_latency", exact_canvas=True)


def plot_offloading_resource_usage(
    resource_usage: dict[str, dict[str, list[dict[str, Any]]]],
) -> None:
    fig, ax = plt.subplots(figsize=CONFIG.figure_sizes["offloading_resource"])
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.25, top=0.86)
    ax.plot(
        [0, 100],
        [0, 100],
        color=COLORS["light_gray"],
        linewidth=0.55,
        linestyle=":",
        zorder=1,
    )
    policy_styles = {
        policy_key: (legend_label, color, marker)
        for policy_key, _, legend_label, color, _, marker in OFFLOADING_EVAL_POLICIES
    }
    for policy, (legend_label, color, marker) in policy_styles.items():
        first_model = True
        for model in resource_usage:
            points = resource_usage[model][policy]
            ax.scatter(
                [point["cpu_percent"] for point in points],
                [point["gpu_busy_percent"] for point in points],
                s=19,
                color=color,
                marker=marker,
                alpha=0.72,
                edgecolor=COLORS["white"],
                linewidth=0.22,
                label=legend_label if first_model else None,
                zorder=3,
            )
            first_model = False

    ax.set_xlabel("CPU util. (%)", labelpad=1)
    ax.set_ylabel("GPU busy (%)", labelpad=1)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    box_axes(ax, grid_axis="both")
    handles, legend_labels = ax.get_legend_handles_labels()
    figure_style.boxed_legend(
        fig,
        handles,
        legend_labels,
        ncol=2,
        anchor=(0.56, 0.99),
        columnspacing=0.8,
        handlelength=0.85,
        handletextpad=0.25,
        markerscale=1.15,
        prop={"size": CONFIG.font["legend"] - 0.5},
    )
    savefig(fig, "offloading/offloading_resource_usage", exact_canvas=True)


def plot_offloading_policy_outcomes() -> dict[str, Any]:
    source_csv = offloading_eval_csv()
    rows = read_csv_rows(source_csv)
    oracle_selection = build_offloading_oracle_selection(rows)
    latency_seconds, latency_details = build_offloading_eval_latency(rows)
    resource_usage = build_offloading_eval_resource_usage(rows)

    plot_offloading_latency(latency_seconds)
    plot_offloading_resource_usage(resource_usage)

    return {
        "source_csv": path_for_record(source_csv),
        "latency_policy_methods": {
            policy_key: method_id
            for policy_key, method_id, _, _, _, _ in OFFLOADING_EVAL_POLICIES
        },
        "latency_aggregation": "mean over completed rows from both evaluation devices",
        "model_selection": {
            "plot_label": OFFLOADING_EVAL_SELECTED_LABEL,
            "available_models": OFFLOADING_EVAL_MODEL_LABELS,
            **oracle_selection,
        },
        "resource_device": OFFLOADING_EVAL_RESOURCE_DEVICE,
        "successful_latency_seconds": latency_seconds,
        "successful_latency_details": latency_details,
        "resource_usage_workflow_points_percent": summarize_resource_workflow_points(resource_usage),
        "interpretation": (
            "Latest evaluation rows show the cost of static CPU/GPU/NPU-only placement. "
            "The oracle policy uses the per-device/per-model best p=2/p=3 selection "
            "from the source CSV. Fig. 6 selects the model with the largest oracle gain "
            "over the fastest single-U policy; latency averages the two devices; CPU/GPU "
            "resource points use 8 Gen 3, where GPU-busy samples are available, and "
            "each workflow point comes from the selected model."
        ),
    }


def write_provenance(summary: dict[str, Any]) -> None:
    provenance = {
        "generated_by": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "wfgen_results": path_for_record(WFGEN_RESULTS),
        "generated_figures": list(CONFIG.output_files),
        "used_successful": {
            "overview_stack": {
                "path": "script-drawn schematic",
                "outputs": [
                    "overview/background.pdf",
                    "overview/background.svg",
                ],
                "status": (
                    "reproducible vector schematic of the ODA stack, showing "
                    "application, framework, runtime, hardware, unified memory, "
                    "and a Qualcomm SoC execution path"
                ),
                "metrics": summary["overview_stack"],
            },
            "planning_cost": {
                "path": CONFIG.planning_run_name,
                "outputs": [
                    "planning/planning_cost_latency.pdf",
                    "planning/planning_cost_latency.svg",
                    "planning/planning_cost_breakdown.pdf",
                    "planning/planning_cost_breakdown.svg",
                ],
                "status": (
                    "Qwen cloud workflow planning over all 280 tau-bench train samples; "
                    "prompt-component token attribution uses character-share estimates"
                ),
                "metrics": summary["planning_cost"],
            },
            "codegen_cost": {
                "path": f"{CONFIG.planning_codegen_run_name}/breakdown/codegen_breakdown.json",
                "outputs": [
                    "codegen/codegen_latency.pdf",
                    "codegen/codegen_latency.svg",
                    "codegen/codegen_tokens.pdf",
                    "codegen/codegen_tokens.svg",
                ],
                "status": (
                    "qwen-plus executable-code generation for all 280 planned workflows; "
                    "costs include failed first-pass attempts and retry requests needed "
                    "to obtain syntax-compilable Python outputs"
                ),
                "metrics": summary["codegen_cost"],
            },
            "topology_parallelism": {
                "path": topology_provenance_path(),
                "outputs": [
                    "topology/topology_parallelism.pdf",
                    "topology/topology_parallelism.svg",
                    "topology/topology_swimlane.pdf",
                    "topology/topology_swimlane.svg",
                ],
                "status": (
                    "full-builder 280-workflow planning artifact for per-workflow reduction CDF; "
                    f"{CONFIG.topology_swimlane_workflow_id} illustrates sequential versus dependency-wave execution"
                ),
                "metrics": summary["topology_parallelism"],
            },
            "successful_offloading": {
                "path": f"{CONFIG.offloading_eval_run_name}/method_time_comparison.csv",
                "outputs": [
                    "offloading/offloading_latency.pdf",
                    "offloading/offloading_latency.svg",
                    "offloading/offloading_resource_usage.pdf",
                    "offloading/offloading_resource_usage.svg",
                ],
                "status": (
                    "latest evaluation method comparison rows are filtered to the "
                    "model with the largest oracle gain over the fastest single-U "
                    "baseline; latency averages the two devices, and resource points "
                    "use 8 Gen 3 where GPU-busy samples are available"
                ),
                "metrics": summary["offloading_policy_outcomes"],
            },
        },
        "aborted_or_not_plotted": {
            "deployment_benchmark_20260605_*": (
                "partial, timeout-filtered, non-retryable, or retry-contaminated "
                "comparison roots; not used for motivation claims"
            ),
            "deployment_benchmark_8elite_2models_20260609": (
                "root has manifest/progress only and no final summary; one model has a "
                "completed subset, but the comparison root is incomplete"
            ),
            "backend_smoke_startup_resume_cooldown": (
                "smoke/startup/resume tests are useful for debugging only, not paper "
                "motivation figures"
            ),
            "batch_throughput": (
                "no completed batch-sweep result found in WFGen/results, so batching "
                "speedup is not claimed"
            ),
            "codegen_llm_benchmark_all10_20260610": (
                "initial 8192-output-token codegen run is kept as an audit trail but "
                "not plotted directly because four larger workflows were truncated; "
                "the final plotted result merges those workflows from the 16384-token retry"
            ),
        },
    }
    provenance_path = OUT_DIR / "provenance" / "source_classification.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


def topology_provenance_path() -> str:
    topology_path = f"{CONFIG.topology_run_name}/planning/workflow.json"
    swimlane_path = f"{CONFIG.topology_swimlane_run_name}/planning/workflow.json"
    if topology_path == swimlane_path:
        return topology_path
    return f"{topology_path}; {swimlane_path} for the swimlane example"


def main() -> None:
    configure_style()
    summary = {
        
        "planning_cost": plot_planning_cost(),
        "codegen_cost": plot_codegen_cost(),
        "topology_parallelism": plot_topology_parallelism(),
        
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("Rendered active motivation planning, code-generation and topology panels.")


if __name__ == "__main__":
    main()
