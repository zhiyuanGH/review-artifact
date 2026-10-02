"""Generate generation-efficiency figures for the MOBench case study.

The figure compares two benchmark-generation pipelines over the same 280
tau-bench workflows:

* LLM code generation: LLM planning + LLM full-code generation.
* MOBench: LLM call-chain planning + deterministic interpretation.

The script reads the paper-facing WFGen figure-data snapshot and exports vector figures
for direct inclusion in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_studies_generation_figures"
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
FROZEN_INPUT_DIR = REPO_ROOT / "data"
GITHUB_ROOT = REPO_ROOT.parent
WFGEN_ROOT = REPO_ROOT / "unbundled_raw"
FIGURE_DATA_ROOT = WFGEN_ROOT / "figure_data"
FIG8_DATA_DIR = (
    FIGURE_DATA_ROOT
    / "fig8-tau_bench_all_samples_call_chain_qwen_20260626_r2-generation_efficiency_20260629"
)
CALL_CHAIN_RESULTS_DIR = (
    WFGEN_ROOT / "results" / "tau_bench_all_samples_call_chain_qwen_20260626_r2"
)
CODEGEN_RESULTS_DIR = CALL_CHAIN_RESULTS_DIR / "codegen_qwen_20260627"
CODEGEN_FIGURE_DATA_DIR = (
    FIGURE_DATA_ROOT
    / "fig4-tau_bench_all_samples_call_chain_qwen_20260626_r2-codegen_qwen_20260627"
)


@dataclass(frozen=True)
class GenerationFigureConfig:
    planning_usage_csv: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "planning" / "per_sample_usage.csv"
    )
    planning_breakdown: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "planning" / "final_breakdown.json"
    )
    codegen_breakdown: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "codegen" / "codegen_breakdown.json"
    )
    codegen_per_sample_csv: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "codegen" / "per_sample_codegen.csv"
    )
    interpreter_timing: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "interpreter" / "interpreter_timing.json"
    )
    interpreter_breakdown: Path = field(
        default_factory=lambda: FIG8_DATA_DIR / "interpreter" / "interpret_breakdown.json"
    )
    output_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "figures" / "studies" / "generation"
    )
    colors: dict[str, str] = field(
        default_factory=lambda: {
            "planning": figure_style.METHOD_COLORS["planning"],
            "codegen": figure_style.METHOD_COLORS["codegen"],
            "odabench": figure_style.METHOD_COLORS["odabench"],
            "naive": figure_style.METHOD_COLORS["naive"],
            "interpreter": figure_style.SEMANTIC_COLORS["component"]["interpreter"],
            "ink": figure_style.BASE_COLORS["ink"],
            "muted": figure_style.BASE_COLORS["muted"],
            "line": figure_style.BASE_COLORS["line"],
            "grid": figure_style.BASE_COLORS["grid"],
            "white": figure_style.BASE_COLORS["white"],
        }
    )


CONFIG = GenerationFigureConfig()
PROVENANCE_PATH = CONFIG.output_dir / "generation_efficiency_provenance.json"
GENERATION_FONT = {
    "base": 8.4,
    "title": 8.5,
    "label": 8.4,
    "tick": 7.6,
    "legend": 6.9,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_source_path(primary: Path, candidates: list[Path]) -> Path:
    if primary.exists():
        return primary
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return primary


def row_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key) or 0.0)


def row_int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key) or 0))


def configure_style() -> None:
    figure_style.configure_matplotlib(plt)
    plt.rcParams.update(
        {
            "font.size": GENERATION_FONT["base"],
            "axes.titlesize": GENERATION_FONT["title"],
            "axes.labelsize": GENERATION_FONT["label"],
            "xtick.labelsize": GENERATION_FONT["tick"],
            "ytick.labelsize": GENERATION_FONT["tick"],
            "legend.fontsize": GENERATION_FONT["legend"],
        }
    )


def summarize_planning(usage_csv: Path, final_breakdown_path: Path) -> dict[str, Any]:
    rows = sorted(read_csv_rows(usage_csv), key=lambda row: row_int(row, "sample_index"))
    if not rows:
        raise FileNotFoundError(f"No planning usage rows found in {usage_csv}")

    final_breakdown = read_json(final_breakdown_path)
    workflow_count = len(rows)
    if int(final_breakdown.get("workflow_count") or 0) != workflow_count:
        raise ValueError(
            "Planning CSV and final breakdown disagree on workflow count: "
            f"{workflow_count} vs {final_breakdown.get('workflow_count')}"
        )

    per_workflow = [
        {
            "workflow_index": row_int(row, "sample_index"),
            "workflow_id": row.get("workflow_id", ""),
            "task_id": row.get("task_id", ""),
            "domain": row.get("domain", ""),
            "status": row.get("status", ""),
            "wall_ms": row_float(row, "wall_ms"),
            "prompt_tokens": row_int(row, "prompt_tokens"),
            "completion_tokens": row_int(row, "completion_tokens"),
            "total_tokens": row_int(row, "total_tokens"),
        }
        for row in rows
    ]

    return {
        "workflow_count": workflow_count,
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "wall_ms": sum(float(row.get("wall_ms") or 0.0) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "status_counts": final_breakdown.get("status_counts", {}),
        "domain_counts": final_breakdown.get("domain_counts", {}),
        "per_workflow": per_workflow,
    }


def summarize_codegen(codegen_breakdown: Path, per_sample_csv: Path) -> dict[str, Any]:
    data = read_json(codegen_breakdown)
    cost = data["cost_including_retries"]
    first_pass = data.get("first_pass", {})
    retry = data.get("retry", {})
    rows = sorted(read_csv_rows(per_sample_csv), key=lambda row: row_int(row, "prompt_index"))
    sample_count = int(data["sample_count"])
    if len(rows) != sample_count:
        raise ValueError(
            "Code-generation CSV and breakdown disagree on workflow count: "
            f"{len(rows)} vs {sample_count}"
        )

    per_workflow = [
        {
            "workflow_index": row_int(row, "prompt_index"),
            "selection_index": row_int(row, "selection_index"),
            "workflow_id": row.get("workflow_id", ""),
            "task_id": row.get("task_id", ""),
            "domain": row.get("domain", ""),
            "final_status": row.get("final_status", ""),
            "compile_attempts": row_int(row, "compile_attempts"),
            "wall_ms": row_float(row, "total_wall_ms_including_retries"),
            "prompt_tokens": row_int(row, "total_prompt_tokens_including_retries"),
            "completion_tokens": row_int(row, "total_completion_tokens_including_retries"),
            "total_tokens": row_int(row, "total_tokens_including_retries"),
        }
        for row in rows
    ]
    per_workflow_wall_ms = sum(row["wall_ms"] for row in per_workflow)
    per_workflow_tokens = sum(row["total_tokens"] for row in per_workflow)
    if abs(per_workflow_wall_ms - float(cost["wall_ms_sum"])) > 1e-3:
        raise ValueError("Code-generation per-workflow wall time does not match aggregate")
    if per_workflow_tokens != int(cost["total_tokens"]):
        raise ValueError("Code-generation per-workflow tokens do not match aggregate")

    return {
        "workflow_count": sample_count,
        "request_count": int(cost["request_count"]),
        "wall_ms": float(cost["wall_ms_sum"]),
        "prompt_tokens": int(cost["prompt_tokens"]),
        "completion_tokens": int(cost["completion_tokens"]),
        "total_tokens": int(cost["total_tokens"]),
        "first_pass_compile_success_rate": float(first_pass.get("compile_success_rate", 0.0)),
        "first_pass_compile_success_count": int(first_pass.get("compile_success_count", 0)),
        "retry_request_count": int(retry.get("request_count", 0)),
        "per_workflow": per_workflow,
    }


def summarize_interpreter(
    timing_path: Path,
    breakdown_path: Path,
    fallback_metrics: dict[str, Any],
) -> dict[str, Any]:
    breakdown = read_json(breakdown_path)
    fallback_interpreter = fallback_metrics.get("interpreter", {})
    if timing_path.exists():
        timing = read_json(timing_path)
        workflow_count = int(timing["workflow_count"])
        wall_ms = float(timing["wall_ms"])
        visualization_count = int(timing.get("visualization_count") or 0)
        timing_scope = timing.get("timing_scope", "")
        allocation_note = "measured aggregate timing"
    elif fallback_interpreter:
        workflow_count = int(
            fallback_interpreter.get("workflow_count")
            or breakdown.get("workflow_count")
            or 0
        )
        wall_ms = float(fallback_interpreter["wall_ms"])
        visualization_count = int(
            fallback_interpreter.get("visualization_count")
            or breakdown.get("visualization_count")
            or 0
        )
        timing_scope = fallback_interpreter.get("timing_scope", "")
        allocation_note = "aggregate timing from existing provenance"
    else:
        raise FileNotFoundError(
            f"Missing interpreter timing source {timing_path} and no provenance fallback"
        )

    if int(breakdown.get("workflow_count") or 0) != workflow_count:
        raise ValueError(
            "Interpreter timing and breakdown disagree on workflow count: "
            f"{workflow_count} vs {breakdown.get('workflow_count')}"
        )

    return {
        "workflow_count": workflow_count,
        "wall_ms": wall_ms,
        "runtime_config_count": int(breakdown.get("runtime_config_count") or 0),
        "compiled_script_count": int(breakdown.get("compiled_script_count") or 0),
        "visualization_count": visualization_count,
        "timing_scope": timing_scope,
        "per_workflow_allocation": allocation_note,
    }


def build_metrics() -> dict[str, Any]:
    frozen = FROZEN_INPUT_DIR / "generation.json"
    if frozen.exists():
        return read_json(frozen)
    planning_usage_csv = resolve_source_path(
        CONFIG.planning_usage_csv,
        [CALL_CHAIN_RESULTS_DIR / "breakdown" / "per_sample_usage.csv"],
    )
    planning_breakdown = resolve_source_path(
        CONFIG.planning_breakdown,
        [CALL_CHAIN_RESULTS_DIR / "breakdown" / "final_breakdown.json"],
    )
    codegen_breakdown = resolve_source_path(
        CONFIG.codegen_breakdown,
        [
            CODEGEN_FIGURE_DATA_DIR / "breakdown" / "codegen_breakdown.json",
            CODEGEN_RESULTS_DIR / "breakdown" / "codegen_breakdown.json",
        ],
    )
    codegen_per_sample_csv = resolve_source_path(
        CONFIG.codegen_per_sample_csv,
        [
            CODEGEN_FIGURE_DATA_DIR / "breakdown" / "per_sample_codegen.csv",
            CODEGEN_RESULTS_DIR / "breakdown" / "per_sample_codegen.csv",
        ],
    )
    interpreter_breakdown = resolve_source_path(
        CONFIG.interpreter_breakdown,
        [CALL_CHAIN_RESULTS_DIR / "deployment" / "breakdown" / "interpret_breakdown.json"],
    )
    fallback_metrics = read_json(PROVENANCE_PATH) if PROVENANCE_PATH.exists() else {}

    source_paths = [
        planning_usage_csv,
        planning_breakdown,
        codegen_breakdown,
        codegen_per_sample_csv,
        interpreter_breakdown,
    ]
    if any(not path.exists() for path in source_paths) and PROVENANCE_PATH.exists():
        metrics = read_json(PROVENANCE_PATH)
        metrics["source_mode"] = "existing_provenance_fallback"
        return metrics

    planning = summarize_planning(planning_usage_csv, planning_breakdown)
    codegen = summarize_codegen(codegen_breakdown, codegen_per_sample_csv)
    interpreter = summarize_interpreter(
        CONFIG.interpreter_timing,
        interpreter_breakdown,
        fallback_metrics,
    )

    if planning["workflow_count"] != codegen["workflow_count"]:
        raise ValueError(
            "Planning and code-generation runs cover different numbers of workflows: "
            f"{planning['workflow_count']} vs {codegen['workflow_count']}"
        )
    if planning["workflow_count"] != interpreter["workflow_count"]:
        raise ValueError(
            "Planning and interpreter runs cover different numbers of workflows: "
            f"{planning['workflow_count']} vs {interpreter['workflow_count']}"
        )

    workflow_count = int(planning["workflow_count"])
    planning_wall_ms = float(planning["wall_ms"])
    interpreter_wall_ms = float(interpreter["wall_ms"])
    codegen_wall_ms = float(codegen["wall_ms"])
    odabench_wall_ms = planning_wall_ms + interpreter_wall_ms
    naive_wall_ms = planning_wall_ms + codegen_wall_ms
    odabench_tokens = int(planning["total_tokens"])
    naive_tokens = odabench_tokens + int(codegen["total_tokens"])
    interpreter_wall_ms_per_workflow = interpreter_wall_ms / workflow_count
    codegen_by_index = {
        row["workflow_index"]: row for row in codegen["per_workflow"]
    }
    per_workflow: list[dict[str, Any]] = []
    for planning_row in planning["per_workflow"]:
        workflow_index = planning_row["workflow_index"]
        codegen_row = codegen_by_index.get(workflow_index)
        if codegen_row is None:
            raise ValueError(f"Missing code-generation row for workflow {workflow_index}")
        if codegen_row["workflow_id"] != planning_row["workflow_id"]:
            raise ValueError(
                "Planning and code-generation workflow IDs disagree at "
                f"{workflow_index}: {planning_row['workflow_id']} vs "
                f"{codegen_row['workflow_id']}"
            )
        odabench_workflow_wall_ms = (
            planning_row["wall_ms"] + interpreter_wall_ms_per_workflow
        )
        naive_workflow_wall_ms = planning_row["wall_ms"] + codegen_row["wall_ms"]
        per_workflow.append(
            {
                "workflow_index": workflow_index,
                "workflow_id": planning_row["workflow_id"],
                "domain": planning_row["domain"],
                "planning_wall_ms": planning_row["wall_ms"],
                "codegen_wall_ms": codegen_row["wall_ms"],
                "interpreter_wall_ms": interpreter_wall_ms_per_workflow,
                "odabench_wall_ms": odabench_workflow_wall_ms,
                "naive_wall_ms": naive_workflow_wall_ms,
                "planning_tokens": planning_row["total_tokens"],
                "codegen_tokens": codegen_row["total_tokens"],
                "odabench_tokens": planning_row["total_tokens"],
                "naive_tokens": planning_row["total_tokens"] + codegen_row["total_tokens"],
            }
        )

    return {
        "workflow_count": workflow_count,
        "planning": planning,
        "codegen": codegen,
        "interpreter": interpreter,
        "per_workflow": per_workflow,
        "odabench": {
            "wall_ms": odabench_wall_ms,
            "planning_wall_ms": planning_wall_ms,
            "interpreter_wall_ms": interpreter_wall_ms,
            "avg_wall_ms": odabench_wall_ms / workflow_count,
            "total_tokens": odabench_tokens,
            "planning_tokens": odabench_tokens,
            "interpreter_tokens": 0,
            "avg_tokens": odabench_tokens / workflow_count,
        },
        "naive": {
            "wall_ms": naive_wall_ms,
            "planning_wall_ms": planning_wall_ms,
            "codegen_wall_ms": codegen_wall_ms,
            "avg_wall_ms": naive_wall_ms / workflow_count,
            "total_tokens": naive_tokens,
            "planning_tokens": odabench_tokens,
            "codegen_tokens": int(codegen["total_tokens"]),
            "avg_tokens": naive_tokens / workflow_count,
        },
        "improvement": {
            "speedup_x": naive_wall_ms / odabench_wall_ms,
            "time_reduction_pct": 100.0 * (naive_wall_ms - odabench_wall_ms) / naive_wall_ms,
            "token_reduction_pct": 100.0 * (naive_tokens - odabench_tokens) / naive_tokens,
        },
        "sources": {
            "planning_usage_csv": str(planning_usage_csv),
            "planning_breakdown": str(planning_breakdown),
            "codegen_breakdown": str(codegen_breakdown),
            "codegen_per_sample_csv": str(codegen_per_sample_csv),
            "interpreter_timing": str(CONFIG.interpreter_timing),
            "interpreter_breakdown": str(interpreter_breakdown),
        },
    }


def add_vertical_panel(
    ax: plt.Axes,
    *,
    ylabel: str,
    naive_components: list[tuple[str, float, str]],
    odabench_components: list[tuple[str, float, str]],
    ylim: float,
    legend_labels: list[str],
) -> None:
    colors = CONFIG.colors
    x_naive = 0.0
    x_odabench = 1.0
    width = 0.52
    seen_labels: set[str] = set()

    def draw_stack(x: float, components: list[tuple[str, float, str]]) -> None:
        bottom = 0.0
        for key, value, label in components:
            if value <= 0:
                continue
            plot_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)
            ax.bar(
                x,
                value,
                bottom=bottom,
                width=width,
                color=colors[key],
                edgecolor=colors["white"],
                linewidth=0.5,
                label=plot_label,
            )
            bottom += value

    draw_stack(x_naive, naive_components)
    draw_stack(x_odabench, odabench_components)

    ax.set_xticks([x_naive, x_odabench])
    ax.set_xticklabels(["LLM code\ngeneration", "MOBench"])
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", length=2.5)
    ax.set_xlim(-0.62, 1.62)
    figure_style.box_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    figure_style.boxed_legend(
        ax.figure,
        [handle_by_label[label] for label in legend_labels if label in handle_by_label],
        [label for label in legend_labels if label in handle_by_label],
        ncol=2,
        anchor=(0.54, 0.99),
        columnspacing=0.55,
        handlelength=0.70,
        handletextpad=0.22,
        labelspacing=0.16,
        prop={"size": 5.4},
    )


def save_figure(fig: plt.Figure, stem: Path, preview_dir: Path | None = None) -> None:
    figure_style.save_pdf_svg(
        fig,
        stem,
        exact_canvas=True,
        top_margin_points=2.0,
        creator="Paper_MOBench generation figure generator",
        preview_dir=preview_dir,
    )
    plt.close(fig)
    strip_trailing_whitespace(stem.with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def empirical_cdf(values: list[float]) -> tuple[list[float], list[float]]:
    sorted_values = sorted(value for value in values if value >= 0)
    if not sorted_values:
        return [], []
    count = len(sorted_values)
    return sorted_values, [(index + 1) / count for index in range(count)]


def cdf_step_points(values: list[float], x_min: float, x_max: float) -> tuple[list[float], list[float]]:
    sorted_values = sorted(value for value in values if value >= 0)
    if not sorted_values:
        return [], []
    count = len(sorted_values)
    segment_values = [value for value in sorted_values if x_min < value <= x_max]

    def cdf_at(x_value: float) -> float:
        return sum(value <= x_value for value in sorted_values) / count

    x_points = [x_min, *segment_values, x_max]
    y_points = [cdf_at(x_min), *(cdf_at(value) for value in segment_values), cdf_at(x_max)]
    return x_points, y_points


def draw_axis_break(left_ax: plt.Axes, right_ax: plt.Axes) -> None:
    color = CONFIG.colors["ink"]
    marker_size = 0.022
    kwargs = {"color": color, "clip_on": False, "linewidth": 0.65}
    left_ax.plot(
        (1 - marker_size, 1 + marker_size),
        (-marker_size, marker_size),
        transform=left_ax.transAxes,
        **kwargs,
    )
    left_ax.plot(
        (1 - marker_size, 1 + marker_size),
        (1 - marker_size, 1 + marker_size),
        transform=left_ax.transAxes,
        **kwargs,
    )
    right_ax.plot(
        (-marker_size, marker_size),
        (-marker_size, marker_size),
        transform=right_ax.transAxes,
        **kwargs,
    )
    right_ax.plot(
        (-marker_size, marker_size),
        (1 - marker_size, 1 + marker_size),
        transform=right_ax.transAxes,
        **kwargs,
    )


def plot_broken_cdf_panel(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    specs: list[tuple[list[float], str, str, str]],
    xlabel: str,
    x_segments: list[tuple[float, float]],
    x_ticks: list[list[float]],
) -> None:
    all_values: list[float] = []
    for ax, (x_min, x_max) in zip(axes, x_segments):
        for values, label, color, linestyle in specs:
            cdf_x, cdf_y = cdf_step_points(values, x_min, x_max)
            all_values.extend(cdf_x)
            ax.plot(
                cdf_x,
                cdf_y,
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                drawstyle="steps-post",
                label=label if ax is axes[0] else "_nolegend_",
                zorder=3,
            )

    if not all_values:
        raise ValueError("No CDF values to plot")

    for index, ax in enumerate(axes):
        ax.set_xlim(*x_segments[index])
        ax.set_xticks(x_ticks[index])
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        figure_style.box_axes(ax, grid_axis="both")
        ax.tick_params(axis="both", length=2.5)
        if index > 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
        else:
            ax.set_ylabel("CDF")

    fig.text(0.575, 0.075, xlabel, ha="center", va="center", color=CONFIG.colors["ink"])
    draw_axis_break(axes[0], axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        labels,
        ncol=2,
        labelspacing=0.15,
        columnspacing=0.75,
        handlelength=1.0,
        handletextpad=0.25,
        prop={"size": GENERATION_FONT["legend"]},
    )


def plot_generation_efficiency(metrics: dict[str, Any], preview_dir: Path | None = None) -> None:
    configure_style()
    colors = CONFIG.colors
    rows = metrics.get("per_workflow", [])
    if len(rows) != int(metrics["workflow_count"]):
        raise ValueError("Generation CDF requires one per-workflow row per workflow")

    naive_time_s = [float(row["naive_wall_ms"]) / 1000.0 for row in rows]
    odabench_time_s = [float(row["odabench_wall_ms"]) / 1000.0 for row in rows]
    naive_tokens_k = [float(row["naive_tokens"]) / 1000.0 for row in rows]
    odabench_tokens_k = [float(row["odabench_tokens"]) / 1000.0 for row in rows]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2.55, 1.35),
        sharey=True,
        gridspec_kw={"width_ratios": [0.72, 1.0], "wspace": 0.08},
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.28, top=0.78)
    plot_broken_cdf_panel(
        fig,
        list(axes),
        specs=[
            (naive_time_s, "LLM code generation", colors["naive"], "-"),
            (odabench_time_s, "MOBench", colors["odabench"], "-"),
        ],
        xlabel="Generation time (s)",
        x_segments=[(0, 20), (35, 270)],
        x_ticks=[[0, 10], [50, 150, 250]],
    )
    save_figure(fig, CONFIG.output_dir / "generation_time_efficiency", preview_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(2.55, 1.35),
        sharey=True,
        gridspec_kw={"width_ratios": [0.72, 1.0], "wspace": 0.08},
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.28, top=0.78)
    plot_broken_cdf_panel(
        fig,
        list(axes),
        specs=[
            (naive_tokens_k, "LLM code generation", colors["naive"], "-"),
            (odabench_tokens_k, "MOBench", colors["odabench"], "-"),
        ],
        xlabel="Tokens per workflow (k)",
        x_segments=[(0, 3), (4.5, 26.5)],
        x_ticks=[[0, 2], [5, 15, 25]],
    )
    save_figure(fig, CONFIG.output_dir / "generation_token_efficiency", preview_dir)


def summary_without_per_workflow(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "per_workflow"}


def compact_cdf_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "workflow_index": row["workflow_index"],
            "workflow_id": row["workflow_id"],
            "domain": row["domain"],
            "odabench_wall_ms": row["odabench_wall_ms"],
            "naive_wall_ms": row["naive_wall_ms"],
            "odabench_tokens": row["odabench_tokens"],
            "naive_tokens": row["naive_tokens"],
        }
        for row in rows
    ]


def write_provenance(metrics: dict[str, Any]) -> None:
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "figures": [
            "generation_time_efficiency",
            "generation_token_efficiency",
        ],
        "workflow_count": metrics["workflow_count"],
        "sources": metrics["sources"],
        "planning": summary_without_per_workflow(metrics["planning"]),
        "codegen": summary_without_per_workflow(metrics["codegen"]),
        "interpreter": metrics["interpreter"],
        "per_workflow": compact_cdf_rows(metrics.get("per_workflow", [])),
        "odabench": metrics["odabench"],
        "naive": metrics["naive"],
        "improvement": metrics["improvement"],
    }
    if metrics.get("source_mode"):
        provenance["source_mode"] = metrics["source_mode"]
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG previews used in visual QA.",
    )
    args = parser.parse_args()

    metrics = build_metrics()
    plot_generation_efficiency(metrics, preview_dir=args.preview_dir)
    write_provenance(metrics)

    print(f"Wrote {CONFIG.output_dir / 'generation_time_efficiency.pdf'}")
    print(f"Wrote {CONFIG.output_dir / 'generation_token_efficiency.pdf'}")
    print(
        "Generation improvement: "
        f"{metrics['improvement']['speedup_x']:.2f}x faster, "
        f"{metrics['improvement']['time_reduction_pct']:.1f}% lower time, "
        f"{metrics['improvement']['token_reduction_pct']:.1f}% lower tokens"
    )


if __name__ == "__main__":
    main()
