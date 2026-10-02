"""Plot fixed p=3 same-U latency against the frozen offline Oracle selection.

The current ten-workflow snapshot compares CPU, GPU, and NPU on 8 Gen 3
with the best-of-eight successful measured configuration per workflow,
including all three plotted single-backend baselines.
The historical Oracle CSV loader remains available if the snapshot is absent.
Failed workflows have explicit markers and no complete-latency bars.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_p3_same_u_all_workflow_preview"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from . import figure_style
except ImportError:
    import figure_style


REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_ROOT = REPO_ROOT.parent
WFGEN_ROOT = REPO_ROOT / "unbundled_raw"

MODEL_ID = "qwen1_5_1_8b_chat_q4_0"
MODEL_LABEL = "Qwen1.5 1.8B"
DEVICE_ORDER = ["gen3"]
DEVICE_LABELS = {
    "gen3": "8 Gen 3",
}
METHODS = [
    ("p3_cpu", "p=3 CPU", "cpu"),
    ("p3_gpu", "p=3 GPU", "gpu"),
    ("p3_npu", "p=3 NPU", "npu"),
    ("oracle", "Oracle", "oracle"),
]
P3_FRAMEWORK_TO_METHOD = {
    "llama_cpp_cpu_p3_same_u": "p3_cpu",
    "llama_cpp_gpu_p3_same_u": "p3_gpu",
    "llama_cpp_npu_p3_same_u": "p3_npu",
}


@dataclass(frozen=True)
class Config:
    oracle_by_workflow_csv: Path = (
        WFGEN_ROOT
        / "figure_data"
        / "fig6-qwen1_5_1_8b_oracle_20260628"
        / "execution_latency_qwen15_by_workflow.csv"
    )
    p3_summary_csv: Path = (
        WFGEN_ROOT
        / "results"
        / "odabench_qwen15_p3_same_u_full_20260628"
        / "summary.csv"
    )
    figure_data_dir: Path = (
        WFGEN_ROOT
        / "figure_data"
        / "workflow_latency_p3_same_u_all_vs_oracle_qwen15_20260628"
    )
    output_stem: Path = (
        REPO_ROOT
        / "figures"
        / "motivation"
        / "offloading"
        / "offloading_policy_qwen15_p3_same_u"
    )


CONFIG = Config()
SELECTED10_SNAPSHOT = REPO_ROOT / "data/offloading_latency.json"
SELECTED10_METHODS = METHODS[:3] + [("odabench", "Oracle", "fig6_odabench")]
EVAL_FIGSIZE = (3.00, 2.05)
EVAL_FONT = {
    "base": 8.2,
    "title": 8.4,
    "label": 8.2,
    "tick": 7.0,
    "legend": 6.8,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float:
    return float(value) if value else math.nan


def wall_s(row: dict[str, str]) -> float:
    return float(row["wall_ms"]) / 1000.0


def failure_marker(row: dict[str, str]) -> str:
    return "ERR"


def method_label(method: str) -> str:
    return next(label for key, label, _ in METHODS if key == method)


def build_records() -> list[dict[str, Any]]:
    oracle_rows = read_rows(CONFIG.oracle_by_workflow_csv)
    p3_rows = read_rows(CONFIG.p3_summary_csv)
    p3_by_key = {
        (row["device"], int(row["workflow_index"]), P3_FRAMEWORK_TO_METHOD[row["framework_id"]]): row
        for row in p3_rows
        if row.get("model_id") == MODEL_ID and row.get("framework_id") in P3_FRAMEWORK_TO_METHOD
    }

    records: list[dict[str, Any]] = []
    for oracle in oracle_rows:
        workflow_index = int(oracle["workflow_index"])
        for device in DEVICE_ORDER:
            records.append(
                {
                    "workflow_index": workflow_index,
                    "workflow_id": oracle["workflow_id"],
                    "workflow_label": oracle["workflow_label"],
                    "device": device,
                    "device_label": DEVICE_LABELS[device],
                    "model_id": MODEL_ID,
                    "model_label": MODEL_LABEL,
                    "method": "oracle",
                    "method_label": "Oracle",
                    "latency_s": as_float(oracle[f"oracle_{device}_s"]),
                    "partial_wall_s": as_float(oracle[f"oracle_{device}_s"]),
                    "status": "completed",
                    "marker": "",
                    "failure_reason": "",
                    "placements": "",
                    "llm_failed_count": 0,
                    "llm_node_count": "",
                }
            )
            for method in ("p3_cpu", "p3_gpu", "p3_npu"):
                p3 = p3_by_key[(device, workflow_index, method)]
                status = p3["status"]
                records.append(
                    {
                        "workflow_index": workflow_index,
                        "workflow_id": oracle["workflow_id"],
                        "workflow_label": oracle["workflow_label"],
                        "device": device,
                        "device_label": DEVICE_LABELS[device],
                        "model_id": MODEL_ID,
                        "model_label": MODEL_LABEL,
                        "method": method,
                        "method_label": method_label(method),
                        "latency_s": wall_s(p3) if status == "completed" else math.nan,
                        "partial_wall_s": wall_s(p3),
                        "status": status,
                        "marker": "" if status == "completed" else failure_marker(p3),
                        "failure_reason": p3.get("failure_reason", ""),
                        "placements": p3.get("placements", ""),
                        "llm_failed_count": int(p3.get("llm_failed_count") or 0),
                        "llm_node_count": int(p3.get("llm_node_count") or 0),
                    }
                )
    return records


def write_figure_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    CONFIG.figure_data_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = CONFIG.figure_data_dir / "workflow_latency_p3_same_u_all_vs_oracle_qwen15.csv"
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
        "failure_reason",
        "placements",
        "llm_failed_count",
        "llm_node_count",
    ]
    with rows_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            if isinstance(row["latency_s"], float) and math.isnan(row["latency_s"]):
                row["latency_s"] = ""
            writer.writerow(row)

    totals: list[dict[str, Any]] = []
    for device in DEVICE_ORDER:
        for method, label, _ in METHODS:
            subset = [
                record
                for record in records
                if record["device"] == device and record["method"] == method
            ]
            completed = [record for record in subset if record["status"] == "completed"]
            failed = [record for record in subset if record["status"] != "completed"]
            totals.append(
                {
                    "device": device,
                    "device_label": DEVICE_LABELS[device],
                    "method": method,
                    "method_label": label,
                    "completed_count": len(completed),
                    "failed_count": len(failed),
                    "completed_total_s": sum(record["latency_s"] for record in completed),
                    "partial_wall_failed_total_s": sum(record["partial_wall_s"] for record in failed),
                    "failed_workflows": [
                        {
                            "workflow_index": record["workflow_index"],
                            "workflow_label": record["workflow_label"],
                            "marker": record["marker"],
                            "partial_wall_s": record["partial_wall_s"],
                            "failure_reason": record["failure_reason"],
                        }
                        for record in failed
                    ],
                }
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "metric": "complete_workflow_latency_s",
        "workflow_count": len({record["workflow_id"] for record in records}),
        "source_oracle_by_workflow_csv": str(CONFIG.oracle_by_workflow_csv),
        "source_p3_summary_csv": str(CONFIG.p3_summary_csv),
        "rows_csv": str(rows_csv),
        "totals": totals,
        "note": (
            "latency_s is blank for failed workflows. partial_wall_s records the "
            "time spent before timeout/error and is not plotted as complete latency."
        ),
    }
    summary_path = CONFIG.figure_data_dir / "workflow_latency_p3_same_u_all_vs_oracle_qwen15_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_lines = [
        "# p=3 Same-U Full-Workflow Preview",
        "",
        "Standalone preview data comparing Oracle against p=3 same-U CPU/GPU/NPU runs.",
        "",
        f"- Oracle source: `{CONFIG.oracle_by_workflow_csv}`",
        f"- p=3 same-U source: `{CONFIG.p3_summary_csv}`",
        f"- Rows CSV: `{rows_csv}`",
        "",
        "| Device | Method | Completed | Completed total (s) | Failed |",
        "|---|---|---:|---:|---:|",
    ]
    for row in totals:
        readme_lines.append(
            f"| {row['device_label']} | {row['method_label']} | "
            f"{row['completed_count']}/10 | {row['completed_total_s']:.1f} | "
            f"{row['failed_count']} |"
        )
    readme_lines.extend(
        [
            "",
            "`ERR` marks any failed workflow, including 360 s timeouts and runtime errors such as Hexagon DSP queue abort.",
            "",
        ]
    )
    (CONFIG.figure_data_dir / "README.md").write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )
    return summary


def full_box_axes(ax: Any) -> None:
    figure_style.box_axes(ax, grid_axis="y")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(figure_style.STYLE["axes_linewidth"])
        ax.spines[side].set_color(figure_style.BASELINE)


def plot(
    records: list[dict[str, Any]], preview_dir: Path | None,
    methods: list[tuple[str, str, str]] | None = None,
) -> None:
    methods = METHODS if methods is None else methods
    figure_style.configure_matplotlib(plt)
    plt.rcParams.update(
        {
            "font.size": EVAL_FONT["base"],
            "axes.titlesize": EVAL_FONT["title"],
            "axes.labelsize": EVAL_FONT["label"],
            "xtick.labelsize": EVAL_FONT["tick"],
            "ytick.labelsize": EVAL_FONT["tick"],
            "legend.fontsize": EVAL_FONT["legend"],
        }
    )
    colors = {
        "oracle": figure_style.bar_color("oracle"),
        "odabench": figure_style.bar_color("fig6_odabench"),
        "p3_cpu": figure_style.bar_color("p3_cpu"),
        "p3_gpu": figure_style.bar_color("p3_gpu"),
        "p3_npu": figure_style.bar_color("p3_npu"),
        "fail": figure_style.DESIGN_COLORS["red"],
        "white": figure_style.BASE_COLORS["white"],
    }
    by_device = {
        device: [record for record in records if record["device"] == device]
        for device in DEVICE_ORDER
    }
    workflow_labels = [
        record["workflow_label"]
        for record in records
        if record["device"] == DEVICE_ORDER[0] and record["method"] == methods[-1][0]
    ]
    x = list(range(len(workflow_labels)))
    bar_w = 0.17
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]
    completed_values = [
        record["latency_s"]
        for record in records
        if record["status"] == "completed" and not math.isnan(record["latency_s"])
    ]
    ymax = max(completed_values) * 1.18
    marker_y = ymax * 0.055

    fig, ax = plt.subplots(1, 1, figsize=EVAL_FIGSIZE)
    axes = [ax]
    fig.subplots_adjust(left=0.16, right=0.995, bottom=0.22, top=0.73)

    for ax, device in zip(axes, DEVICE_ORDER):
        device_records = by_device[device]
        for offset, (method, label, _) in zip(offsets, methods):
            method_rows = [record for record in device_records if record["method"] == method]
            values = [record["latency_s"] for record in method_rows]
            ax.bar(
                [pos + offset for pos in x],
                values,
                width=bar_w,
                color=colors[method],
                edgecolor=figure_style.BAR_EDGE,
                linewidth=figure_style.BAR_EDGE_WIDTH,
                label=label,
                zorder=2,
            )
            for pos, record in zip(x, method_rows):
                if record["status"] == "completed":
                    continue
                ax.scatter(
                    pos + offset,
                    marker_y,
                    marker="X",
                    s=14,
                    facecolor=colors["fail"],
                    edgecolor=colors["white"],
                    linewidth=0.4,
                    zorder=4,
                )
        ax.set_title(DEVICE_LABELS[device], pad=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(workflow_labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Workflow")
        ax.set_xlim(-0.58, len(workflow_labels) - 0.42)
        ax.set_ylim(0, ymax)
        full_box_axes(ax)
        ax.tick_params(axis="both", length=2.5, labelbottom=True)
        ax.text(
            0.030,
            0.915,
            "   ERR: failed",
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=figure_style.INK,
            fontsize=EVAL_FONT["legend"],
            bbox={
                "boxstyle": "square,pad=0.16",
                "facecolor": colors["white"],
                "edgecolor": figure_style.NEUTRAL,
                "linewidth": 0.45,
            },
            zorder=6,
        )
        ax.scatter(
            [0.052],
            [0.915],
            transform=ax.transAxes,
            marker="X",
            s=20,
            facecolor=colors["fail"],
            edgecolor=colors["white"],
            linewidth=0.35,
            zorder=7,
        )
    for ax in axes:
        ax.set_ylabel("Latency (s)")

    handles: list[Any] = [
        Patch(facecolor=colors[method], edgecolor=figure_style.BAR_EDGE,
              linewidth=figure_style.BAR_EDGE_WIDTH, label=label)
        for method, label, _ in methods
    ]
    # Matplotlib fills legend columns first; retain the CPU/GPU/NPU/Oracle row order.
    handles = handles[::2] + handles[1::2]
    figure_style.boxed_legend_above_axes(
        fig,
        handles,
        [handle.get_label() for handle in handles],
        ncol=2,
        columnspacing=0.7,
        handlelength=0.72,
        handletextpad=0.24,
        prop={"size": EVAL_FONT["legend"]},
    )

    figure_style.save_pdf_svg(
        fig,
        CONFIG.output_stem,
        exact_canvas=True,
        creator="Paper_MOBench p=3 same-U all workflow preview",
        preview_dir=preview_dir,
    )
    plt.close(fig)
    strip_trailing_whitespace(CONFIG.output_stem.with_suffix(".svg"))


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path, default=None)
    args = parser.parse_args()

    if SELECTED10_SNAPSHOT.exists():
        snapshot = json.loads(SELECTED10_SNAPSHOT.read_text(encoding="utf-8"))
        records = [dict(row, latency_s=math.nan if row["latency_s"] is None
                        else row["latency_s"]) for row in snapshot["latency_records"]]
        plot(records, args.preview_dir, SELECTED10_METHODS)
        print(f"Wrote {CONFIG.output_stem.with_suffix('.pdf')} from the frozen selected-ten source")
        return

    records = build_records()
    summary = write_figure_data(records)
    plot(records, args.preview_dir)
    for row in summary["totals"]:
        print(
            f"{row['device_label']} {row['method_label']}: "
            f"{row['completed_count']}/10 completed, "
            f"{row['completed_total_s']:.1f}s completed total, "
            f"{row['failed_count']} failed"
        )
    print(f"Wrote {CONFIG.output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
