"""Generate the column-sized Fig. 6 resource timeline subfigure."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "odabench_fig6_resource_timeline_column"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import figure_style


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "figures" / "motivation" / "offloading"
OUTPUT_BASE = FIGURE_DIR / "offloading_policy_qwen15_resource_timeline_8gen3_column"
OUTPUT_CSV = FIGURE_DIR / "offloading_policy_qwen15_resource_timeline_8gen3_timeseries.csv"
OUTPUT_SUMMARY = FIGURE_DIR / "offloading_policy_qwen15_resource_timeline_8gen3_summary.json"
EVAL_FIGSIZE = (2.30, 2.05)
EVAL_FONT = {
    "base": 8.2,
    "title": 8.4,
    "label": 8.2,
    "tick": 7.5,
    "legend": 6.8,
}

DEVICE = "gen3"
DEVICE_LABEL = "8 Gen 3"
MODEL_ID = "qwen1_5_1_8b_chat_q4_0"
MODEL_LABEL = "Qwen1.5-1.8B-Q4"
WORKFLOW_INDEX = 179
WORKFLOW_LABEL = "R179"
WORKFLOW_ID = "tau_bench_cloud_retail_179_train_179"
DEFAULT_RUN = REPO_ROOT / "local_data" / "selected10_paper_supplement_parallel_20260930" / "combined"
CPU_COUNTER_INTERVAL_LIMIT_S = 30.0


@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    label: str
    framework: str
    color: str
    linestyle: str


@dataclass(frozen=True)
class Sample:
    elapsed_s: float
    cpu_pct: float
    gpu_pct: float
    mem_gb: float


@dataclass(frozen=True)
class RunSeries:
    repeat: str
    method: MethodConfig
    status: str
    wall_s: float
    placements: str
    run_dir: Path
    resource_samples_csv: Path
    perfetto_trace: Path
    samples: list[Sample]
    result_json: Path
    started_epoch_s: float
    finished_epoch_s: float
    max_interpolation_gap_s: float


METHODS = [
    MethodConfig(
        "cpu_p3_same_u",
        "CPU p=3 (ERR 0/3)",
        "llama_cpp_cpu_p3_same_u",
        figure_style.bar_color("p3_cpu"),
        "-",
    ),
    MethodConfig(
        "gpu_p3_same_u",
        "GPU p=3",
        "llama_cpp_gpu_p3_same_u",
        figure_style.bar_color("p3_gpu"),
        "-",
    ),
    MethodConfig(
        "odabench",
        "Oracle",
        "llama_cpp_max_wave_p4",
        figure_style.bar_color("fig6_odabench"),
        "-",
    ),
]
METHOD_BY_ID = {method.method_id: method for method in METHODS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Audit local measurements and replace the frozen source snapshot.")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    if args.refresh:
        timeline = build_repeated_timeline(args.run.resolve())
    else:
        timeline = load_frozen_timeline(REPO_ROOT / "data/offloading_resource.json")
    # Preserve the original point-sample aggregates and record the separate
    # counter-interval representation used by the corrected CPU-only curve.
    timeline["methods"]["cpu_p3_same_u"]["display_metrics"] = {
        "cpu_pct": cpu_counter_interval_means(timeline["methods"]["cpu_p3_same_u"]["runs"]),
    }
    timeline["display_protocol"] = {
        "cpu_only_cpu_curve": "Three-repeat means and sample standard deviations of observed CPU counter-interval averages, rendered as steps; original samples and point-sample aggregates are preserved.",
        "gpu_curves": "Original jointly valid three-repeat samples. GPU 0/0 counters remain unavailable; internal missing samples remain gaps.",
    }
    timeline["display_window"] = common_display_window(timeline)
    write_summary(timeline, OUTPUT_SUMMARY)
    write_timeseries_csv(timeline, OUTPUT_CSV)
    plot(timeline, OUTPUT_BASE)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        (args.preview_dir / OUTPUT_BASE.with_suffix(".png").name).write_bytes(OUTPUT_BASE.with_suffix(".png").read_bytes())
    print(f"Wrote {OUTPUT_BASE.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.svg')}")
    print(f"Wrote {OUTPUT_BASE.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY}")


def relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deployment_signature(path: Path) -> dict[str, Any]:
    runtime = read_json(path)
    fields = ("model_path", "max_tokens", "temperature", "llama_timeout", "threads", "json_schema", "warmup_runs", "llama_device", "ngl", "llama_cli_target", "device_serial")
    llm = runtime["llm"]
    return {"max_parallel_nodes": runtime["max_parallel_nodes"], "runtime": llm["runtime"],
            "base": {key: llm["llama_cli"].get(key) for key in fields},
            "overrides": {node: {key: config.get(key) for key in fields} for node, config in llm["llama_cli_overrides"].items()}}


def build_repeated_timeline(run: Path = DEFAULT_RUN) -> dict[str, Any]:
    analysis = read_json(run / "analysis" / "summary.json")
    manifest = read_json(run / "manifest.json")
    audit = read_json(run / "final_validation.json")
    assert audit["passed"]
    rows = read_json(run / "summary.json")["rows"] + [item["row"] for item in manifest["reused"]]
    winner = next(item["winner"] for item in analysis["odabench_selection"] if item["device"] == DEVICE and item["model_id"] == MODEL_ID and item["source_index"] == WORKFLOW_INDEX)
    assert winner["success"] and winner["configuration"] == "max_wave" and winner["max_parallel_nodes"] == 4
    assert manifest["resource_selection"]["source"]["result_json"] == winner["result_json"]
    winner_path = Path(winner["result_json"])
    assert sha256(winner_path) == manifest["resource_selection"]["source_sha256"]
    # The combined transport file covers new rows, while this selected row is
    # historical reuse. Preserve its USB provenance from the actual commands.
    winner_raw = read_json(winner_path)
    assert read_json(winner_path.parent / "monitor_start.json")["adb_serial"] == DEVICE
    for step in winner_raw["execution"]["steps"]:
        command = step["raw"]["command"]
        assert command[command.index("-s") + 1] == DEVICE
    transport_rows = [item for item in read_json(run / "transport_provenance.json")["rows"] if item["result_json"] == str(winner_path)]
    if not transport_rows:
        transport_rows.append({"device": DEVICE, "source_index": WORKFLOW_INDEX, "transport": "usb_adb", "result_json": str(winner_path), "evidence": "All original execution commands use physical USB serial gen3; original monitor_start.json has the same serial. Reused rows are absent from the new-only combined transport file."})
    frozen_signature = deployment_signature(winner_path.parent / "179.runtime.json")
    workflow_sha = sha256(winner_path.parent / "179.workflow.json")
    source_sha: dict[str, str] = {}
    used_result_paths = {str(winner_path)}
    for path in [run / "analysis" / "summary.json", run / "manifest.json", run / "final_validation.json", run / "transport_provenance.json", winner_path, winner_path.parent / "179.runtime.json", winner_path.parent / "179.workflow.json", winner_path.parent / "monitor_start.json"]:
        source_sha[relative(path)] = sha256(path)
    methods: dict[str, Any] = {}
    for method in METHODS:
        role = {"cpu_p3_same_u": "cpu_p3", "gpu_p3_same_u": "gpu_p3", "odabench": "odabench"}[method.method_id]
        figure = "Fig.13" if role == "odabench" else "Fig.6"
        cases = sorted([item for item in analysis["resource_runs"] if item["figure"] == figure and item["role"] == role], key=lambda item: item["repeat"])
        assert [item["repeat"] for item in cases] == [1, 2, 3]
        runs = []
        for case in cases:
            row = next(item for item in rows if item["device"] == DEVICE and item["config"].endswith(f"{figure.replace('.', '').replace(' ', '')}_{role}_repeat{case['repeat']}"))
            path = Path(row["result_json"])
            used_result_paths.add(str(path))
            assert case["source_index"] == WORKFLOW_INDEX and row["model_id"] == MODEL_ID
            assert row["success"] == case["success"]
            assert sha256(path) == audit["all_original_sha256"][str(path)]
            assert sha256(path.parent / "179.workflow.json") == workflow_sha
            if role == "odabench":
                assert deployment_signature(path.parent / "179.runtime.json") == frozen_signature
                assert read_json(path)["execution"]["started_at_utc"] > read_json(winner_path)["execution"]["finished_at_utc"]
            for source_path in [path, path.parent / "179.workflow.json", path.parent / "179.runtime.json", Path(case["resource_samples"]), Path(case["resource_by_workflow"]), path.parent / "trace_archive.json", Path(case["resource_samples"]).parent / "wfgen_resource_trace.perfetto-trace"]:
                source_sha[relative(source_path)] = sha256(source_path)
            runs.append(load_run_series(method, f"r{case['repeat']}", path, Path(case["resource_samples"])))
        if len(runs) != 3:
            raise ValueError(f"Expected exactly 3 repeats for {method.method_id}, found {len(runs)}")
        successes = sum(item.status == "completed" for item in runs)
        methods[method.method_id] = {
            "config": method,
            "label": method.label,
            "status": "completed" if all(run.status == "completed" for run in runs) else "failed",
            "repeat_count": len(runs),
            "successes": successes,
            "expected": 3,
            "configuration": "max_wave" if role == "odabench" else role,
            "max_parallel_nodes": 4 if role == "odabench" else 3,
            "wall_s_mean": finite_mean(run.wall_s for run in runs) if successes == 3 else None,
            "partial_wall_s_mean": finite_mean(run.wall_s for run in runs),
            "wall_s_std": finite_std(run.wall_s for run in runs) if successes == 3 else None,
            "partial_wall_s_std": finite_std(run.wall_s for run in runs),
            "wall_s_min": min(run.wall_s for run in runs),
            "wall_s_max": max(run.wall_s for run in runs),
            "placements": first_nonempty(run.placements for run in runs),
            "runs": runs,
            "metrics": {
                "cpu_pct": resample_runs(runs, "cpu_pct", clamp_percent=True),
                "gpu_pct": resample_runs(runs, "gpu_pct", clamp_percent=True),
                "mem_gb": resample_runs(runs, "mem_gb", clamp_percent=False),
            },
        }
    return {
        "mode": "repeated_workflow",
        "repeat_stat": "mean +/- 1 std",
        "device": DEVICE,
        "device_label": DEVICE_LABEL,
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "workflow_index": WORKFLOW_INDEX,
        "workflow_label": WORKFLOW_LABEL,
        "workflow_id": WORKFLOW_ID,
        "schema_version": 2,
        "dataset": relative(run),
        "source_sha256": source_sha,
        "transport_provenance": transport_rows + [item for item in read_json(run / "transport_provenance.json")["rows"] if item["result_json"] in used_result_paths and item["result_json"] != str(winner_path)],
        "workflow_sha256": workflow_sha,
        "selection": {"policy": "Same evaluation MOBench offline fastest successful candidate among naive, wave2, wave3, max_rr, max_wave. Configuration frozen before the three fresh resource repeats; no fastest-repeat selection.", "winner": winner, "configuration": "max_wave", "parallel": 4, "selection_wall_s": winner["wall_ms"] / 1000},
        "methods": methods,
        "note": (
            "R179 on Gen3 Qwen1.5: CPU p=3 is ERR 0/3 and its curves show partial failed attempts, with no complete latency. GPU p=3 and frozen MOBench max_wave p=4 are each 3/3. Mean and sample standard deviation (ddof=1) are shown only where all three repeats have compatible observed support. Linear interpolation is confined to adjacent valid samples, with no extrapolation, missing-row bridging, or conversion of GPU 0/0 counters to zero. Memory is exported but the original two CPU/GPU panels are preserved. Host timing includes ADB overhead. The historical selected row and fresh resource repeats both use USB on different dates; separate provenance is retained without claiming an identical complete environment."
        ),
    }


def load_run_series(method: MethodConfig, repeat: str, result_path: Path, samples_csv: Path) -> RunSeries:
    raw = read_json(result_path)
    run = raw["execution"]
    resource_dir = samples_csv.parent
    perfetto_trace = resource_dir / "wfgen_resource_trace.perfetto-trace"
    started_s = datetime.fromisoformat(run["started_at_utc"].replace("Z", "+00:00")).timestamp()
    finished_s = datetime.fromisoformat(run["finished_at_utc"].replace("Z", "+00:00")).timestamp()
    samples = read_samples(samples_csv, started_s, finished_s)
    gaps = [right.elapsed_s - left.elapsed_s for left, right in zip(samples, samples[1:]) if right.elapsed_s > left.elapsed_s]
    max_gap = 2 * float(np.median(gaps)) if gaps else 0.0
    config = read_json(result_path.parent / "179.runtime.json")
    counts: dict[str, int] = {}
    for item in config["llm"]["llama_cli_overrides"].values():
        backend = "CPU" if item["ngl"] == 0 else str(item["llama_device"])
        counts[backend] = counts.get(backend, 0) + 1
    return RunSeries(
        repeat=repeat,
        method=method,
        status=run.get("status") or "",
        wall_s=float(run.get("total_wall_ms") or 0.0) / 1000.0,
        placements=",".join(f"{key}:{value}" for key, value in sorted(counts.items())),
        run_dir=result_path.parent,
        resource_samples_csv=samples_csv,
        perfetto_trace=perfetto_trace,
        samples=samples,
        result_json=result_path,
        started_epoch_s=started_s,
        finished_epoch_s=finished_s,
        max_interpolation_gap_s=max_gap,
    )


def read_samples(path: Path, started_s: float, finished_s: float) -> list[Sample]:
    rows = read_csv_rows(path)
    rows.sort(key=lambda row: parse_float(row.get("timestamp_epoch_s"), math.inf))
    samples: list[Sample] = []
    for row in rows:
        timestamp = parse_float(row.get("timestamp_epoch_s"))
        if math.isnan(timestamp) or timestamp < started_s or timestamp > finished_s:
            continue
        mem_mb = parse_float(row.get("mem_used_mb"))
        samples.append(
            Sample(
                elapsed_s=timestamp - started_s,
                cpu_pct=clip_percent(parse_float(row.get("cpu_util_percent"))),
                gpu_pct=parse_gpu(row),
                mem_gb=mem_mb / 1024.0 if not math.isnan(mem_mb) else math.nan,
            )
        )
    return samples


def resample_runs(
    runs: list[RunSeries],
    field: str,
    *,
    clamp_percent: bool,
) -> dict[str, list[float] | list[int]]:
    max_duration = max(run.wall_s for run in runs)
    x_grid = np.linspace(0.0, max_duration, 160)
    interpolated: list[np.ndarray] = []
    for run in runs:
        values = np.full_like(x_grid, np.nan, dtype=float)
        # Keep invalid samples in the sequence: dropping them would join the
        # measurements on either side of a missing counter or failed poll.
        for left, right in zip(run.samples, run.samples[1:]):
            y_left, y_right = getattr(left, field), getattr(right, field)
            gap = right.elapsed_s - left.elapsed_s
            if not (math.isfinite(y_left) and math.isfinite(y_right)) or not 0 < gap <= run.max_interpolation_gap_s:
                continue
            inside = (x_grid >= left.elapsed_s) & (x_grid <= right.elapsed_s)
            values[inside] = np.interp(x_grid[inside], [left.elapsed_s, right.elapsed_s], [y_left, y_right])
        interpolated.append(values)

    stack = np.vstack(interpolated)
    means: list[float] = []
    stds: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    counts: list[int] = []
    for column in stack.T:
        mean, std, lower, upper, count = finite_column_stats(column)
        if count != 3:
            mean = std = lower = upper = math.nan
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
        "expected_sample_count": 3,
        "mean_requires_all_repeats": True,
    }


def cpu_counter_interval_means(runs: list[RunSeries]) -> dict[str, Any]:
    """Use measured /proc/stat deltas as interval means, not point samples.

    The acquisition code retains its previous cumulative CPU counters when a
    poll times out. The next finite CPU value therefore averages the entire
    interval since the preceding successful counter read, including that gap.
    Host poll-start timestamps approximate the counter-read boundaries. This
    does not recover instantaneous CPU values or extrapolate past observations.
    """
    x_grid = np.linspace(0.0, max(run.wall_s for run in runs), 160)
    interval_values: list[np.ndarray] = []
    for run in runs:
        values = np.full_like(x_grid, np.nan, dtype=float)
        valid = [sample for sample in run.samples if math.isfinite(sample.cpu_pct)]
        for left, right in zip(valid, valid[1:]):
            duration = right.elapsed_s - left.elapsed_s
            if not 0 < duration <= CPU_COUNTER_INTERVAL_LIMIT_S:
                continue
            inside = (x_grid >= left.elapsed_s) & (x_grid < right.elapsed_s)
            values[inside] = right.cpu_pct
        interval_values.append(values)
    stack = np.vstack(interval_values)
    means, stds, lowers, uppers, counts = [], [], [], [], []
    for column in stack.T:
        mean, std, lower, upper, count = finite_column_stats(column)
        if count != 3:
            mean = std = lower = upper = math.nan
        means.append(mean)
        stds.append(std)
        lowers.append(clip_percent(lower))
        uppers.append(clip_percent(upper))
        counts.append(count)
    return {
        "elapsed_s": [float(value) for value in x_grid],
        "mean": means, "std": stds, "lower": lowers, "upper": uppers,
        "sample_count": counts, "expected_sample_count": 3,
        "mean_requires_all_repeats": True,
        "aggregation": "Observed CPU cumulative-counter interval means; right-hand reading covers the preceding successful-read interval.",
        "max_interval_s": CPU_COUNTER_INTERVAL_LIMIT_S,
        "timestamp_basis": "Host poll-start times approximate the CPU counter-read boundaries.",
        "extrapolation": False,
    }


def plotted_metric(method_data: dict[str, Any], metric: str) -> dict[str, Any]:
    return method_data.get("display_metrics", {}).get(metric, method_data["metrics"][metric])


def crop_metric(values: dict[str, Any], start_s: float, *, steps: bool = False) -> dict[str, Any]:
    """Crop a display series, evaluating the boundary only within valid support."""
    xs = values["elapsed_s"]
    index = int(np.searchsorted(xs, start_s))
    result = dict(values)
    columns = ("elapsed_s", "mean", "std", "lower", "upper", "sample_count")
    for key in columns:
        result[key] = values[key][index:]
    if index < len(xs) and xs[index] == start_s:
        return result
    # Keep missing boundaries missing; never bridge a gap or extrapolate.
    boundary = {key: math.nan for key in ("mean", "std", "lower", "upper")}
    count = 0
    if 0 < index < len(xs):
        left, right = index - 1, index
        if math.isfinite(values["mean"][left]) and math.isfinite(values["mean"][right]):
            fraction = (start_s - xs[left]) / (xs[right] - xs[left])
            for key in boundary:
                boundary[key] = values[key][left] if steps else (
                    values[key][left] + fraction * (values[key][right] - values[key][left])
                )
            count = min(values["sample_count"][left], values["sample_count"][right])
    result["elapsed_s"].insert(0, start_s)
    result["sample_count"].insert(0, count)
    for key, value in boundary.items():
        result[key].insert(0, value)
    return result


def common_display_window(timeline: dict[str, Any]) -> dict[str, Any]:
    starts = {}
    unavailable = []
    for method in METHODS:
        for metric in ("cpu_pct", "gpu_pct"):
            values = plotted_metric(timeline["methods"][method.method_id], metric)
            valid = [x for x, y in zip(values["elapsed_s"], values["mean"]) if math.isfinite(y)]
            key = f"{method.method_id}/{metric}"
            if valid:
                starts[key] = valid[0]
            else:
                unavailable.append(key)
    start_s = max(starts.values())
    for key in starts:
        method_id, metric = key.split("/")
        values = crop_metric(
            plotted_metric(timeline["methods"][method_id], metric), start_s,
            steps=method_id == "cpu_p3_same_u" and metric == "cpu_pct",
        )
        if not math.isfinite(values["mean"][0]) or values["sample_count"][0] != 3:
            raise ValueError(f"Shared start has no three-repeat support for {key}")
    return {
        "start_elapsed_s": start_s,
        "axis_origin_elapsed_s": start_s,
        "axis_label": "Time from shared start (s)",
        "first_valid_elapsed_s": starts,
        "unavailable_mean_series": unavailable,
        "policy": "Crop both panels at the latest first valid mean among available series, then subtract this identical workflow-relative time from every method. CPU-only GPU has no three-repeat mean and remains unavailable. Internal gaps and original workflow times are preserved.",
        "boundary_evaluation": "Existing point at the boundary, or adjacent finite interpolation for point series and previous interval value for CPU steps; no extrapolation or gap bridging.",
    }


def chart_metric(timeline: dict[str, Any], method_id: str, metric: str) -> dict[str, Any]:
    return crop_metric(
        plotted_metric(timeline["methods"][method_id], metric),
        timeline["display_window"]["start_elapsed_s"],
        steps=method_id == "cpu_p3_same_u" and metric == "cpu_pct",
    )


def plot(timeline: dict[str, Any], output_base: Path) -> None:
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

    fig, axes = plt.subplots(2, 1, figsize=EVAL_FIGSIZE, sharex=True)
    fig.subplots_adjust(left=0.22, right=0.965, bottom=0.22, top=0.73, hspace=0.20)
    panels = [
        ("CPU (%)", "cpu_pct", (-3, 103)),
        ("GPU (%)", "gpu_pct", (-3, 103)),
    ]
    start_s = timeline["display_window"]["start_elapsed_s"]

    for ax, (ylabel, metric, ylim) in zip(axes, panels):
        for method in METHODS:
            method_data = timeline["methods"][method.method_id]
            values = chart_metric(timeline, method.method_id, metric)
            chart_times = np.asarray(values["elapsed_s"]) - start_s
            counter_intervals = method.method_id == "cpu_p3_same_u" and metric == "cpu_pct"
            finite_mean = any(math.isfinite(value) for value in values["mean"])
            if finite_mean:
                ax.fill_between(
                    chart_times,
                    values["lower"],
                    values["upper"],
                    color=method.color,
                    alpha=0.15 if method_data["status"] == "completed" else 0.08,
                    linewidth=0,
                    zorder=2,
                    step="post" if counter_intervals else None,
                )
                ax.plot(
                    chart_times,
                    values["mean"],
                    color=method.color,
                    linestyle=method.linestyle,
                    linewidth=1.25 if method_data["status"] == "completed" else 1.05,
                    alpha=0.98 if method_data["status"] == "completed" else 0.7,
                    clip_on=True,
                    zorder=3,
                    drawstyle="steps-post" if counter_intervals else "default",
                )
            elif method_data["status"] != "completed":
                plot_failed_samples(ax, method_data["runs"], method, metric, start_s)
        ax.set_ylabel(ylabel, labelpad=1.4)
        figure_style.box_axes(ax, grid_axis="both")
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.set_yticks([0, 50, 100])
        else:
            values = finite_metric_values(timeline, metric)
            ax.set_ylim(max(0.0, min(values) * 0.94), max(values) * 1.04)
            ax.set_yticks([6, 8, 10])

    mark_err(axes, timeline)
    max_time = max(
        timeline["methods"][method.method_id]["wall_s_max"]
        for method in METHODS
    )
    axes[-1].set_xlim(0, (max_time - start_s) * 1.02)
    axes[-1].set_xlabel(timeline["display_window"]["axis_label"], labelpad=1.3)
    figure_style.boxed_legend_above_axes(
        fig,
        legend_handles(),
        [method.label for method in METHODS],
        ncol=1,
        columnspacing=0.50,
        handlelength=1.0,
        handletextpad=0.28,
        labelspacing=0.18,
        prop={"size": EVAL_FONT["legend"]},
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches=None, pad_inches=0.0, facecolor=figure_style.WHITE)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches=None, pad_inches=0.0, facecolor=figure_style.WHITE)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches=None, pad_inches=0.0, facecolor=figure_style.WHITE)
    plt.close(fig)
    strip_trailing_whitespace(output_base.with_suffix(".svg"))


def plot_failed_samples(ax: Any, runs: list[RunSeries], method: MethodConfig, metric: str, start_s: float) -> None:
    for run in runs:
        points = [
            (sample.elapsed_s - start_s, getattr(sample, metric))
            for sample in run.samples
            if sample.elapsed_s >= start_s and math.isfinite(getattr(sample, metric))
        ]
        if not points:
            continue
        ax.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color=method.color,
            s=6,
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )


def mark_err(axes: Any, timeline: dict[str, Any]) -> None:
    start_s = timeline["display_window"]["start_elapsed_s"]
    for method in METHODS:
        method_data = timeline["methods"][method.method_id]
        if method_data["status"] == "completed":
            continue
        err_x = method_data["partial_wall_s_mean"] - start_s
        err_min = method_data["wall_s_min"] - start_s
        err_max = method_data["wall_s_max"] - start_s
        for ax in axes:
            ax.axvspan(err_min, err_max, color=method.color, alpha=0.08, linewidth=0, zorder=1)
            ax.axvline(err_x, color=method.color, linewidth=0.7, linestyle=":", alpha=0.9)
        axes[0].scatter(
            [err_x],
            [100],
            marker="X",
            s=24,
            color=figure_style.DESIGN_COLORS["red"],
            edgecolors=figure_style.BASE_COLORS["white"],
            linewidths=0.35,
            zorder=7,
            clip_on=False,
        )
        axes[0].annotate(
            "ERR 0/3",
            xy=(err_x, 100),
            xytext=(-5, -9),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=6.0,
            fontweight="bold",
            color=figure_style.DESIGN_COLORS["red"],
            bbox={
                "boxstyle": "round,pad=0.13",
                "facecolor": figure_style.DESIGN_COLORS["red_bg"],
                "edgecolor": figure_style.DESIGN_COLORS["red"],
                "linewidth": 0.45,
            },
            zorder=8,
            clip_on=False,
        )


def write_timeseries_csv(timeline: dict[str, Any], path: Path) -> None:
    fields = [
        "method_id",
        "method_label",
        "status",
        "repeat_count",
        "wall_s_mean",
        "wall_s_std",
        "wall_s_min",
        "wall_s_max",
        "placements",
        "metric",
        "aggregation",
        "elapsed_s",
        "shared_start_elapsed_s",
        "chart_time_s",
        "mean",
        "std",
        "lower",
        "upper",
        "sample_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method_id, method in timeline["methods"].items():
            for metric in method["metrics"]:
                in_chart = metric in {"cpu_pct", "gpu_pct"}
                values = chart_metric(timeline, method_id, metric) if in_chart else plotted_metric(method, metric)
                for index, elapsed_s in enumerate(values["elapsed_s"]):
                    writer.writerow(
                        {
                            "method_id": method_id,
                            "method_label": method["label"],
                            "status": method["status"],
                            "repeat_count": method["repeat_count"],
                            "wall_s_mean": method["wall_s_mean"],
                            "wall_s_std": method["wall_s_std"],
                            "wall_s_min": method["wall_s_min"],
                            "wall_s_max": method["wall_s_max"],
                            "placements": method["placements"],
                            "metric": metric,
                            "aggregation": values.get("aggregation", "Adjacent valid sample interpolation; all three repeats required."),
                            "elapsed_s": elapsed_s,
                            "shared_start_elapsed_s": timeline["display_window"]["start_elapsed_s"],
                            "chart_time_s": elapsed_s - timeline["display_window"]["start_elapsed_s"] if in_chart else None,
                            "mean": values["mean"][index],
                            "std": values["std"][index],
                            "lower": values["lower"][index],
                            "upper": values["upper"][index],
                            "sample_count": values["sample_count"][index],
                        }
                    )


def write_summary(timeline: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(timeline), indent=2), encoding="utf-8")


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def finite_metric_values(timeline: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    for method_data in timeline["methods"].values():
        for bound in ("lower", "upper", "mean"):
            values.extend(
                value
                for value in method_data["metrics"][metric][bound]
                if math.isfinite(value)
            )
        for run in method_data["runs"]:
            values.extend(
                getattr(sample, metric)
                for sample in run.samples
                if math.isfinite(getattr(sample, metric))
            )
    return values or [0.0]


def legend_handles() -> list[Any]:
    return [
        Line2D(
            [0],
            [0],
            color=method.color,
            linestyle=method.linestyle,
            linewidth=1.2,
        )
        for method in METHODS
    ]


def parse_gpu(row: dict[str, str]) -> float:
    # A 0/0 counter is an unavailable observation, not measured idle time.
    if row.get("gpu_busy_counter") in {"0", "0.0"} and row.get("gpu_total_counter") in {"0", "0.0"}:
        return math.nan
    gpu = clip_percent(parse_float(row.get("gpu_busy_percent")))
    if not math.isnan(gpu):
        return gpu
    return math.nan


def extract_placements(run: dict[str, Any]) -> str:
    runtime_config = run.get("runtime_config")
    if runtime_config:
        path = Path(runtime_config)
        if path.exists():
            payload = read_json(path)
            deployments = payload.get("deployment", {}).get("node_deployments", [])
            counts: dict[str, int] = {}
            for node in deployments:
                if node.get("kind") != "llm":
                    continue
                backend = str(node.get("device_class") or node.get("device") or "unknown")
                counts[backend] = counts.get(backend, 0) + 1
            if counts:
                return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))
    result_json = run.get("result_json")
    if not result_json:
        return ""
    path = Path(result_json)
    if not path.exists():
        return ""
    payload = read_json(path)
    placements = payload.get("placement_summary") or payload.get("placements")
    if isinstance(placements, dict):
        return ",".join(f"{key}:{value}" for key, value in sorted(placements.items()))
    return str(placements or "")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clip_percent(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return max(0.0, min(100.0, value))


def finite_column_stats(values: np.ndarray) -> tuple[float, float, float, float, int]:
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    if count == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if count > 1 else 0.0
    return mean, std, mean - std, mean + std, count


def finite_mean(values: Any) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def finite_std(values: Any) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def first_nonempty(values: Any) -> str:
    for value in values:
        if value:
            return value
    return ""


def json_safe(value: Any) -> Any:
    if isinstance(value, MethodConfig):
        return {
            "method_id": value.method_id,
            "label": value.label,
            "framework": value.framework,
            "color": value.color,
            "linestyle": value.linestyle,
        }
    if isinstance(value, RunSeries):
        return {
            "repeat": value.repeat,
            "status": value.status,
            "wall_s": value.wall_s,
            "placements": value.placements,
            "run_dir": relative(value.run_dir),
            "resource_samples_csv": relative(value.resource_samples_csv),
            "perfetto_trace": relative(value.perfetto_trace),
            "sample_count": len(value.samples),
            "samples": [json_safe(sample) for sample in value.samples],
            "result_json": relative(value.result_json),
            "started_epoch_s": value.started_epoch_s,
            "finished_epoch_s": value.finished_epoch_s,
            "max_interpolation_gap_s": value.max_interpolation_gap_s,
        }
    if isinstance(value, Sample):
        return {key: json_safe(getattr(value, key)) for key in ("elapsed_s", "cpu_pct", "gpu_pct", "mem_gb")}
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items() if key != "config"}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_frozen_timeline(path: Path) -> dict[str, Any]:
    timeline = read_json(path)
    if timeline.get("schema_version") != 2:
        raise ValueError("Refresh once to create the current R179 frozen snapshot.")
    for method_id, data in timeline["methods"].items():
        method = METHOD_BY_ID[method_id]
        data["config"] = method
        for values in data["metrics"].values():
            for key in ("mean", "std", "lower", "upper"):
                values[key] = [math.nan if value is None else value for value in values[key]]
        for values in data.get("display_metrics", {}).values():
            for key in ("mean", "std", "lower", "upper"):
                values[key] = [math.nan if value is None else value for value in values[key]]
        runs = []
        for item in data["runs"]:
            samples = [Sample(**{key: math.nan if value is None else value for key, value in sample.items()}) for sample in item["samples"]]
            runs.append(RunSeries(repeat=item["repeat"], method=method, status=item["status"], wall_s=item["wall_s"], placements=item["placements"], run_dir=REPO_ROOT / item["run_dir"], resource_samples_csv=REPO_ROOT / item["resource_samples_csv"], perfetto_trace=REPO_ROOT / item["perfetto_trace"], samples=samples, result_json=REPO_ROOT / item["result_json"], started_epoch_s=item["started_epoch_s"], finished_epoch_s=item["finished_epoch_s"], max_interpolation_gap_s=item["max_interpolation_gap_s"]))
        data["runs"] = runs
    return timeline


if __name__ == "__main__":
    main()
