"""Combine saved 280-workflow static metrics and 10-workflow execution evidence.

No model or device calls. Empty/empty tool lists receive 1 for all-set macro
P/R/F1; empty references are excluded from the order denominator. Missing
execution evidence is N/A, never a success, a failure, or an extrapolated rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figure_style

ROOT = Path(__file__).resolve().parents[1]


def combine(static, business):
    rows = static["per_task"]
    ids = [r["selection_index"] for r in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Static sample IDs must be nonempty and unique")
    nonempty = [r for r in rows if r["reference_calls"]]
    # Explicit convention: both empty is a perfect static tool-list match.
    # An empty GT with extra predictions is zero, not a skipped sample.
    averages = {}
    for key in ("tool_precision", "tool_recall", "tool_f1"):
        values = [r[key] if r["reference_calls"] else float(r["actual_calls"] == 0) for r in rows]
        averages[key] = mean(values)
    averages["exact_tools"] = mean(r["tools_exact"] for r in rows)
    averages["exact_actions_arguments"] = mean(r["actions_and_parameters_match"] for r in rows)
    pairs = sum(r["strict_order"]["pair_count"] for r in nonempty)
    kept = sum(r["strict_order"]["preserved_pairs"] for r in nonempty)
    order = {
        "n": len(nonempty), "pairs": pairs, "preserved_pairs": kept,
        "pair_preservation": kept / pairs if pairs else None,
        "strict_order": mean(r["strict_order"]["strict_order_correct"] for r in nonempty) if nonempty else None,
        "tools_and_strict_order": mean(r["strict_complete"] for r in nonempty) if nonempty else None,
    }
    runs = business["runs"]
    keys = [(r["mode"], r["source_index"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate workflow/mode executions")
    tested = {r["source_index"] for r in runs}
    if not tested <= set(ids):
        raise ValueError("Execution records must belong to the static population")
    execution = {}
    for mode in ("serial", "parallel"):
        selected = [r for r in runs if r["mode"] == mode]
        count = len(selected)
        successes = sum(r["business_state_success"] for r in selected)
        execution[mode] = {"successes": successes, "n": count,
                           "rate": successes / count if count else None,
                           "full_population_rate": successes / len(ids) if count == len(ids) else None}
    full_rates = [execution[mode]["full_population_rate"] for mode in ("serial", "parallel")]
    return {
        "population_n": len(ids), "empty_n": len(rows) - len(nonempty),
        "static_macro_averages": averages, "strict_order_diagnostics": order,
        "business_state": execution, "executed_unique_workflows": len(tested),
        "full_population_business_state_mode_average": mean(full_rates) if all(r is not None for r in full_rates) else None,
        "unevaluated_workflows": len(ids) - len(tested),
        "definitions": {
            "static_average": "Per-workflow macro over all samples; both-empty tool lists score 1; empty GT with predicted tools scores 0.",
            "exact_actions_arguments": "Business-node list names, multiplicities, saved order and serialized arguments all match GT.",
            "strict_order": "All GT precedence pairs must have DAG paths; excludes empty tasks and penalizes even legal independent-read parallelism.",
            "business_state": "Observed fixed-action reference-state success only; missing executions are N/A, never counted as successes or failures.",
            "population_business_rate": "Not estimated unless the entire population has saved execution records in the same mode.",
        },
    }


def draw(metrics, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1.45, 1.05, 1.1]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.75, bottom=0.32, wspace=0.27)
    n = metrics["population_n"]
    observed = metrics["executed_unique_workflows"]
    missing = metrics["unevaluated_workflows"]
    fig.suptitle(f"Workflow accuracy overview — {n} tasks", x=0.05, y=0.98,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(0.05, 0.92, f"Static comparisons: {n} workflows  |  Saved business-state executions: {observed} unique workflows", fontsize=12, color="#555555")
    averages = metrics["static_macro_averages"]
    order = metrics["strict_order_diagnostics"]
    panels = [
        (axes[0], [averages[k] for k in ("tool_precision", "tool_recall", "tool_f1", "exact_tools", "exact_actions_arguments")],
         ["Tool\nprecision", "Tool\nrecall", "Tool F1", "Exact\ntools", "Actions +\narguments"],
         f"(a) Static agreement · n={n}", figure_style.DIAGNOSTIC_BLUE_DARK),
        (axes[1], [order[k] for k in ("pair_preservation", "strict_order", "tools_and_strict_order")],
         ["Ordered\npairs", "Strict\norder", "Tools +\nstrict order"],
         f"(b) Strict-order diagnostics · n={order['n']}", figure_style.BASELINE_2),
        (axes[2], [metrics["business_state"][m]["rate"] for m in ("serial", "parallel")] + [metrics["full_population_business_state_mode_average"]],
         [f"Serial\n(n={metrics['business_state']['serial']['n']})", f"Parallel\n(n={metrics['business_state']['parallel']['n']})", f"Full set\n(n={n})"],
         "(c) Observed business-state success", figure_style.COMPONENT_GREEN),
    ]
    for ax, values, labels, title, color in panels:
        for i, value in enumerate(values):
            if value is None:
                # A boxed missing-data indicator, not a zero-height/zero-rate bar.
                ax.add_patch(Rectangle((i - 0.32, 0), 0.64, 100, fill=False,
                                       edgecolor="#999999", linestyle="--", linewidth=1.2))
                ax.text(i, 52, "N/A\nnot measured", ha="center", va="center", fontsize=9, color="#666666")
            else:
                ax.bar(i, value * 100, width=0.62, color=color, zorder=3)
                ax.text(i, value * 100 + 2, f"{value * 100:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=12)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score (%)")
    fig.text(0.05, 0.23, f"Static averages include {metrics['empty_n']} empty-GT tasks: matching empty tool lists score 1. Empty tasks are excluded from order metrics.", fontsize=10)
    fig.text(0.05, 0.185, f"Strict-order diagnostics: {order['preserved_pairs']}/{order['pairs']} GT pairs preserved; even legal independent-read parallelism fails strict total-order matching.", fontsize=10)
    fig.text(0.05, 0.14, f"Business-state evidence covers {observed}/{n} workflows ({100 * observed / n:.1f}%); {missing} are unevaluated, not failed. No full-set success rate is inferred.", fontsize=10)
    fig.text(0.05, 0.095, "Business-state success uses saved final-state comparisons; workflow 179 uses a repaired execution plan. Static metrics use the original DAGs.", fontsize=10)
    fig.text(0.05, 0.05, "Final tool nodes/arguments are GT-derived. These are not independent planning accuracy or dialogue/refusal success. No new experiments were run.", fontsize=10)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"accuracy_overview_280.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-metrics", type=Path, default=ROOT / "local_data/fidelity_preview/expanded_metrics.json")
    parser.add_argument("--business-metrics", type=Path, default=ROOT / "local_data/business_success_preview/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "local_data/accuracy_overview")
    args = parser.parse_args()
    static_raw, business_raw = args.static_metrics.read_bytes(), args.business_metrics.read_bytes()
    metrics = combine(json.loads(static_raw), json.loads(business_raw))
    metrics["inputs"] = {str(p.resolve()): hashlib.sha256(raw).hexdigest()
                         for p, raw in ((args.static_metrics, static_raw), (args.business_metrics, business_raw))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(metrics, args.output_dir)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
