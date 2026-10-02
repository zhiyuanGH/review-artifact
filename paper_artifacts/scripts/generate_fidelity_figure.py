"""Plot expanded-workflow reference agreement and compact-plan diagnostics.

No model calls or workflow execution. The expanded figure compares business-node
lists and serialized arguments. The compact diagnostic uses the catalog saved
with each prediction; its repetition markers (*) reduce to type presence.
Strict order diagnostics test DAG reachability against the total GT order,
including independent reads; they do not verify semantic execution success.
Final workflows are GT-derived construction controls, not independent LLM
predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import figure_style

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANNING = ROOT / "examples/280_workflows"
DEFAULT_OUTPUT = ROOT / "local_data/fidelity_preview"
METHODS = ("llm_plan", "final_workflow")
LABELS = ("Saved LLM plan", "Final workflow (GT-derived)")


def strict_order_audit(workflow: dict, nodes: list[dict], actions: list[dict]) -> dict:
    """Check all GT precedence pairs; paths may pass through checker nodes.

    Duplicate tool calls are matched by tool and serialized arguments first,
    then by remaining occurrence order. No GT edges are inserted into the DAG.
    All GT pairs are required under this STRICT metric, even commuting reads.
    """
    node_ids = [n["node_id"] for n in workflow["nodes"]]
    errors = []
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node IDs")
    adjacency = {n: set() for n in node_ids}
    for edge in workflow.get("edges", []):
        a, b = edge["from_node"], edge["to_node"]
        if a not in adjacency or b not in adjacency:
            errors.append("unknown edge endpoint")
        else:
            adjacency[a].add(b)
    closure = {}
    for node_id in adjacency:
        reached, pending = set(), list(adjacency[node_id])
        while pending:
            current = pending.pop()
            if current in reached:
                continue
            reached.add(current)
            pending.extend(adjacency[current])
        closure[node_id] = reached
        if node_id in reached:
            errors.append("cycle")
    pools = defaultdict(list)
    for n in nodes:
        pools[n["entrypoint"]].append(n)
    matched = []
    for action in actions:
        candidates = pools[action["name"]]
        expected = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in action["arguments"].items()}
        choice = next((i for i, n in enumerate(candidates)
                       if {p["name"]: p["value"] for p in n["parameters"]} == expected), 0)
        matched.append(candidates.pop(choice)["node_id"] if candidates else None)
    pairs = []
    for i, source in enumerate(matched):
        for j in range(i + 1, len(matched)):
            target = matched[j]
            if errors:
                status = "invalid_graph"
            elif source is None or target is None:
                status = "missing_call"
            elif target in closure[source]:
                status = "preserved"
            elif source in closure[target]:
                status = "reversed"
            else:
                status = "unordered"
            pairs.append({"gt_before": i, "gt_after": j,
                          "from_node": source, "to_node": target, "status": status})
    return {"pair_count": len(pairs),
            "preserved_pairs": sum(p["status"] == "preserved" for p in pairs),
            "strict_order_correct": not errors and all(n is not None for n in matched)
                                    and all(p["status"] == "preserved" for p in pairs),
            "graph_errors": sorted(set(errors)),
            "unpreserved_pairs": [p for p in pairs if p["status"] != "preserved"]}


def expanded_audit(samples: dict, workflows: dict) -> dict:
    """Compare saved business-node order and wire-format arguments, including repeats.

    GT arguments are serialized to the documented WFGen representation: scalar
    strings and JSON strings for lists/dicts. No predicted value is filled in.
    Node-array order is not a claim about execution order enforced by DAG edges.
    """
    rows = []
    for idx, sample in sorted(samples.items()):
        workflow = workflows[idx]
        gt = sample["ground_truth_actions"]
        nodes = [n for n in workflow["nodes"] if n["kind"] in ("app", "dnn")]
        if workflow["metadata"]["task_id"] != sample["task_id"]:
            raise ValueError(f"Task identity mismatch: {idx}")
        names_match = [n["entrypoint"] for n in nodes] == [a["name"] for a in gt]
        predicted_counts = Counter(n["entrypoint"] for n in nodes)
        reference_counts = Counter(a["name"] for a in gt)
        overlap = sum((predicted_counts & reference_counts).values())
        tools_exact = predicted_counts == reference_counts
        order = strict_order_audit(workflow, nodes, gt)
        steps = []
        for node, action in zip(nodes, gt):
            supplied = {p["name"]: p["value"] for p in node["parameters"]}
            expected = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                        for k, v in action["arguments"].items()}
            match = (node["entrypoint"] == action["name"] and supplied == expected
                     and len(supplied) == len(node["parameters"]))
            steps.append({"node_id": node["node_id"], "match": match,
                          "expected_parameters": expected, "actual_parameters": supplied})
        rows.append({"selection_index": idx, "domain": sample["domain"],
                     "reference_calls": len(gt), "actual_calls": len(nodes),
                     "parameter_values": sum(len(a["arguments"]) for a in gt),
                     "tool_precision": overlap / len(nodes) if nodes else 0.0,
                     "tool_recall": overlap / len(gt) if gt else None,
                     "tool_f1": 2 * overlap / (len(nodes) + len(gt)) if gt else None,
                     "tools_exact": tools_exact,
                     "strict_order": order,
                     "strict_complete": tools_exact and order["strict_order_correct"],
                     "tool_sequence_match": names_match,
                     "actions_and_parameters_match": names_match and all(s["match"] for s in steps),
                     "steps": steps})
    nonempty = [r for r in rows if r["reference_calls"]]
    empty = [r for r in rows if not r["reference_calls"]]
    return {
        "scope": "Tool multiset accuracy and strict GT total-order preservation in original DAG; not semantic execution success",
        "order_definition": "Each earlier/later pair of GT actions must have a directed path from the matched earlier node to the later node, including checker paths. Legal independent-read parallelism also fails strict total-order matching.",
        "complete_definition": "Exact tool multiset AND strict GT order; parameters are reported separately.",
        "normalization": "GT scalars -> str(value); GT lists/dicts -> json.dumps(value), matching WFGen ABI. No tool aliases applied.",
        "nonempty_n": len(nonempty), "empty_n": len(empty),
        "tool_calls": sum(r["reference_calls"] for r in rows),
        "parameter_values": sum(r["parameter_values"] for r in rows),
        "tool_sequence_matches": sum(r["tool_sequence_match"] for r in nonempty),
        "actions_and_parameters_matches": sum(r["actions_and_parameters_match"] for r in nonempty),
        "empty_list_matches": sum(r["actual_calls"] == 0 for r in empty),
        "tool_precision_macro": mean(r["tool_precision"] for r in nonempty) if nonempty else None,
        "tool_recall_macro": mean(r["tool_recall"] for r in nonempty) if nonempty else None,
        "tool_f1_macro": mean(r["tool_f1"] for r in nonempty) if nonempty else None,
        "tools_exact_matches": sum(r["tools_exact"] for r in nonempty),
        "strict_order_matches": sum(r["strict_order"]["strict_order_correct"] for r in nonempty),
        "strict_complete_matches": sum(r["strict_complete"] for r in nonempty),
        "reference_order_pairs": sum(r["strict_order"]["pair_count"] for r in nonempty),
        "preserved_order_pairs": sum(r["strict_order"]["preserved_pairs"] for r in nonempty),
        "per_task": rows,
    }


def draw_expanded(audit: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.7))
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.28, top=0.77, wspace=0.25)
    fig.suptitle("Expanded workflow fidelity: tools and ordering", x=0.07, y=0.98,
                 ha="left", fontsize=18, fontweight="bold")
    n = audit["nonempty_n"]
    fig.text(0.07, 0.92, f"{n} nonempty workflows / {audit['tool_calls']} business calls; original saved DAGs", fontsize=11, color="#555555")
    values = [audit[k] * 100 for k in ("tool_precision_macro", "tool_recall_macro", "tool_f1_macro")]
    counts = [audit[k] for k in ("tools_exact_matches", "strict_order_matches", "strict_complete_matches")]
    panels = [
        (axes[0], values, ["Precision", "Recall", "F1"], None, "(a) Tool-call multiset (macro)"),
        (axes[1], [100 * c / n for c in counts],
         ["Exact tools", "Strict GT-order", "Tools +\nstrict order"], counts,
         "(b) Workflow-level exact match"),
    ]
    for ax, heights, labels, numerators, title in panels:
        colors = [figure_style.DIAGNOSTIC_BLUE_DARK] * 3
        if numerators is not None:
            colors[1:] = [figure_style.BASELINE_2, figure_style.OURS]
        ax.bar(range(3), heights, width=0.56, color=colors, zorder=3)
        for i, value in enumerate(heights):
            label = f"{value:.1f}%"
            if numerators is not None:
                label += f"\n{numerators[i]}/{n}"
            ax.text(i, value + 2, label, ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3), labels)
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Score (%)")
        ax.set_title(title, loc="left", pad=12, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    pairs, kept = audit["reference_order_pairs"], audit["preserved_order_pairs"]
    pair_rate = 100 * kept / pairs if pairs else 100
    fig.text(0.07, 0.185, f"DAGs preserve {kept}/{pairs} GT precedence pairs ({pair_rate:.1f}%). Saved node-list order alone matches {audit['tool_sequence_matches']}/{n}.", fontsize=9)
    fig.text(0.07, 0.14, "Strict order requires a DAG path for every GT precedence pair; legal independent-read parallelism also fails this criterion.", fontsize=9)
    fig.text(0.07, 0.095, f"Repeated calls retained; checker paths included. Empty GT reported separately: {audit['empty_list_matches']}/{audit['empty_n']} empty lists match.", fontsize=9)
    fig.text(0.07, 0.05, "Tool nodes/arguments are GT-derived. These are static metrics, not semantic task success or independent LLM accuracy.", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"fidelity_expanded.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def load(path: Path, sources: dict) -> dict:
    raw = path.read_bytes()
    sources[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def canonical(name: str, domain: str) -> str:
    name = name.rstrip("*")
    if domain == "airline":
        name = {"get_reservation_details": "get_order_details",
                "cancel_reservation": "cancel_order"}.get(name, name)
    return name


def compact_types(plan: dict, catalog: dict, domain: str) -> set[str]:
    chains = {c["chain_id"]: c["tools"] for c in catalog["chains"]}
    tools = set()
    for step in plan["steps"]:
        if step["kind"] == "chain":
            names = chains[step["chain_id"]]
        elif step["kind"] == "tool":
            names = [step["tool_name"]]
        else:
            raise ValueError(f"Unknown step kind: {step['kind']}")
        tools.update(canonical(n, domain) for n in names)
    return tools


def score(predicted: set[str], reference: set[str]) -> dict:
    overlap = len(predicted & reference)
    # Empty references are reported separately, not assigned arbitrary P/R/F1.
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(reference) if reference else None
    return {
        "precision": precision if reference else None,
        "recall": recall,
        "f1": 2 * overlap / (len(predicted) + len(reference)) if reference else None,
        "exact_type_match": predicted == reference,
        "missing_types": sorted(reference - predicted),
        "extra_types": sorted(predicted - reference),
        "predicted_types": sorted(predicted),
    }


def summarize(rows: list[dict]) -> dict:
    groups = {
        "nonempty": [r for r in rows if r["reference_types"]],
        "retail_nonempty": [r for r in rows if r["domain"] == "retail" and r["reference_types"]],
        "airline_nonempty": [r for r in rows if r["domain"] == "airline" and r["reference_types"]],
        "empty_reference": [r for r in rows if not r["reference_types"]],
        "source_validated_true": [r for r in rows if r["source_validated"] is True],
        "source_validated_false": [r for r in rows if r["source_validated"] is False],
    }
    result = {}
    for group, members in groups.items():
        result[group] = {"n": len(members)}
        result[group]["prf_n"] = sum(bool(r["reference_types"]) for r in members)
        for method in METHODS:
            result[group][method] = {}
            for metric in ("precision", "recall", "f1", "exact_type_match"):
                values = [r[method][metric] for r in members if r[method][metric] is not None]
                result[group][method][metric] = mean(values) if values else None
            result[group][method]["exact_matches"] = sum(r[method]["exact_type_match"] for r in members)
    return result


def draw(summary: dict, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = [figure_style.DIAGNOSTIC_BLUE_DARK, figure_style.NEUTRAL]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.72, wspace=0.25)
    fig.suptitle("Reference-to-workflow fidelity", x=0.07, y=0.98, ha="left", fontsize=18, fontweight="bold")
    total = summary["nonempty"]["n"] + summary["empty_reference"]["n"]
    fig.text(0.07, 0.92, f"Static tool-type agreement on {total} saved tau-bench-synthetic tasks", color="#555555", fontsize=11)
    fig.legend([Patch(facecolor=c) for c in colors], LABELS, loc="upper left",
               bbox_to_anchor=(0.064, 0.887), ncol=2, frameon=False)
    panels = [
        (axes[0], ["Precision", "Recall", "F1"], ["precision", "recall", "f1"],
         "(a) Nonempty reference tasks", ["nonempty"] * 3),
        (axes[1], ["Retail", "Airline", "Empty GT"], ["exact_type_match"] * 3,
         "(b) Exact tool-type set match", ["retail_nonempty", "airline_nonempty", "empty_reference"]),
    ]
    for ax, labels, metrics, title, groups in panels:
        for j, method in enumerate(METHODS):
            values = [summary[g][method][m] for g, m in zip(groups, metrics)]
            if any(v is None for v in values):
                raise ValueError("Plot requires nonempty retail, airline, and empty-GT groups")
            heights = [100 * v for v in values]
            positions = [i + (j - 0.5) * 0.34 for i in range(3)]
            ax.bar(positions, heights, width=0.31, color=colors[j], zorder=3)
            for x, y in zip(positions, heights):
                ax.text(x, y + 2, f"{y:.1f}", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(3), [f"{label}\n(n={summary[g]['n']})" for label, g in zip(labels, groups)])
        ax.set_title(title, loc="left", pad=13, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 113)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Agreement (%)")
        ax.grid(axis="y", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.07, 0.14, "Tool types only: chain '*' markers indicate presence; counts, arguments, order and behavior are not scored.", fontsize=9)
    fig.text(0.07, 0.10, "LLM prompts contained reference actions. Final workflows were built from GT; their scores are construction checks.", fontsize=9)
    fig.text(0.07, 0.06, f"P/R/F1 are per-task macro averages over {summary['nonempty']['n']} nonempty tasks. Empty-GT matching does not verify refusal language.", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"fidelity_tool_types.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning-dir", type=Path, default=DEFAULT_PLANNING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sources = {}
    samples = load(args.planning_dir / "selected_samples.json", sources)["samples"]
    workflows = load(args.planning_dir / "workflow.json", sources)["workflows"]
    sample_map = {s["selection_index"]: s for s in samples}
    workflow_map = {w["metadata"]["selection_index"]: w for w in workflows}
    if len(sample_map) != len(samples) or len(workflow_map) != len(workflows) or sample_map.keys() != workflow_map.keys():
        raise ValueError("Duplicate or mismatched source indices")
    expanded = expanded_audit(sample_map, workflow_map)
    raw_paths = {}
    for path in (args.planning_dir / "samples").glob("*/llm/chain_workflow.json"):
        idx = int(path.parent.parent.name.split("_", 1)[0])
        if idx in raw_paths:
            raise ValueError(f"Duplicate raw plan: {idx}")
        raw_paths[idx] = path
    if raw_paths.keys() != sample_map.keys():
        raise ValueError("Raw plans and selected samples do not match")
    rows = []
    for idx, sample in sorted(sample_map.items()):
        workflow = workflow_map[idx]
        if workflow["metadata"]["task_id"] != sample["task_id"]:
            raise ValueError(f"Task identity mismatch: {idx}")
        raw = load(raw_paths[idx], sources)["workflows"]
        if len(raw) != 1 or raw[0]["steps"] != workflow["metadata"]["compact_steps"]:
            raise ValueError(f"Stored compact steps differ from raw output: {idx}")
        catalog = load(raw_paths[idx].parent.parent / "prompts/chain_registry.json", sources)
        domain = sample["domain"]
        reference = {canonical(a["name"], domain) for a in sample["ground_truth_actions"]}
        predicted = compact_types(raw[0], catalog, domain)
        final = {canonical(n["entrypoint"], domain) for n in workflow["nodes"] if n["kind"] in ("app", "dnn")}
        rows.append({"selection_index": idx, "task_id": sample["task_id"], "domain": domain,
                     "source_validated": sample.get("validated"), "reference_types": sorted(reference),
                     "llm_plan": score(predicted, reference), "final_workflow": score(final, reference)})
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"metric_scope": "tool-type presence only", "aggregation": "per-task macro average; empty GT separated",
               "airline_aliases": {"get_reservation_details": "get_order_details", "cancel_reservation": "cancel_order"},
               "source_sha256": sources, "summary": summary, "per_task": rows}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output_dir / "expanded_metrics.json").write_text(json.dumps({**expanded, "source_sha256": sources}, indent=2) + "\n")
    with (args.output_dir / "per_task.csv").open("w", newline="") as stream:
        fields = ["selection_index", "task_id", "domain", "source_validated", "method", "precision", "recall", "f1", "exact_type_match", "missing_types", "extra_types"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                out = {k: row[k] for k in fields[:4]}
                out.update({"method": method, **{k: row[method][k] for k in fields[5:]}})
                for key in ("missing_types", "extra_types"):
                    out[key] = json.dumps(out[key])
                writer.writerow(out)
    draw(summary, args.output_dir)
    draw_expanded(expanded, args.output_dir)
    print(json.dumps({"output": str(args.output_dir.resolve()), "expanded": {k: v for k, v in expanded.items() if k != "per_task"}, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
