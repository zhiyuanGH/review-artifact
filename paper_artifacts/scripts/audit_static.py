"""Recompute the current static construction audit from external inputs offline.

This tests reference-derived tool/argument agreement and strict GT-order reachability.
It does not call models, execute workflows, or infer full-population task success.
The ten-workflow business-state observations remain a frozen measured snapshot.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return json.loads(path.read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, help="External directory containing workflow.json and selected_samples.json.")
    parser.add_argument("--fidelity-input", type=Path, help="External fidelity.json snapshot including the recorded business-state observations.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/fidelity")
    args = parser.parse_args()
    if args.examples_dir is None or args.fidelity_input is None:
        parser.error("Workflow/sample and fidelity measurement data are not bundled. Supply --examples-dir PATH and --fidelity-input PATH; see DATA_PROVENANCE.md.")
    inputs = args.examples_dir.resolve()
    missing = [str(path) for path in [inputs / "selected_samples.json", inputs / "workflow.json", args.fidelity_input] if not path.is_file()]
    if missing:
        parser.error("Missing external input files: " + ", ".join(missing))
    from generate_fidelity_figure import expanded_audit
    from generate_accuracy_overview import combine
    samples = read(inputs / "selected_samples.json")["samples"]
    workflows = read(inputs / "workflow.json")["workflows"]
    smap = {s["selection_index"]: s for s in samples}
    wmap = {w["metadata"]["selection_index"]: w for w in workflows}
    if len(smap) != len(samples) or len(wmap) != len(workflows) or smap.keys() != wmap.keys():
        raise ValueError("Duplicate or mismatched sample/workflow IDs")
    static = expanded_audit(smap, wmap)
    frozen = read(args.fidelity_input)
    metrics = combine(static, {"runs": frozen["business_runs"]})
    for key in ("population_n", "empty_n", "static_macro_averages", "strict_order_diagnostics", "business_state", "executed_unique_workflows", "unevaluated_workflows"):
        if metrics[key] != frozen[key]:
            raise ValueError(f"Static audit disagrees with the current paper snapshot: {key}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    columns = ["selection_index", "domain", "reference_calls", "actual_calls", "tool_precision", "tool_recall", "tool_f1", "tools_exact", "actions_and_parameters_match", "strict_complete"]
    with (output / "per_workflow.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in static["per_task"])
    print(f"Verified {len(samples)} frozen workflows against the current static audit; wrote {output}")

if __name__ == "__main__":
    main()
