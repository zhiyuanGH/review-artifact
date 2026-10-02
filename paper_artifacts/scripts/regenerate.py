"""Redraw the current paper data figures from separately supplied frozen inputs."""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def read(data_dir, name):
    return json.loads((data_dir / name).read_text())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "motivation", "offloading", "generation", "execution", "characterization"], default="all")
    parser.add_argument("--data-dir", type=Path, help="Directory containing the external frozen input JSON files; measurement data is not bundled.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    expected = {
        "motivation": ["motivation_planning.json", "motivation_codegen.json", "motivation_topology.json"],
        "offloading": ["offloading_latency.json", "offloading_resource.json"],
        "generation": ["generation.json"],
        "execution": ["evaluation_sources.json"],
        "characterization": ["evaluation_sources.json"],
    }
    groups = list(expected) if args.group == "all" else [args.group]
    if args.data_dir is None:
        parser.error("Figure measurement data is not included in this code-only archive. Supply --data-dir PATH; see DATA_PROVENANCE.md for the required file schemas.")
    data_dir = args.data_dir.resolve()
    missing = sorted({name for group in groups for name in expected[group] if not (data_dir / name).is_file()})
    if missing:
        parser.error("Missing external input files: " + ", ".join(missing) + ". See DATA_PROVENANCE.md.")
    import matplotlib
    matplotlib.use("Agg")
    import generate_motivation_figures as motivation
    import generate_studies_generation_figures as generation
    import generate_studies_execution_figures as execution
    import generate_studies_characterization_figures as characterization
    import generate_selected10_followup_figures as followup
    import generate_p3_same_u_all_workflow_preview as offloading
    import generate_fig6_resource_timeline_column as resource
    motivation.FROZEN_INPUT_DIR = data_dir
    generation.FROZEN_INPUT_DIR = data_dir
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preview = output / "previews"
    preview.mkdir(parents=True, exist_ok=True)
    if "motivation" in groups:
        target = output / "motivation"
        motivation.CONFIG = replace(motivation.CONFIG, output_dir=target)
        motivation.OUT_DIR = target
        motivation.configure_style()
        summary = {"planning": motivation.plot_planning_cost(), "codegen": motivation.plot_codegen_cost(), "topology": motivation.plot_topology_parallelism()}
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if "offloading" in groups:
        target = output / "motivation/offloading"
        target.mkdir(parents=True, exist_ok=True)
        offloading.CONFIG = replace(offloading.CONFIG, output_stem=target / "offloading_policy_qwen15_p3_same_u")
        data = read(data_dir, "offloading_latency.json")
        records = [dict(row, latency_s=math.nan if row["latency_s"] is None else row["latency_s"]) for row in data["latency_records"]]
        offloading.plot(records, preview, offloading.SELECTED10_METHODS)
        timeline = resource.load_frozen_timeline(data_dir / "offloading_resource.json")
        timeline["methods"]["cpu_p3_same_u"]["display_metrics"] = {"cpu_pct": resource.cpu_counter_interval_means(timeline["methods"]["cpu_p3_same_u"]["runs"])}
        timeline["display_window"] = resource.common_display_window(timeline)
        resource.plot(timeline, target / "offloading_policy_qwen15_resource_timeline_8gen3_column")
    if "generation" in groups:
        target = output / "studies/generation"
        generation.CONFIG = replace(generation.CONFIG, output_dir=target)
        generation.PROVENANCE_PATH = target / "generation_efficiency_provenance.json"
        metrics = generation.build_metrics()
        generation.plot_generation_efficiency(metrics, preview)
        generation.write_provenance(metrics)
    data = read(data_dir, "evaluation_sources.json") if any(g in groups for g in ["execution", "characterization"]) else None
    if "execution" in groups:
        target = output / "studies/execution"
        execution.CONFIG = replace(execution.CONFIG, output_dir=target)
        followup.OUT = target
        execution.plot_execution_latency(data["latency"], preview)
        followup.draw_timeline(data["followup"], preview)
    if "characterization" in groups:
        target = output / "studies/characterization"
        metrics = data["characterization"]
        characterization.plot_hardware_by_workflow(metrics, target, preview)
        characterization.plot_model_by_workflow(metrics, target, preview)
        characterization.plot_framework_by_workflow(metrics, target, preview)
    paths = sorted(str(p.relative_to(output)) for p in output.rglob("*.pdf"))
    report = {"source_mode": "frozen_snapshot_redraw", "groups": groups, "pdf_outputs": paths, "phone_or_cloud_calls": 0}
    (output / "regeneration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Redrew {len(paths)} current data figures in {output}")

if __name__ == "__main__":
    main()
