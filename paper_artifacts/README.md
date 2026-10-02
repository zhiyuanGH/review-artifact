# Paper plotting and analysis source

This is a **code-only** package for the current MOBench paper. It contains the current plotting/analysis source, shared colors and styles, dependencies, and input documentation. Experimental measurements, frozen figure inputs, workflow/sample populations, figure PDFs, and raw trace archives are **not included**.

The source preserves the current figure aggregation, fastest-successful selection, failure markers and styles. Data-dependent reproduction requires the corresponding datasets to be supplied separately. The sources do not start mobile, cloud or model-inference experiments.

## Commands

Use Python 3.10 or newer with the packages in `requirements.txt`; the validated environment used Python 3.12, Matplotlib 3.10.3 and NumPy 2.2.6. These help commands work without measurement data or installed plotting dependencies:

```sh
python scripts/regenerate.py --help
python scripts/audit_static.py --help
```

With separately supplied input files:

```sh
python scripts/regenerate.py --data-dir /path/to/figure_inputs
python scripts/regenerate.py --group characterization --data-dir /path/to/figure_inputs --output-dir output/characterization_run
python scripts/audit_static.py --examples-dir /path/to/280_workflows --fidelity-input /path/to/figure_inputs/fidelity.json
```

`regenerate.py` writes the 15 current data-figure PDFs, matching SVGs and summaries to `output/`. Valid groups are `motivation`, `offloading`, `generation`, `execution`, `characterization`, and `all`. The selected group requires only its inputs listed below. Missing inputs produce a clear error rather than substituted measurements. Once dependencies and external inputs are available, these commands run offline.

`audit_static.py` recomputes the 280-workflow static reference-conversion audit from external workflow/input files and checks it against the separately supplied fidelity snapshot. It exports JSON/CSV and does not generate manuscript source. Business-state results remain previously recorded observations; no phone execution is performed.

## Current figure map

Input paths in this table are relative to the externally supplied `--data-dir`. Output paths are relative to `output/`.

| Paper figure | Expected external input | Rendering source | Group and output |
| --- | --- | --- | --- |
| Fig. 3a/b: planning latency and token breakdown | `motivation_planning.json` | `scripts/generate_motivation_figures.py` | `--group motivation`; `motivation/planning/planning_cost_latency.pdf` and `planning_cost_breakdown.pdf` |
| Fig. 4a/b: code-generation latency and tokens | `motivation_codegen.json` | Same source | `--group motivation`; `motivation/codegen/codegen_latency.pdf` and `codegen_tokens.pdf` |
| Fig. 5a/b: workflow swimlane and topology reduction | `motivation_topology.json` | Same source | `--group motivation`; `motivation/topology/topology_swimlane.pdf` and `topology_parallelism.pdf` |
| Fig. 6a: fixed-backend latency and Oracle | `offloading_latency.json` | `scripts/generate_p3_same_u_all_workflow_preview.py` | `--group offloading`; `motivation/offloading/offloading_policy_qwen15_p3_same_u.pdf` |
| Fig. 6b: R179 CPU/GPU timeline | `offloading_resource.json` | `scripts/generate_fig6_resource_timeline_column.py` | `--group offloading`; `motivation/offloading/offloading_policy_qwen15_resource_timeline_8gen3_column.pdf` |
| Fig. 9a/b: generation time and token CDFs | `generation.json` | `scripts/generate_studies_generation_figures.py` | `--group generation`; `studies/generation/generation_time_efficiency.pdf` and `generation_token_efficiency.pdf` |
| Fig. 10a: execution latency | `evaluation_sources.json`, `latency` | `scripts/generate_studies_execution_figures.py` | `--group execution`; `studies/execution/execution_latency.pdf` |
| Fig. 10b: R179 execution resources | `evaluation_sources.json`, `followup.timeline` | `scripts/generate_selected10_followup_figures.py` | `--group execution`; `studies/execution/execution_resource_usage.pdf` |
| Fig. 11: hardware comparison | `evaluation_sources.json`, `characterization.hardware_by_workflow` | `scripts/generate_studies_characterization_figures.py` | `--group characterization`; `studies/characterization/characterization_hardware_by_workflow.pdf` |
| Fig. 12: framework comparison | `evaluation_sources.json`, `characterization.framework_by_workflow` | Same source | `--group characterization`; `studies/characterization/characterization_framework_by_workflow.pdf` |
| Fig. 13: eight model packages | `evaluation_sources.json`, `characterization.model_by_workflow` | Same source | `--group characterization`; `studies/characterization/characterization_model_by_workflow.pdf` |
| Table 2: construction fidelity | External `workflow.json`, `selected_samples.json`, and `fidelity.json` | `scripts/audit_static.py`, `generate_fidelity_figure.py`, `generate_accuracy_overview.py` | `python scripts/audit_static.py` with both input options; `fidelity/metrics.json` and `per_workflow.csv` |

Figs. 1, 2, 7, and 8 are conceptual diagrams authored separately; they are not generated by these data scripts, and their PDF/editable originals are not included in this code-only upload. The manuscript and LaTeX source are also excluded.

## Interpretation and limits

Redrawing from derived snapshots reproduces the plots and their numerical aggregation. Reanalysis from complete raw experiment records requires the separate saved raw cohorts, manifests, execution results and resource traces. Source parsing, aggregation, selection and resource-processing functions remain in the copied scripts for inspection; their raw-cohort refresh paths require those omitted files. Paths below `unbundled_raw/` denote evidence outside this package, not files to download automatically.

The plotting rules distinguish incomplete groups from successful latency observations. Fig. 6 uses its original best-of-eight Oracle; the execution/model/hardware comparisons retain their five-candidate selection; framework characterization retains its runtime-specific candidate sets. Resource curves preserve the original missing-reading and GPU zero-counter conventions. Fig. 6b retains three-run sample standard deviations and the counter-interval CPU representation; Fig. 10b is one predeclared observed run per method. Whole-device memory usage is not isolated model memory, and NPU utilization is unavailable.

The static audit tests construction fidelity of reference-derived tools and arguments, not independent planner accuracy. Strict reference order checks all ground-truth precedence pairs, including independent reads; it is not a semantic-dependency or task-success score. The saved business-state observations cover ten fixed-action workflows and do not estimate success for all 280 workflows.

Identifiers `elite` and `gen3` are consistent pseudonyms for the measured Snapdragon 8 Elite and 8 Gen 3 devices, not physical ADB serials. Author-machine paths and original repository commit references have been removed.

Before excluding the data from this upload, local verification with the corresponding existing frozen inputs regenerated all 15 active data figures. Their dimensions and rendered pixels matched the current figures at 140 dpi, and the 280-workflow static audit matched the current frozen metrics. This verifies the code against those local inputs; the code-only archive by itself cannot reproduce the measured figures or independently verify the mobile results. Original and packaged file hashes and the local verification evidence are retained outside the upload.
