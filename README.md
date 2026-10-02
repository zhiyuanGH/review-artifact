# MOBench

MOBench constructs executable workflow benchmarks from prompt datasets and
measures their execution on mobile CPU, GPU, and NPU backends. The source is
organized into three stages: workflow planning, deployment/interpretation, and
execution. The `wfgen_*` entrypoint filenames and lowercase workflow schema
identifiers are retained for compatibility with recorded artifacts.

This review snapshot provides the framework implementation, native tool and
measurement harness sources, and the analysis/plotting sources for the current
paper.

## Contents

| Directory or entrypoint | Purpose |
| --- | --- |
| `wfgen_plan.py`, `workflow_planning/`, `workflow_generation/` | Prompt selection, cloud planning, and call-chain expansion. |
| `wfgen_deploy.py`, `deployment_planning/` | Workflow interpretation, placement, and executable wrapper generation. |
| `wfgen_execute.py`, `workflow_execution/`, `tools/` | Workflow execution and tool/LLM dispatch. |
| `resource_monitoring/` | Perfetto and device resource sampling. |
| `scripts/` | Pipeline and deployment benchmark commands. |
| `experiment_support/` | Native mobile tools and paper-specific measurement sources. |
| `examples/` | Frozen workflow/configuration examples. |
| `paper_artifacts/` | Current plotting/analysis sources and the required external-input mapping. |
| `docs/` | Installation, scope, and anonymous upload instructions. |

## Install and run

Python 3.10 or newer is required. Follow [RUNNING.md](docs/RUNNING.md) for
installation and the offline pipeline check. The offline check uses deterministic
planning and mock execution: it does not contact a cloud LLM, start mobile
inference, or collect hardware measurements.

Cloud planning requires a user-supplied DashScope API key. Real measurements
require a configured Android device, ADB, and the appropriate model/runtime
packages. See the framework guide and `experiment_support/` documentation.

## Paper analysis and plotting

The `paper_artifacts/` guide lists the current figures, the external input files
required by each script, and the commands to redraw them after those inputs are
supplied. This source-only repository does not include measurement snapshots
or generated figures. The scripts preserve failure and missing-data accounting.

## Artifact scope

This source repository includes small workflow/configuration examples. The
measurement and analysis sources document how results are collected and
processed. See [DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for the paper's
availability statement.

See [SOURCE_SNAPSHOT.md](docs/SOURCE_SNAPSHOT.md) for packaging changes and
[NOTICE.md](NOTICE.md) for third-party notices.

## Upload anonymously

Follow [ANONYMOUS_UPLOAD.md](docs/ANONYMOUS_UPLOAD.md). Upload this directory into
a fresh GitHub repository, then create a read-only mirror on
[Anonymous GitHub](https://anonymous.4open.science/). Share the generated anonymous
URL with reviewers.
