# Measurement support

This is a **source-only upload**. It contains the source used to run the current phone measurements: the native business tools, the selected-ten host executor, its two connection guards, and the corrected ONNX Runtime GenAI Java wrapper. It also contains the small pinned tau-bench reference implementation used to check business outputs and state. The JSON files under `../examples/` are frozen workflow input specifications and runtime configuration examples, not recorded experimental outcomes. Complete cohorts, result records, measurements, initial/expected states, databases, traces and model packages remain external. No experiments were run to produce this source package.

## Source and measured workflow

| Included source | Role in the measured workflow |
| --- | --- |
| `mobile_tools/src/main.cpp`, `CMakeLists.txt`, `vendor/` | Fourteen phone-native business tool entrypoints, JSON input/output, whole-state locking and atomic updates. |
| `mobile_tools/scripts/prepare.py`, `validate.py`, `scheduling.py` | Frozen ten-workflow selection, independent reference replay, state/result checks, and explicit dependencies for conflicting primary records. |
| `mobile_tools/scripts/run_comparison.py` | Original three-model, five-configuration cohort: 150 workflow attempts per phone. |
| `mobile_tools/scripts/run_selected10_supplement.py`, `selected10_matrix.py` | Expanded model/hardware/framework/placement/structure conditions. The completed cohort contained 995 new attempts plus 300 audited original observations, including 15 separate resource-repeat observations. |
| `guarded_selected10_runner.py` | The selected-ten continuation runner. It checks device identity before each next workflow, outside that workflow's timing. |
| `guarded_ort_runner.py` | The independent corrected ORT cohort runner. It retains failed attempts, checks connection/boot identity, and stops on connection anomalies or failed pilot workflows. |
| `ort_wrapper/OgaGenerate.java`, `build_wrapper.py` | Corrected ORT generation wrapper and the actual DEX replacement/merge build procedure. |

The six executor dependency files in the frozen selected-ten and corrected-ORT runs were byte-identical to the source files retained under `mobile_tools/scripts/`. Their copied imports and local paths have been adapted for this package. The private inventory outside the public repository records both source and packaged hashes.

The eight-package model characterization used this same selected-ten executor and matrix, with 800 candidate observations: two phones × eight model packages × ten workflows × five configurations. There is no separate invented measurement script. The eight models and the original five candidate conditions are retained in the catalog. Figure aggregation and selection belong to the analysis artifacts, rather than the executor.

## Execution flags and attempt scope

`run_selected10_supplement.py` prepares local frozen input files by default. Only `--execute` or `--execute-prepared` reaches phone preflight and execution. `--execute-prepared` requires an untouched `prepared_not_run` manifest and verifies its runner/input hashes. It refuses automatic restart of a started or completed run. `--reuse-run` accepts a stopped checkpoint and reuses every recorded attempt, including a failure. `--no-reuse` disables reuse of condition observations but still requires the historical baseline audit and resource-reference selection.

The main measurement conditions have **one observation per device/model/configuration/workflow cell**. The resource suite is a distinct fixed cohort of three fresh repetitions for five conditions of workflow 179 on the 8 Gen 3 profile. The corrected ORT cohort contains 40 new cells: two phones × ten workflows × p=1/p=2, one observation each. Original ORT failures remain distinct from those corrected observations.

The continuation controllers and device-routing/state files are omitted because they describe the original environment. Their policy was to resume only unattempted jobs and never retry a recorded error. The two measured guard sources are retained. A previously attempted manifest is never a fresh prepared run.

## Timing, validation and failures

The host schedules the DAG. Business tool/state operations and checker inference run on the phone. The executor records runtime workflow wall time and host execute-call time separately; subprocess/ADB overhead is part of the host-observed path. Warmup and the connection checks before the workflow occur outside measured workflow execution. Trace sampling is retained as implemented.

The common checker budget is 64 new tokens with full prompts. llama.cpp uses temperature 0, six threads and a 240-second LLM timeout. The corrected ORT wrapper sets `max_length = encoded_prompt_tokens + requested_new_tokens`, checks the model context bound and fails instead of truncating. It retains deterministic generation, the native end-of-generation guard, output streaming and wrapper timing boundaries.

A successful workflow requires completion, valid generation, matching tool outputs and matching complete final phone state. Checker process success does not establish an independent semantic verdict. The ORT guard stores a failed attempt's partial time in `partial_wall_ms` and sets its complete `wall_ms` to null. Missing observations and failed observations must remain distinguishable; do not substitute a partial failed duration into complete-workflow latency means.

The state-conflict dependency algorithm and native tool semantics are preserved. Its primary-record guard is scoped to this sample, and it is not a general transaction-ordering analysis.

## Portable paths and device profiles

Run commands from the repository root. The Python import package is `MOBench`. The copied runners locate it without an original home-directory path. `adb` defaults to the executable on `PATH`; `--adb` can set another location. `--runtime-source` points to this repository root and defaults to that root, regardless of the clone directory name. Runtime snapshots exclude generated `local_data/` and Git metadata. Network proxy settings are supplied by the caller: the runners preserve `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` and their lowercase variants exactly as present in the environment, without adding a proxy fallback. Existing `NO_PROXY` values are retained; localhost exclusions are defaulted only when absent. `--resource-tools-dir` selects the local Perfetto/trace-processor support directory.

The two measured profiles are parameterized as `--device-a-serial` (SM8750, 8 Elite) and `--device-b-serial` (SM8650, 8 Gen 3). Environment defaults are `MOBENCH_DEVICE_A_SERIAL` and `MOBENCH_DEVICE_B_SERIAL`; the published catalog uses placeholders. SoC checks and profile-specific condition order remain unchanged. The original comparison runner uses `--serial` and `--device-profile elite|gen3` to preserve the configuration-order rotation/reversal.

The `wfgen.v1` schema, `wfgen-tools` executable name, remote tool directory convention, and `WFGEN_OWNER_TOKEN` environment field are retained as protocol compatibility identifiers. They do not refer to an original repository path or account.

## Build and external inputs

The support scripts use Python's standard library plus the copied framework. Native tools require CMake 3.22+, a C++17 compiler, and for Android an arm64 Android NDK toolchain targeting API 26+; `build.py` accepts `--ndk` and `--cmake`. JSON 3.11.3 is vendored with its MIT notice. tau-bench reference code and its MIT notice are retained.

A local native build can be requested with:

```sh
python3 experiment_support/mobile_tools/scripts/build.py \
  --target host --output local_data/build-host
```

The following inputs are intentionally external:

- The full compact280 frozen workflow input used for the deterministic length-spaced selection. The ten published workflows are inspection examples, not the original 280-workflow selection population.
- The pinned complete retail/airline databases and independently replayed expected outputs/state. The original initial state files total approximately 7.8 MB, and ten expected-state/step pairs total approximately 32.5 MB. The native state envelope is `{"domain":"retail|airline","data":{...}}`. Provide actual pinned data, rather than inferring initial state from the example actions.
- A freshly built `wfgen-tools` binary, and phone installations of llama.cpp, GGUF model packages, ORT GenAI model/native library/JAR packages, or the Genie/QNN package selected by the catalog.
- Perfetto/trace-processor tools and the matching Android device environment.
- Historical baseline/checkpoint raw records if using the supplementary runner's audit/reuse path. These records, reference snapshots and controller state are not reconstructed from figure summaries.

`prepare.py --workflow FULL_WORKFLOW_JSON --tau-root TAU_BENCH_ROOT --output local_data/phone_tools_v1` uses the actual complete input and databases to generate a bundle. `tau_bench_reference/` supplies the retained source implementation, but does not contain the omitted database files. The measurement runner must receive the matching bundle/binary/runtime and audited baseline explicitly. This package provides source and examples; it does not claim a self-contained reproduction of omitted model installations or raw cohorts.

See `ort_wrapper/README.md` for the original-wrapper build dependency and `../examples/README.md` for the frozen examples.
