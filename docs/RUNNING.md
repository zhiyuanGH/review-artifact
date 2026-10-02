# Running the framework

Use Python 3.10 or later. Commands below run from the repository root.
The editable installation supplies the `MOBench` import namespace even if an
anonymous download renames the enclosing directory.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . --no-deps
# Needed for cloud planning, remote dataset loading, and real HTTP inference:
python -m pip install -r requirements.txt
```

On Windows, activate `.venv\Scripts\activate` instead. Installation needs
`setuptools`; the runtime dependency list is in `requirements.txt`.

## Offline smoke run

This exercises local sample selection, deterministic workflow generation,
deployment assignment, generated runners, and mocked node execution. It needs
no API key, network, Android device, inference binary, model, or profiler.
Dry-run node timing is synthetic and is not a paper measurement.

```sh
python scripts/run_wfgen_pipeline.py \
  --run-dir results/offline_smoke \
  --sample-count 1 \
  --selected-samples data/tau_bench_fixed_1.json \
  --generation-mode deterministic \
  --no-clear --dry-run --resource-monitor off
```

For all ten local samples, use `--sample-count 10` and
`--selected-samples data/tau_bench_diverse_10.json`.
Use a fresh run directory when comparing runs. The pipeline's default behavior
clears its output directory; `--no-clear` in the smoke command preserves it.

## Three stages

```sh
python wfgen_plan.py --run-dir results/demo --sample-count 1 \
  --selected-samples data/tau_bench_fixed_1.json --generation-mode deterministic
python wfgen_deploy.py --run-dir results/demo \
  --deployment-strategy cli-balanced-wave-aware
python wfgen_execute.py --run-dir results/demo --dry-run --resource-monitor off
```

The stages write `planning/workflow.json`,
`deployment/workflow_deployment.json`, per-workflow runtime configurations and
compiled runners, then `execution/` JSON and HTML artifacts. Use `--help` on
an entrypoint to see current options. To run one compiled runner, pass its
matching runtime JSON with `--config` and use `--dry-run` for an offline check.

The primary entrypoint modules can also be invoked after installation as
`python -m MOBench.wfgen_plan`, `python -m MOBench.wfgen_deploy`, and
`python -m MOBench.wfgen_execute`.

## Cloud workflow generation

Install a current `openai` SDK with Responses API support and `datasets` from
`requirements.txt`, then export the API key
named by `--api-key-env`. `configs/qwen.example.json` and
`configs/openai.example.json` contain public example settings and no keys.
The command below uses the supplied local samples, so dataset downloading is
unnecessary. Cloud generation makes billable requests to the configured service.

```sh
python wfgen_plan.py --run-dir results/cloud --sample-count 1 \
  --selected-samples data/tau_bench_fixed_1.json --generation-mode cloud \
  --cloud-config configs/qwen.example.json --api-key-env DASHSCOPE_API_KEY
```

Supply a key in your environment before running this command. Private configs
such as `configs/qwen.local.json` are intentionally absent. Override `--model`
and `--base-url` to select your own OpenAI-compatible service. HTTP clients
read the usual proxy environment variables; preserve `NO_PROXY` for local
endpoints when using a proxy.

## Real Android execution

Provision Android platform-tools (`adb`), enable USB debugging, and install the
chosen inference runtime and its model on your own device. The default paths
are generic Android locations under `/data/local/tmp`; set `--adb`,
`--adb-serial <device-serial>`, `--llama-remote-dir`, and `--llama-model-path`
when deploying. `ADB` and the executable search path also configure ADB.

The CLI planners preserve empirical backend priority rules. For the measured
Snapdragon 8 Gen 3 Llama 3.2 3B profile, export
`MOBENCH_NPU_FIRST_LLAMA3B_SERIAL` with your own device serial before deployment
to select NPU, CPU, GPU order. Other Llama 3.2 3B devices use CPU, NPU, GPU order.
The original device identity is not included. Model-specific priorities and
all scheduling parameters are otherwise preserved.

After provisioning, rerun deployment with your device/model arguments and run
`wfgen_execute.py` without `--dry-run`. The `--list` option lists workflows
without executing them. For resource traces, the phone needs Perfetto and the
host needs Trace Processor. `scripts/setup_perfetto_tools.py` downloads the
host tool and checks a connected phone. Use `--resource-monitor off` when
resource collection is unnecessary.

ORT GenAI and Genie adapters also remain in the runtime. They need their own
Android runner, libraries, model bundle and configuration, supplied separately;
no vendor binaries, licensed models or build artifacts are bundled here.
The real OCR wrapper imports `pyagent.llm_utils.qnn_ocr_pipeline`, an optional
external module that is not included. Its simulated mode is self-contained.

## Additional source utilities

The `scripts/` directory contains deployment comparisons, framework
characterization, prompt-builder comparisons, resumed cloud planning, codegen
prompt generation and benchmarking, and phone inference/length/load-balance
utilities. Run each script with `--help`; real phone/cloud benchmarks require
explicitly provisioned inputs. The public characterization matrix is
`configs/characterization.example.json`. Length/load-balance sweeps accept
`--runtime-config` or a user-supplied `--method-csv`; historical result directories
are not bundled. Hardware benchmark `--dry-run` flags should not be assumed to
avoid device discovery; use the offline pipeline above for a device-free check.
