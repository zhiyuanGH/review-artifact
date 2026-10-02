"""Deployment planning stage helpers."""

DEPLOYMENT_STAGE_DIR = "deployment"
DEPLOYMENT_SCHEMA_VERSION = "wfgen.deployment.v1"

FAIR_CLI_DEPLOYMENT_STRATEGIES = [
    "cli-cpu-only",
    "cli-gpu-only",
    "cli-npu-only",
    "cli-balanced-round-robin",
    "cli-balanced-wave-aware",
    "cli-odabench-wo-decider",
]

BASELINE_DEPLOYMENT_STRATEGIES = [
    "server-batch-npu",
]

FRAMEWORK_CHARACTERIZATION_STRATEGIES = [
    "framework-ort-genai-cpu",
    "framework-genie-npu",
]

LEGACY_DEPLOYMENT_STRATEGIES = [
    "cli-parallel-cpu",
    "cli-parallel-gpu",
    "cli-parallel-npu",
]

DEPLOYMENT_STRATEGIES = [
    *FAIR_CLI_DEPLOYMENT_STRATEGIES,
    *BASELINE_DEPLOYMENT_STRATEGIES,
    *FRAMEWORK_CHARACTERIZATION_STRATEGIES,
    *LEGACY_DEPLOYMENT_STRATEGIES,
    "mock",
]
