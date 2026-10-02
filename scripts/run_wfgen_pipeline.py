"""Run the full MOBench pipeline with one command.

Default behavior:
1. Clear ./results with --yes.
2. Generate N workflows.
3. Generate deployment plans/runtime configs.
4. Execute the deployed workflows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    from MOBench.deployment_planning import DEPLOYMENT_STRATEGIES
except ImportError:
    from deployment_planning import DEPLOYMENT_STRATEGIES

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_LOCAL_SELECTED_SAMPLES = REPO_ROOT / "data" / "tau_bench_diverse_10.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_name = args.run_name or f"wfgen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_n{args.sample_count}"
    results_dir = args.results_dir.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir else results_dir / run_name

    python = str(args.python.resolve()) if args.python else sys.executable

    if not args.no_clear:
        clear_target = run_dir if args.run_dir else results_dir
        clear_cmd = [python, str(REPO_ROOT / "scripts" / "clear_results.py"), "--results-dir", str(clear_target), "--yes"]
        if clear_target.resolve() != DEFAULT_RESULTS_DIR.resolve():
            clear_cmd.append("--allow-nonstandard-results-dir")
        run_command(clear_cmd)

    plan_cmd = [
        python,
        str(REPO_ROOT / "wfgen_plan.py"),
        "--run-dir",
        str(run_dir),
        "--sample-count",
        str(args.sample_count),
        "--selection-strategy",
        args.selection_strategy,
        "--sample-offset",
        str(args.sample_offset),
        "--seed",
        str(args.seed),
        "--generation-mode",
        args.generation_mode,
        "--prompt-builder",
        args.prompt_builder,
    ]
    if args.generation_mode == "cloud":
        plan_cmd.extend(
            [
                "--cloud-provider",
                args.cloud_provider,
                "--model",
                args.model,
                "--base-url",
                args.base_url,
                "--api-key-env",
                args.api_key_env,
                "--cloud-config",
                str(args.cloud_config),
                "--cloud-timeout",
                str(args.cloud_timeout),
                "--cloud-temperature",
                str(args.cloud_temperature),
                "--cloud-max-tokens",
                str(args.cloud_max_tokens),
            ]
        )
        if args.skip_api_smoke_test:
            plan_cmd.append("--skip-api-smoke-test")
    if args.extra_instruction:
        plan_cmd.extend(["--extra-instruction", args.extra_instruction])
    selected_samples = resolve_selected_samples(args)
    if selected_samples:
        plan_cmd.extend(["--selected-samples", str(selected_samples)])
    if args.include_empty_actions:
        plan_cmd.append("--include-empty-actions")
    if args.no_plan_visualize:
        plan_cmd.append("--no-visualize")
    run_command(plan_cmd)

    deploy_cmd = [
        python,
        str(REPO_ROOT / "wfgen_deploy.py"),
        "--run-dir",
        str(run_dir),
        "--deployment-strategy",
        args.deployment_strategy,
        "--max-parallel-nodes",
        str(args.max_parallel_nodes),
        "--llm-max-tokens",
        str(args.llm_max_tokens),
        "--llama-cli-warmup-runs",
        str(args.llama_cli_warmup_runs),
        "--llama-cli-warmup-max-tokens",
        str(args.llama_cli_warmup_max_tokens),
        "--llama-cli-warmup-prompt",
        args.llama_cli_warmup_prompt,
    ]
    if args.adb:
        deploy_cmd.extend(["--adb", str(args.adb)])
    if args.adb_serial:
        deploy_cmd.extend(["--adb-serial", args.adb_serial])
    if args.no_deploy_visualize:
        deploy_cmd.append("--no-visualize")
    run_command(deploy_cmd)

    execute_cmd = [
        python,
        str(REPO_ROOT / "wfgen_execute.py"),
        "--run-dir",
        str(run_dir),
        "--resource-monitor",
        args.resource_monitor,
        "--resource-sample-interval",
        str(args.resource_sample_interval),
        "--perfetto-buffer-mb",
        str(args.perfetto_buffer_mb),
        "--perfetto-tools-dir",
        str(args.perfetto_tools_dir),
    ]
    if args.dry_run:
        execute_cmd.append("--dry-run")
    if args.adb:
        execute_cmd.extend(["--adb", str(args.adb)])
    if args.adb_serial:
        execute_cmd.extend(["--adb-serial", args.adb_serial])
    if args.no_ensure_llama_server:
        execute_cmd.append("--no-ensure-llama-server")
    if args.no_execute_visualize:
        execute_cmd.append("--no-visualize")
    if args.stop_on_failure:
        execute_cmd.append("--stop-on-failure")
    if args.fail_on_error:
        execute_cmd.append("--fail-on-error")
    if args.perfetto_required:
        execute_cmd.append("--perfetto-required")
    run_command(execute_cmd)

    print("", flush=True)
    print("[pipeline] completed", flush=True)
    print(f"[pipeline] run:        {run_dir}", flush=True)
    print(f"[pipeline] planning:   {run_dir / 'planning'}", flush=True)
    print(f"[pipeline] deployment: {run_dir / 'deployment'}", flush=True)
    print(f"[pipeline] execution:  {run_dir / 'execution'}", flush=True)
    print(f"[pipeline] summary:    {run_dir / 'execution' / 'EXECUTION_SUMMARY.md'}", flush=True)
    print(f"[pipeline] html:       {run_dir / 'execution' / 'visualizations'}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear results, generate workflows, and execute them.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--sample-count", type=int, default=10, help="Number of workflows to generate and execute.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Results directory to clear and write into.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run root. Defaults to <results-dir>/<run-name>.")
    parser.add_argument("--run-name", default=None, help="Prefix for generated plan/execute directories.")
    parser.add_argument("--python", type=Path, default=None, help="Python executable used to run MOBench scripts. Defaults to the current interpreter.")
    parser.add_argument("--no-clear", action="store_true", default=False, help="Do not clear results before running.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Execute with mocked node outputs.")
    parser.add_argument("--selection-strategy", choices=["diverse", "fixed"], default="diverse", help="Planning selection strategy.")
    parser.add_argument("--generation-mode", choices=["cloud", "deterministic"], default="cloud", help="Workflow generation backend.")
    parser.add_argument("--prompt-builder", choices=["full", "compact-chain"], default="full", help="Prompt builder used during cloud planning.")
    parser.add_argument(
        "--deployment-strategy",
        choices=DEPLOYMENT_STRATEGIES,
        default="cli-balanced-wave-aware",
        help="Rule-based deployment strategy.",
    )
    parser.add_argument("--cloud-provider", default="qwen", help="Cloud LLM provider label recorded in metadata.")
    parser.add_argument("--model", default="qwen-plus", help="Cloud LLM model used for workflow generation.")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="OpenAI-compatible cloud LLM base URL.")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY", help="Environment variable containing the cloud LLM API key.")
    parser.add_argument("--cloud-config", type=Path, default=REPO_ROOT / "configs" / "qwen.local.json", help="Optional local JSON config with api_key. This file should be git-ignored.")
    parser.add_argument("--cloud-timeout", type=int, default=120, help="Cloud LLM request timeout in seconds.")
    parser.add_argument("--cloud-temperature", type=float, default=0.0, help="Cloud LLM generation temperature.")
    parser.add_argument("--cloud-max-tokens", type=int, default=8192, help="Cloud LLM max tokens for workflow JSON generation.")
    parser.add_argument("--skip-api-smoke-test", action="store_true", default=False, help="Skip the Qwen API smoke test during planning.")
    parser.add_argument("--extra-instruction", default=None, help="Additional instruction appended to the cloud generation prompt.")
    parser.add_argument("--sample-offset", type=int, default=0, help="Starting tau-bench row for fixed planning.")
    parser.add_argument("--seed", type=int, default=0, help="Planning seed.")
    parser.add_argument("--include-empty-actions", action="store_true", default=False, help="Allow samples with no ground-truth tool actions.")
    parser.add_argument("--selected-samples", type=Path, default=None, help="Existing selected_samples JSON to use for planning.")
    parser.add_argument(
        "--no-auto-local-samples",
        action="store_true",
        default=False,
        help="Do not auto-use data/tau_bench_diverse_10.json when it can satisfy --sample-count.",
    )
    parser.add_argument("--max-parallel-nodes", type=int, default=4, help="Maximum workflow nodes to run in parallel per wave.")
    parser.add_argument("--llm-max-tokens", type=int, default=64, help="Maximum generated tokens per LLM node/tool call.")
    parser.add_argument("--llama-cli-warmup-runs", type=int, default=1, help="Warmup llama-cli calls per unique model/backend before measured execution; use 0 for cold-start timing.")
    parser.add_argument("--llama-cli-warmup-max-tokens", type=int, default=1, help="Generated tokens for llama-cli warmup calls.")
    parser.add_argument("--llama-cli-warmup-prompt", default="OK", help="Prompt text used for llama-cli warmup calls.")
    parser.add_argument("--adb", type=Path, default=None, help="adb executable for phone execution.")
    parser.add_argument("--adb-serial", default=None, help="Optional adb device serial.")
    parser.add_argument("--no-ensure-llama-server", action="store_true", default=False, help="Do not auto-start phone llama-server.")
    parser.add_argument("--resource-monitor", choices=["perfetto", "off"], default="perfetto", help="Phone resource monitor used around execution.")
    parser.add_argument("--resource-sample-interval", type=float, default=1.0, help="Seconds between lightweight ADB resource samples.")
    parser.add_argument("--perfetto-buffer-mb", type=int, default=128, help="Perfetto ring buffer size in MB.")
    parser.add_argument("--perfetto-tools-dir", type=Path, default=REPO_ROOT / ".wfgen_perfetto", help="Local directory for PC-side Perfetto trace_processor tooling.")
    parser.add_argument("--perfetto-required", action="store_true", default=False, help="Fail execution if Perfetto monitoring cannot start or stop cleanly.")
    parser.add_argument("--no-plan-visualize", action="store_true", default=False, help="Skip planned workflow visualizations.")
    parser.add_argument("--no-deploy-visualize", action="store_true", default=False, help="Skip deployment visualizations.")
    parser.add_argument("--no-execute-visualize", action="store_true", default=False, help="Skip execution-result visualizations.")
    parser.add_argument("--stop-on-failure", action="store_true", default=False, help="Stop after the first failed workflow.")
    parser.add_argument("--fail-on-error", action="store_true", default=False, help="Return non-zero if any workflow fails.")
    return parser.parse_args(argv)


def resolve_selected_samples(args: argparse.Namespace) -> Path | None:
    if args.selected_samples:
        return args.selected_samples.resolve()
    if args.no_auto_local_samples:
        return None
    if DEFAULT_LOCAL_SELECTED_SAMPLES.exists() and local_sample_count(DEFAULT_LOCAL_SELECTED_SAMPLES) >= args.sample_count:
        print(f"[pipeline] using local selected samples: {DEFAULT_LOCAL_SELECTED_SAMPLES}", flush=True)
        return DEFAULT_LOCAL_SELECTED_SAMPLES
    return None


def local_sample_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    return len(samples) if isinstance(samples, list) else 0


def run_command(cmd: list[str]) -> None:
    print("", flush=True)
    print("[pipeline] " + " ".join(quote_part(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def quote_part(part: object) -> str:
    text = str(part)
    if not text or any(char.isspace() for char in text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


if __name__ == "__main__":
    raise SystemExit(main())
