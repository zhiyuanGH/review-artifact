"""Run deployment benchmarks with sequential workflow execution.

This wrapper reuses scripts/run_deployment_benchmarks.py but forces
max_parallel_nodes=1 for both initial attempts and retries. Deployment
algorithms are still compared normally; only execution of runnable workflow
nodes is serialized.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_deployment_benchmarks  # noqa: E402


FORCED_OPTIONS = {
    "--max-parallel-nodes": "1",
    "--retry-max-parallel-nodes": "1",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    forwarded = strip_options_with_values(args, set(FORCED_OPTIONS))

    if "--output-dir" not in forwarded and "--retry-results-dir" not in forwarded:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        forwarded.extend(["--output-dir", str(REPO_ROOT / "results" / f"sequential_deployment_benchmark_{stamp}")])

    for option, value in FORCED_OPTIONS.items():
        forwarded.extend([option, value])

    print("[sequential-bench] forcing max_parallel_nodes=1 and retry_max_parallel_nodes=1", flush=True)
    return run_deployment_benchmarks.main(forwarded)


def strip_options_with_values(argv: list[str], option_names: set[str]) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in option_names:
            skip_next = True
            print(f"[sequential-bench] ignoring user supplied {item}; sequential mode forces 1", flush=True)
            continue
        if any(item.startswith(f"{option}=") for option in option_names):
            option = item.split("=", 1)[0]
            print(f"[sequential-bench] ignoring user supplied {option}; sequential mode forces 1", flush=True)
            continue
        stripped.append(item)
    return stripped


if __name__ == "__main__":
    raise SystemExit(main())
