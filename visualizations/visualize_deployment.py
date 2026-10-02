"""Render deployment-stage MOBench visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment_planning.visualization import render_deployment_html, write_deployment_visualizations


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    deployment_path = args.deployment_json or run_dir / "deployment" / "workflow_deployment.json"
    deployment_doc = json.loads(deployment_path.read_text(encoding="utf-8"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_deployment_html(deployment_doc, args.workflow_index), encoding="utf-8")
        print(f"wfgen: wrote deployment visualization to {args.output}")
        return 0
    output_dir = args.output_dir or run_dir / "deployment" / "visualizations"
    paths = write_deployment_visualizations(deployment_doc, output_dir)
    print(f"wfgen: wrote {len(paths)} deployment visualization(s) to {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a MOBench deployment plan.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--deployment-json", type=Path, default=None)
    parser.add_argument("--workflow-index", type=int, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
