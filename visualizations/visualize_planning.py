"""Render planning-stage workflow visualizations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualizations.visualize_workflow import render_html  # noqa: E402



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    workflow_path = args.workflow_json or run_dir / "planning" / "workflow.json"
    doc = json.loads(workflow_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir or run_dir / "planning" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, workflow in enumerate(doc.get("workflows") or []):
        if args.workflow_index is not None and idx != args.workflow_index:
            continue
        name = sanitize(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{idx}")
        path = output_dir / f"{idx:02d}_{name}.html"
        path.write_text(render_html(workflow, doc, None), encoding="utf-8")
        paths.append(path)
    print(f"wfgen: wrote {len(paths)} planning visualization(s) to {output_dir}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize planning-stage workflow JSON.")
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--workflow-json", type=Path, default=None)
    parser.add_argument("--workflow-index", type=int, default=None)
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def sanitize(value: object) -> str:
    text = str(value)
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text.strip())
    return clean.strip("_") or "workflow"


if __name__ == "__main__":
    raise SystemExit(main())
