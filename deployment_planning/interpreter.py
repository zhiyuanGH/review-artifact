"""Compile workflow JSON into executable Python workflow scripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from MOBench.workflow_planning.generate import validate_workflow_semantics
except ImportError:
    from workflow_planning.generate import validate_workflow_semantics


def compile_workflow_document(
    workflow_json_path: Path,
    *,
    output_path: Path,
    workflow_index: int = 0,
    project_root: Path | None = None,
) -> Path:
    doc = json.loads(workflow_json_path.read_text(encoding="utf-8"))
    validate_workflow_semantics(doc)

    workflows = doc["workflows"]
    if workflow_index < 0 or workflow_index >= len(workflows):
        raise IndexError(f"workflow_index {workflow_index} out of range for {len(workflows)} workflows")

    selected_doc = {
        "schema_version": doc.get("schema_version"),
        "benchmark": doc.get("benchmark"),
        "generation_metadata": doc.get("generation_metadata", {}),
        "generation_notes": doc.get("generation_notes", []),
        "workflows": [workflows[workflow_index]],
    }
    project_root = (project_root or Path.cwd()).resolve()
    script = render_executable_script(selected_doc, project_root=project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def render_executable_script(workflow_doc: dict[str, Any], *, project_root: Path) -> str:
    workflow = workflow_doc["workflows"][0]
    title = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or "workflow")
    embedded = json.dumps(workflow_doc, indent=2, ensure_ascii=True)
    project_root_json = json.dumps(str(project_root))
    return f'''#!/usr/bin/env python3
"""Generated executable workflow: {title}.

This file was produced by `python -m MOBench.deployment_planning.interpreter`.
It executes dependency waves from the workflow JSON. Nodes inside the same wave
run concurrently.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path({project_root_json})
PACKAGE_ROOT = PROJECT_ROOT / "MOBench" if (PROJECT_ROOT / "MOBench").is_dir() else PROJECT_ROOT
for import_root in (PACKAGE_ROOT.parent, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import MOBench
from MOBench.workflow_execution.runtime import MobileWorkflowRuntime, load_runtime_config

PACKAGE_ROOT = Path(MOBench.__file__).resolve().parent


WORKFLOW_DOCUMENT = json.loads({embedded!r})
WORKFLOW = WORKFLOW_DOCUMENT["workflows"][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generated workflow {title}")
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs" / "mobile_runtime.example.json"),
        help="Runtime config for llama-server and QNN tool nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call llama-server or qnn-net-run; emit mock node outputs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(PROJECT_ROOT / "generated_workflows" / "{title}.result.json"),
        help="Where to write execution results.",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    runtime = MobileWorkflowRuntime(config, dry_run=args.dry_run)
    result = runtime.execute_workflow(WORKFLOW)
    result["workflow_document_metadata"] = {{
        "schema_version": WORKFLOW_DOCUMENT.get("schema_version"),
        "benchmark": WORKFLOW_DOCUMENT.get("benchmark"),
        "generation_metadata": WORKFLOW_DOCUMENT.get("generation_metadata", {{}}),
    }}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] workflow={{result['workflow_id']}} status={{result['status']}} output={{output_path}}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def sanitize_identifier(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "workflow"
    if value[0].isdigit():
        value = f"workflow_{value}"
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile MOBench workflow JSON into an executable Python script.",
    )
    parser.add_argument("workflow_json", type=Path, help="Input workflow JSON file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Python script. Defaults to generated_workflows/<workflow>.py.",
    )
    parser.add_argument(
        "--workflow-index",
        type=int,
        default=0,
        help="Workflow index to compile when the JSON contains multiple workflows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    doc = json.loads(args.workflow_json.read_text(encoding="utf-8"))
    workflow = doc["workflows"][args.workflow_index]
    output = args.output
    if output is None:
        output = Path("generated_workflows") / f"{sanitize_identifier(workflow.get('name', 'workflow'))}.py"
    compile_workflow_document(
        args.workflow_json,
        output_path=output,
        workflow_index=args.workflow_index,
    )
    print(f"wfgen: wrote executable workflow to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
