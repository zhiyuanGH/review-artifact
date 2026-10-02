"""Loading and validation helpers for deployed MOBench workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from MOBench.deployment_planning import DEPLOYMENT_SCHEMA_VERSION
    from MOBench.deployment_planning.core import validate_deployment_document
except ImportError:
    from deployment_planning import DEPLOYMENT_SCHEMA_VERSION
    from deployment_planning.core import validate_deployment_document


def load_workflow_deployment(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION:
        if payload.get("schema_version") == "wfgen.v1" or "workflows" in payload:
            raise ValueError(
                f"{path} is a planned workflow, not a deployment artifact. "
                "Run `python wfgen_deploy.py --run-dir <run-dir>` and execute "
                "`deployment/workflow_deployment.json`."
            )
        raise ValueError(f"{path} is not a MOBench deployment artifact with schema_version {DEPLOYMENT_SCHEMA_VERSION}.")
    validate_deployment_document(payload)
    return payload


def workflow_document(deployment_doc: dict[str, Any]) -> dict[str, Any]:
    return deployment_doc["workflow_document"]


def deployment_plan(deployment_doc: dict[str, Any]) -> dict[str, Any]:
    return deployment_doc["deployment_plan"]
