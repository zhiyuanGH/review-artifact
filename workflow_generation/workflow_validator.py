"""Validation for cloud-generated MOBench workflow documents."""

from __future__ import annotations

from typing import Any

from ..workflow_planning.generate import validate_workflow_semantics
from .tool_registry import tool_names


def validate_cloud_workflow(workflow_doc: dict[str, Any], selected_payload: dict[str, Any]) -> None:
    validate_workflow_semantics(workflow_doc)
    validate_workflow_count(workflow_doc, selected_payload)
    validate_known_tool_entrypoints(workflow_doc)


def validate_workflow_count(workflow_doc: dict[str, Any], selected_payload: dict[str, Any]) -> None:
    expected = len(selected_payload.get("samples") or [])
    actual = len(workflow_doc.get("workflows") or [])
    if expected and actual != expected:
        raise ValueError(f"Cloud generation returned {actual} workflows for {expected} selected samples.")


def validate_known_tool_entrypoints(workflow_doc: dict[str, Any]) -> None:
    known = tool_names()
    errors = []
    for workflow in workflow_doc.get("workflows") or []:
        workflow_id = workflow.get("workflow_id", "<missing>")
        for node in workflow.get("nodes") or []:
            if node.get("kind") not in {"app", "dnn"}:
                continue
            entrypoint = node.get("entrypoint") or node.get("node_id")
            if entrypoint not in known:
                errors.append(f"{workflow_id}:{node.get('node_id')} unknown tool entrypoint {entrypoint!r}")
    if errors:
        raise ValueError("Unknown executable tool entrypoints: " + "; ".join(errors))
