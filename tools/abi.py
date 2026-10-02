"""Small JSON tool ABI used by MOBench runtime adapters."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


def read_request(input_path: str | None = None) -> dict[str, Any]:
    if input_path:
        text = Path(input_path).read_text(encoding="utf-8")
    elif sys.stdin.isatty():
        text = ""
    else:
        text = sys.stdin.read()
    if not text.strip():
        return {}
    return json.loads(text)


def write_response(response: dict[str, Any], output_path: str | None = None) -> None:
    text = json.dumps(response, indent=2, ensure_ascii=True)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


def ok(
    *,
    artifacts: dict[str, Any],
    raw: dict[str, Any] | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    wall_ms = None if started_at is None else (time.time() - started_at) * 1000.0
    return {
        "success": True,
        "artifacts": artifacts,
        "raw": raw or {},
        "wall_ms": wall_ms,
    }


def fail(
    error: str,
    *,
    raw: dict[str, Any] | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    wall_ms = None if started_at is None else (time.time() - started_at) * 1000.0
    return {
        "success": False,
        "artifacts": {},
        "raw": raw or {},
        "error": error,
        "wall_ms": wall_ms,
    }


def upstream_value(request: dict[str, Any], artifact_name: str) -> Any:
    upstream = request.get("upstream") or {}
    value = upstream.get(artifact_name)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value

