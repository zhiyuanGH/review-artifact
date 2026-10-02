"""OpenAI-compatible cloud LLM client for Qwen workflow generation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import DEFAULT_QWEN_API_KEY_ENV, DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL


@dataclass(frozen=True)
class CloudLLMConfig:
    provider: str = "qwen"
    model: str = DEFAULT_QWEN_MODEL
    base_url: str = DEFAULT_QWEN_BASE_URL
    api_key_env: str = DEFAULT_QWEN_API_KEY_ENV
    api_key_file: str | None = None
    timeout_seconds: int = 120
    temperature: float = 0.0
    max_tokens: int = 8192

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_file": self.api_key_file or "",
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def require_api_key(config: CloudLLMConfig) -> str:
    api_key = os.getenv(config.api_key_env)
    if not api_key and config.api_key_file:
        api_key = read_api_key_file(Path(config.api_key_file))
    if not api_key:
        raise RuntimeError(
            f"Missing cloud LLM API key. Set environment variable {config.api_key_env} "
            f"or provide api_key in {config.api_key_file or 'a local config file'}."
        )
    return api_key


def read_api_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Cloud config must be a JSON object: {path}")
    value = data.get("api_key")
    return str(value).strip() if value else None


def make_client(config: CloudLLMConfig) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Install it in the active Python environment.") from exc
    return OpenAI(
        api_key=require_api_key(config),
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )


def smoke_test(config: CloudLLMConfig) -> dict[str, Any]:
    started = time.time()
    payload = {
        "status": "error",
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "started_at_unix": started,
        "wall_ms": None,
        "response_format": "json_object",
    }
    try:
        response = chat_completion(
            config,
            system_prompt="Return compact JSON only.",
            user_prompt='Return {"ok": true, "component": "wfgen_qwen_smoke_test"}.',
            response_schema=None,
            prefer_json_schema=False,
            max_tokens=64,
        )
        content = response["content"]
        parsed = json.loads(content)
        raw = response["raw_response"]
        payload.update(
            {
                "status": "ok" if parsed.get("ok") is True else "unexpected_response",
                "parsed": parsed,
                "response_format": response.get("response_format"),
                "usage": extract_usage(raw),
                "response_id": raw.get("id") if isinstance(raw, dict) else "",
                "raw_response": raw,
            }
        )
    except Exception as exc:
        payload.update({"status": "error", "error": safe_error(exc)})
    finally:
        payload["wall_ms"] = (time.time() - started) * 1000.0
    return payload


def chat_completion(
    config: CloudLLMConfig,
    *,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any] | None,
    prefer_json_schema: bool = True,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    client = make_client(config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    errors: list[str] = []

    if prefer_json_schema and response_schema is not None:
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=max_tokens or config.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "wfgen_workflow_spec",
                        "strict": True,
                        "schema": response_schema,
                    },
                },
            )
            return normalize_chat_response(response, "json_schema")
        except Exception as exc:
            errors.append(f"json_schema failed: {safe_error(exc)}")

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=max_tokens or config.max_tokens,
            response_format={"type": "json_object"},
        )
        normalized = normalize_chat_response(response, "json_object")
        if errors:
            normalized["fallback_errors"] = errors
        return normalized
    except Exception as exc:
        errors.append(f"json_object failed: {safe_error(exc)}")
        raise RuntimeError("; ".join(errors)) from exc


def normalize_chat_response(response: Any, response_format: str) -> dict[str, Any]:
    raw = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    content = raw["choices"][0]["message"].get("content") if isinstance(raw, dict) else response.choices[0].message.content
    if not content:
        raise RuntimeError("Cloud LLM response did not contain message content.")
    return {
        "response_format": response_format,
        "content": content,
        "raw_response": raw,
    }


def extract_usage(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    usage = raw.get("usage")
    return usage if isinstance(usage, dict) else {}


def safe_error(exc: Exception) -> str:
    text = str(exc)
    key_prefixes = ("sk-",)
    for prefix in key_prefixes:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[:idx] + prefix + "[redacted]"
    return text
