"""Benchmark cloud LLM code generation from prepared workflow prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
for import_root in (PACKAGE_PARENT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from MOBench.workflow_generation import DEFAULT_QWEN_API_KEY_ENV, DEFAULT_QWEN_BASE_URL, DEFAULT_QWEN_MODEL  # noqa: E402
from MOBench.workflow_generation.cloud_llm import CloudLLMConfig, extract_usage, make_client, safe_error  # noqa: E402


DEFAULT_PROMPT_DIR = REPO_ROOT / "results" / "my_run" / "planning" / "codegen_prompts"
DEFAULT_SYSTEM_PROMPT = "You are a careful Python code-generation assistant. Follow the user prompt exactly."


@dataclass(frozen=True)
class CredentialBundle:
    csv_path: Path | None
    values: dict[str, str]

    @property
    def api_key(self) -> str:
        return first_present(self.values, "apiKey", "api_key", "apikey", "key")

    @property
    def base_url(self) -> str:
        return first_present(self.values, "openAiCompatible", "openai_compatible", "base_url", "baseUrl")

    def metadata(self, *, api_key_source: str) -> dict[str, Any]:
        return {
            "csv_path": str(self.csv_path) if self.csv_path else "",
            "api_key_source": api_key_source,
            "has_csv_api_key": bool(self.api_key),
            "has_csv_base_url": bool(self.base_url),
            "workspace_id": first_present(self.values, "workspaceId", "workspace_id"),
            "api_host": first_present(self.values, "apiHost", "api_host"),
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = select_prompt_files(args.prompt_dir, args.prompt_glob, args.include_readme, args.offset, args.limit)
    if not prompts:
        raise SystemExit(f"No prompt files found under {args.prompt_dir}")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.fail_fast and args.workers > 1:
        raise SystemExit("--fail-fast is only supported with --workers 1")

    output_dir = (args.output_dir or default_output_dir()).resolve()
    raw_dir = output_dir / "raw_responses"
    code_dir = output_dir / "generated_code"
    raw_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    credential_csv = args.credentials_csv or default_credentials_csv()
    credentials = load_credentials(credential_csv) if credential_csv else CredentialBundle(None, {})
    base_url = args.base_url or credentials.base_url or DEFAULT_QWEN_BASE_URL
    api_key_source = configure_api_key(args.api_key_env, credentials, require_key=not args.dry_run)

    config = CloudLLMConfig(
        provider=args.provider,
        model=args.model,
        base_url=base_url,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"[codegen-bench] prompt_dir={args.prompt_dir}", flush=True)
    print(f"[codegen-bench] output_dir={output_dir}", flush=True)
    print(
        f"[codegen-bench] model={config.model} base_url={config.base_url} "
        f"prompts={len(prompts)} repeat={args.repeat} dry_run={args.dry_run}",
        flush=True,
    )

    client = None if args.dry_run or args.workers > 1 else make_client(config)
    thread_local = threading.local()
    results: list[dict[str, Any]] = []
    started_unix = time.time()
    tasks = [(repeat_index, prompt_index, prompt_path) for repeat_index in range(args.repeat) for prompt_index, prompt_path in enumerate(prompts)]

    def worker_client() -> Any:
        if args.dry_run:
            return None
        if args.workers == 1:
            return client
        local_client = getattr(thread_local, "client", None)
        if local_client is None:
            local_client = make_client(config)
            thread_local.client = local_client
        return local_client

    def run_task(repeat_index: int, prompt_index: int, prompt_path: Path) -> dict[str, Any]:
        result = None
        if args.resume:
            result = resume_existing_result(
                prompt_path=prompt_path,
                prompt_index=prompt_index,
                repeat_index=repeat_index,
                raw_dir=raw_dir,
                code_dir=code_dir,
                system_prompt=args.system_prompt,
            )
        if result is None:
            result = benchmark_prompt(
                client=worker_client(),
                config=config,
                prompt_path=prompt_path,
                prompt_index=prompt_index,
                repeat_index=repeat_index,
                output_dir=output_dir,
                raw_dir=raw_dir,
                code_dir=code_dir,
                system_prompt=args.system_prompt,
                dry_run=args.dry_run,
                compile_check=not args.skip_compile_check,
            )
        return result

    if args.workers == 1:
        for repeat_index, prompt_index, prompt_path in tasks:
            result = run_task(repeat_index, prompt_index, prompt_path)
            results.append(result)
            print_result(result, args.repeat)
            if args.fail_fast and result["status"] not in {"ok", "dry_run"}:
                break
            if args.inter_request_sleep > 0 and not args.dry_run:
                time.sleep(args.inter_request_sleep)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, int, Path]] = {}
            for task in tasks:
                future_to_task[executor.submit(run_task, *task)] = task
                if args.inter_request_sleep > 0 and not args.dry_run:
                    time.sleep(args.inter_request_sleep)
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                results.append(result)
                print_result(result, args.repeat)

    results.sort(key=lambda item: (int(item.get("repeat_index") or 0), int(item.get("prompt_index") or 0)))

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_ms": (time.time() - started_unix) * 1000.0,
        "status": overall_status(results),
        "cloud_config": {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
            "api_key_env": config.api_key_env,
            "timeout_seconds": config.timeout_seconds,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        },
        "credentials": credentials.metadata(api_key_source=api_key_source),
        "prompt_dir": str(args.prompt_dir.resolve()),
        "prompt_count": len(prompts),
        "repeat": args.repeat,
        "dry_run": args.dry_run,
        "aggregate": aggregate_results(results),
        "results": results,
    }

    json_path = output_dir / "codegen_llm_benchmark.json"
    csv_path = output_dir / "codegen_llm_benchmark.csv"
    md_path = output_dir / "codegen_llm_benchmark.md"
    write_json(json_path, report)
    write_csv(csv_path, results)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[codegen-bench] wrote {json_path}", flush=True)
    print(f"[codegen-bench] wrote {csv_path}", flush=True)
    print(f"[codegen-bench] wrote {md_path}", flush=True)
    return 0 if report["status"] in {"ok", "dry_run"} else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure latency and token usage for LLM-generated executable workflow code.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR, help="Directory containing standalone .txt codegen prompts.")
    parser.add_argument("--prompt-glob", default="*.txt", help="Prompt filename glob inside --prompt-dir.")
    parser.add_argument("--include-readme", action="store_true", help="Include README.txt if it matches --prompt-glob.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many selected prompt files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of prompt files to benchmark.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of times to run each prompt.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for benchmark artifacts.")
    parser.add_argument("--credentials-csv", type=Path, default=None, help="Aliyun API key CSV export. Defaults to the first *-apiKey-*.csv in the repo root.")
    parser.add_argument("--provider", default="qwen", help="Cloud provider label recorded in artifacts.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="OpenAI-compatible model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL. Defaults to the CSV openAiCompatible field, then MOBench's Qwen default.")
    parser.add_argument("--api-key-env", default=DEFAULT_QWEN_API_KEY_ENV, help="Environment variable used by the shared cloud LLM client.")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum output tokens per codegen request.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System message sent with each standalone prompt.")
    parser.add_argument("--inter-request-sleep", type=float, default=0.0, help="Seconds to sleep between real API requests.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent prompt requests. Use 1 for sequential execution.")
    parser.add_argument("--skip-compile-check", action="store_true", help="Do not syntax-compile extracted Python code.")
    parser.add_argument("--dry-run", action="store_true", help="Discover prompts and write artifacts without calling the LLM.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failed prompt.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing completed raw_responses/generated_code artifacts and only run missing or error prompts.")
    return parser.parse_args(argv)


def benchmark_prompt(
    *,
    client: Any,
    config: CloudLLMConfig,
    prompt_path: Path,
    prompt_index: int,
    repeat_index: int,
    output_dir: Path,
    raw_dir: Path,
    code_dir: Path,
    system_prompt: str,
    dry_run: bool,
    compile_check: bool,
) -> dict[str, Any]:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    raw_path, code_path = artifact_paths(prompt_path, prompt_index, repeat_index, raw_dir, code_dir)
    started_unix = time.time()
    started_perf = time.perf_counter()
    base_result: dict[str, Any] = {
        "prompt_file": str(prompt_path),
        "prompt_name": prompt_path.name,
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "prompt_chars": len(prompt_text),
        "prompt_bytes": len(prompt_text.encode("utf-8")),
        "system_prompt_chars": len(system_prompt),
        "request_started_at_unix": started_unix,
        "raw_response_json": "",
        "code_path": "",
    }
    if dry_run:
        result = {
            **base_result,
            "status": "dry_run",
            "wall_ms": 0.0,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "response_id": "",
            "response_content_bytes": 0,
            "code_bytes": 0,
            "code_line_count": 0,
            "code_block_found": False,
            "compile_status": "skipped",
            "compile_error": "",
            "error": "",
        }
        write_json(raw_path, {"status": "dry_run", "prompt_file": str(prompt_path), "prompt_chars": len(prompt_text)})
        result["raw_response_json"] = str(raw_path)
        return result

    raw_response: dict[str, Any] | None = None
    content = ""
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        raw_response = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
        content = extract_message_content(raw_response)
        usage = extract_usage(raw_response)
        code, code_block_found, code_block_language = extract_code_block(content)
        compile_status = "skipped"
        compile_error = ""
        if code:
            code_path.write_text(code, encoding="utf-8")
            if compile_check:
                compile_status, compile_error = compile_python(code, code_path)
            else:
                compile_status = "skipped"
        status = result_status(api_ok=True, code_block_found=code_block_found, compile_status=compile_status)
        wall_ms = (time.perf_counter() - started_perf) * 1000.0
        result = {
            **base_result,
            "status": status,
            "wall_ms": wall_ms,
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
            "usage": usage,
            "response_id": raw_response.get("id", "") if isinstance(raw_response, dict) else "",
            "response_content_bytes": len(content.encode("utf-8")),
            "code_bytes": len(code.encode("utf-8")) if code else 0,
            "code_line_count": len(code.splitlines()) if code else 0,
            "code_block_found": code_block_found,
            "code_block_language": code_block_language,
            "compile_status": compile_status,
            "compile_error": compile_error,
            "error": "",
            "raw_response_json": str(raw_path),
            "code_path": str(code_path) if code else "",
        }
        write_json(
            raw_path,
            {
                "status": status,
                "wall_ms": wall_ms,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "prompt_file": str(prompt_path),
                "prompt_chars": len(prompt_text),
                "usage": usage,
                "response_id": result["response_id"],
                "content": content,
                "raw_response": raw_response,
                "code_path": result["code_path"],
                "compile_status": compile_status,
                "compile_error": compile_error,
            },
        )
        return result
    except Exception as exc:
        wall_ms = (time.perf_counter() - started_perf) * 1000.0
        error = safe_error(exc)
        result = {
            **base_result,
            "status": "error",
            "wall_ms": wall_ms,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "usage": {},
            "response_id": raw_response.get("id", "") if isinstance(raw_response, dict) else "",
            "response_content_bytes": len(content.encode("utf-8")),
            "code_bytes": 0,
            "code_line_count": 0,
            "code_block_found": False,
            "code_block_language": "",
            "compile_status": "skipped",
            "compile_error": "",
            "error": error,
            "raw_response_json": str(raw_path),
            "code_path": "",
        }
        write_json(
            raw_path,
            {
                "status": "error",
                "wall_ms": wall_ms,
                "provider": config.provider,
                "model": config.model,
                "base_url": config.base_url,
                "prompt_file": str(prompt_path),
                "error": error,
                "content": content,
                "raw_response": raw_response,
            },
        )
        return result


def print_result(result: dict[str, Any], repeat_count: int) -> None:
    print(
        "[codegen-bench] "
        f"{result.get('prompt_name', '')} repeat={int(result.get('repeat_index') or 0) + 1}/{repeat_count} "
        f"status={result['status']} wall_ms={float(result.get('wall_ms') or 0.0):.2f} "
        f"prompt_tokens={result.get('prompt_tokens', '')} "
        f"completion_tokens={result.get('completion_tokens', '')} "
        f"total_tokens={result.get('total_tokens', '')}",
        flush=True,
    )


def resume_existing_result(
    *,
    prompt_path: Path,
    prompt_index: int,
    repeat_index: int,
    raw_dir: Path,
    code_dir: Path,
    system_prompt: str,
) -> dict[str, Any] | None:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    raw_path, default_code_path = artifact_paths(prompt_path, prompt_index, repeat_index, raw_dir, code_dir)
    if not raw_path.exists():
        return None
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    status = str(raw.get("status") or "")
    if status in {"", "error"}:
        return None

    usage = raw.get("usage") or {}
    code_path = Path(raw.get("code_path") or default_code_path)
    code = code_path.read_text(encoding="utf-8") if code_path.exists() else ""
    content = str(raw.get("content") or "")
    return {
        "prompt_file": str(prompt_path),
        "prompt_name": prompt_path.name,
        "prompt_index": prompt_index,
        "repeat_index": repeat_index,
        "prompt_chars": len(prompt_text),
        "prompt_bytes": len(prompt_text.encode("utf-8")),
        "system_prompt_chars": len(system_prompt),
        "request_started_at_unix": "",
        "raw_response_json": str(raw_path),
        "code_path": str(code_path) if code else "",
        "status": status,
        "wall_ms": float(raw.get("wall_ms") or 0.0),
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
        "usage": usage,
        "response_id": raw.get("response_id", ""),
        "response_content_bytes": len(content.encode("utf-8")),
        "code_bytes": len(code.encode("utf-8")) if code else 0,
        "code_line_count": len(code.splitlines()) if code else 0,
        "code_block_found": bool(code) or "```" in content,
        "code_block_language": "python" if code else "",
        "compile_status": raw.get("compile_status", ""),
        "compile_error": raw.get("compile_error", ""),
        "error": "",
        "resumed": True,
    }


def artifact_paths(
    prompt_path: Path,
    prompt_index: int,
    repeat_index: int,
    raw_dir: Path,
    code_dir: Path,
) -> tuple[Path, Path]:
    artifact_stem = f"{prompt_index:02d}_r{repeat_index:02d}_{safe_stem(prompt_path.stem)}"
    return raw_dir / f"{artifact_stem}.json", code_dir / f"{artifact_stem}.py"


def select_prompt_files(prompt_dir: Path, prompt_glob: str, include_readme: bool, offset: int, limit: int | None) -> list[Path]:
    if offset < 0:
        raise ValueError("--offset must be non-negative")
    files = sorted(path for path in prompt_dir.glob(prompt_glob) if path.is_file())
    if not include_readme:
        files = [path for path in files if path.name.lower() != "readme.txt"]
    selected = files[offset:]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive when provided")
        selected = selected[:limit]
    return selected


def default_credentials_csv() -> Path | None:
    matches = sorted(REPO_ROOT.glob("*-apiKey-*.csv"))
    return matches[0] if matches else None


def load_credentials(path: Path | None) -> CredentialBundle:
    if path is None:
        return CredentialBundle(None, {})
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Credential CSV not found: {resolved}")
    with resolved.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return CredentialBundle(resolved, {})

    values: dict[str, str] = {}
    if "id" in fieldnames and len(fieldnames) > 1:
        value_columns = [name for name in fieldnames if name != "id"]
        for row in rows:
            key = str(row.get("id") or "").strip()
            value = first_present(row, *value_columns)
            if key and value:
                values[key] = value
    else:
        for row in rows:
            for key, value in row.items():
                if key and value and key not in values:
                    values[key] = str(value).strip()
    return CredentialBundle(resolved, values)


def configure_api_key(api_key_env: str, credentials: CredentialBundle, *, require_key: bool) -> str:
    if os.getenv(api_key_env):
        return f"env:{api_key_env}"
    if credentials.api_key:
        os.environ[api_key_env] = credentials.api_key
        return f"csv:{credentials.csv_path}:apiKey"
    if require_key:
        raise RuntimeError(
            f"Missing API key. Set {api_key_env} or provide --credentials-csv with an apiKey field."
        )
    return ""


def first_present(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_message_content(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        raise RuntimeError("Cloud LLM response did not serialize to a JSON object.")
    choices = raw_response.get("choices") or []
    if not choices:
        raise RuntimeError("Cloud LLM response did not contain choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    if not content:
        raise RuntimeError("Cloud LLM response did not contain message content.")
    return str(content)


def extract_code_block(content: str) -> tuple[str, bool, str]:
    matches = list(re.finditer(r"```(?P<label>[^\n`]*)\n(?P<body>.*?)```", content, flags=re.DOTALL))
    if not matches:
        return content.strip() + "\n" if content.strip() else "", False, ""
    chosen = matches[0]
    for match in matches:
        label = match.group("label").strip().lower()
        if label in {"python", "py"} or label.startswith("python "):
            chosen = match
            break
    label = chosen.group("label").strip()
    code = chosen.group("body").strip()
    return code + "\n" if code else "", True, label


def compile_python(code: str, code_path: Path) -> tuple[str, str]:
    try:
        compile(code, str(code_path), "exec")
        return "ok", ""
    except SyntaxError as exc:
        location = f"{exc.filename}:{exc.lineno}:{exc.offset}" if exc.lineno else str(exc.filename)
        return "error", f"{location}: {exc.msg}"
    except Exception as exc:
        return "error", safe_error(exc)


def result_status(*, api_ok: bool, code_block_found: bool, compile_status: str) -> str:
    if not api_ok:
        return "error"
    if not code_block_found:
        return "no_code_block"
    if compile_status == "error":
        return "compile_error"
    return "ok"


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    token_keys = ["prompt_tokens", "completion_tokens", "total_tokens"]
    wall_values = [float(item.get("wall_ms") or 0.0) for item in results if item.get("status") != "dry_run"]
    request_count = len(results)
    status_counts = Counter(str(item.get("status") or "") for item in results)
    compile_status_counts = Counter(str(item.get("compile_status") or "") for item in results)
    aggregate: dict[str, Any] = {
        "request_count": request_count,
        "ok_count": status_counts.get("ok", 0),
        "dry_run_count": status_counts.get("dry_run", 0),
        "error_count": status_counts.get("error", 0),
        "compile_error_count": status_counts.get("compile_error", 0),
        "no_code_block_count": status_counts.get("no_code_block", 0),
        "status_counts": dict(status_counts),
        "compile_status_counts": dict(compile_status_counts),
        "compile_success_rate": (status_counts.get("ok", 0) / request_count) if request_count else 0.0,
        "compile_failure_rate": (
            (status_counts.get("compile_error", 0) + status_counts.get("no_code_block", 0) + status_counts.get("error", 0))
            / request_count
        )
        if request_count
        else 0.0,
        "wall_ms_total": sum(wall_values),
        "wall_ms_avg": sum(wall_values) / len(wall_values) if wall_values else 0.0,
        "wall_ms_p50": percentile(wall_values, 50),
        "wall_ms_p95": percentile(wall_values, 95),
    }
    for key in token_keys:
        values = [int(item.get(key) or 0) for item in results]
        aggregate[key] = sum(values)
        aggregate[f"avg_{key}"] = sum(values) / len(values) if values else 0.0
    aggregate["response_content_bytes"] = sum(int(item.get("response_content_bytes") or 0) for item in results)
    aggregate["code_bytes"] = sum(int(item.get("code_bytes") or 0) for item in results)
    return aggregate


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def overall_status(results: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status")) for item in results}
    if not results:
        return "empty"
    if statuses == {"dry_run"}:
        return "dry_run"
    if statuses <= {"ok"}:
        return "ok"
    return "partial" if "ok" in statuses else "error"


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate") or {}
    lines = [
        "# Codegen LLM Benchmark",
        "",
        f"Created UTC: `{report.get('created_at_utc', '')}`",
        f"Status: `{report.get('status', '')}`",
        f"Model: `{(report.get('cloud_config') or {}).get('model', '')}`",
        f"Prompt count: `{report.get('prompt_count', 0)}`",
        f"Repeat: `{report.get('repeat', 1)}`",
        "",
        "## Aggregate",
        "",
        f"- Requests: `{aggregate.get('request_count', 0)}`",
        f"- OK: `{aggregate.get('ok_count', 0)}`",
        f"- Compile success rate: `{float(aggregate.get('compile_success_rate', 0.0)):.4f}`",
        f"- Compile errors: `{aggregate.get('compile_error_count', 0)}`",
        f"- API errors: `{aggregate.get('error_count', 0)}`",
        f"- Total wall ms: `{aggregate.get('wall_ms_total', 0):.2f}`",
        f"- Avg wall ms: `{aggregate.get('wall_ms_avg', 0):.2f}`",
        f"- P95 wall ms: `{aggregate.get('wall_ms_p95', 0):.2f}`",
        f"- Prompt tokens: `{int(aggregate.get('prompt_tokens', 0))}`",
        f"- Completion tokens: `{int(aggregate.get('completion_tokens', 0))}`",
        f"- Total tokens: `{int(aggregate.get('total_tokens', 0))}`",
        "",
        "## Requests",
        "",
        "| Prompt | Status | Wall ms | Prompt tok | Completion tok | Total tok | Code bytes | Compile |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in report.get("results") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("prompt_name", "")),
                    str(item.get("status", "")),
                    f"{float(item.get('wall_ms') or 0.0):.2f}",
                    str(item.get("prompt_tokens", "")),
                    str(item.get("completion_tokens", "")),
                    str(item.get("total_tokens", "")),
                    str(item.get("code_bytes", "")),
                    str(item.get("compile_status", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "prompt_name",
        "prompt_file",
        "prompt_index",
        "repeat_index",
        "status",
        "wall_ms",
        "prompt_chars",
        "prompt_bytes",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "response_content_bytes",
        "code_bytes",
        "code_line_count",
        "code_block_found",
        "code_block_language",
        "compile_status",
        "compile_error",
        "error",
        "response_id",
        "raw_response_json",
        "code_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in results:
            writer.writerow(item)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"codegen_llm_benchmark_{stamp}"


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "prompt"


if __name__ == "__main__":
    raise SystemExit(main())
