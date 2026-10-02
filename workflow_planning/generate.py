"""Generate workflow JSON from a benchmark dataset using the OpenAI API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from MOBench.wfgen_core.schema import (
        SYSTEM_PROMPT,
        WORKFLOW_SCHEMA,
        WORKFLOW_SCHEMA_NAME,
        response_text_format,
    )
    from MOBench.workflow_planning.postprocess import hydrate_tool_parameters_from_samples, insert_llm_checker_nodes
except ImportError:
    from wfgen_core.schema import (
        SYSTEM_PROMPT,
        WORKFLOW_SCHEMA,
        WORKFLOW_SCHEMA_NAME,
        response_text_format,
    )
    from workflow_planning.postprocess import hydrate_tool_parameters_from_samples, insert_llm_checker_nodes


TAU_BENCH_REPO = "fuvty/tau-bench-synthetic"
TAU_BENCH_CONFIG = "tasks"
TAU_BENCH_SPLIT = "train"
DEFAULT_CONFIG_PATH = Path("configs/openai.json")
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SAMPLE_COUNT = 20
DEFAULT_MAX_DATASET_CHARS = 200000
DEFAULT_SELECTED_SAMPLES_PATH = Path("data/tau_bench_fixed_20.json")
PLACEHOLDER_API_KEYS = {"", "YOUR_OPENAI_API_KEY", "sk-..."}

TAU_BENCH_EXTRA_INSTRUCTION = """The dataset is fuvty/tau-bench-synthetic/tasks.
Each selected sample includes evaluation_criteria.actions. Treat these actions as the reference ground-truth action sequence for that sample.
Generate exactly one workflow per selected sample.
Preserve each ground-truth action name as an app/tool node entrypoint and preserve action arguments as node parameters.
Use LLM nodes only when the workflow needs reasoning, validation, or user-facing communication beyond a direct tool call.
When a tool output feeds a later tool call, include or allow insertion of an LLM node that reads the tool output, analyzes it, and gates the following tool call.
Infer typed artifacts from action arguments and action outputs. Add edges for true data dependencies, especially when a later action needs an identifier found by an earlier action.
"""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None


@dataclass(frozen=True)
class DatasetPayload:
    source: str
    format: str
    data: Any
    record_count: int | None
    truncated: bool


def load_local_dataset(
    dataset_path: Path,
    *,
    max_records: int | None = None,
    max_chars: int = DEFAULT_MAX_DATASET_CHARS,
) -> tuple[DatasetPayload, str]:
    """Load a dataset and return both structured metadata and prompt-ready text."""

    suffix = dataset_path.suffix.lower()
    truncated = False

    if suffix == ".json":
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        data, truncated = _truncate_records(data, max_records)
        record_count = _count_records(data)
        dataset_format = "json"
    elif suffix == ".jsonl":
        rows = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_records is not None and idx >= max_records:
                    truncated = True
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        data = rows
        record_count = len(rows)
        dataset_format = "jsonl"
    elif suffix == ".csv":
        with dataset_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for idx, row in enumerate(reader):
                if max_records is not None and idx >= max_records:
                    truncated = True
                    break
                rows.append(dict(row))
        data = rows
        record_count = len(rows)
        dataset_format = "csv"
    else:
        text = dataset_path.read_text(encoding="utf-8")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
            truncated = True
        data = {"text": text}
        record_count = None
        dataset_format = "text"

    payload = DatasetPayload(
        source=str(dataset_path),
        format=dataset_format,
        data=data,
        record_count=record_count,
        truncated=truncated,
    )
    dataset_text, char_truncated = _render_dataset_for_prompt(payload, max_chars=max_chars)
    if char_truncated and not payload.truncated:
        payload = DatasetPayload(
            source=payload.source,
            format=payload.format,
            data=payload.data,
            record_count=payload.record_count,
            truncated=True,
        )
    return payload, dataset_text


def load_tau_bench_samples(
    *,
    repo_id: str = TAU_BENCH_REPO,
    config_name: str = TAU_BENCH_CONFIG,
    split: str = TAU_BENCH_SPLIT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_offset: int = 0,
    max_chars: int = DEFAULT_MAX_DATASET_CHARS,
    selected_samples_output: Path | None = DEFAULT_SELECTED_SAMPLES_PATH,
) -> tuple[DatasetPayload, str]:
    """Download tau-bench synthetic tasks and select a deterministic row window."""

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")

    dataset = hf_load_dataset(repo_id, name=config_name, split=split)
    end = sample_offset + sample_count
    if len(dataset) < end:
        raise ValueError(
            f"{repo_id}/{config_name}/{split} has only {len(dataset)} rows; "
            f"cannot select rows {sample_offset}:{end}."
        )

    selected_rows = dataset.select(range(sample_offset, end))
    samples = [
        normalize_tau_bench_task(row, selection_index=sample_offset + idx)
        for idx, row in enumerate(selected_rows)
    ]
    selected_payload = {
        "source_dataset": repo_id,
        "config": config_name,
        "split": split,
        "selection": {
            "method": "fixed_contiguous_window",
            "offset": sample_offset,
            "count": sample_count,
        },
        "samples": samples,
    }

    if selected_samples_output is not None:
        _ensure_parent_dir(selected_samples_output)
        selected_samples_output.write_text(
            json.dumps(selected_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    payload = DatasetPayload(
        source=f"hf://datasets/{repo_id}/{config_name}/{split}",
        format="huggingface/tau-bench-synthetic/tasks",
        data=selected_payload,
        record_count=len(samples),
        truncated=False,
    )
    dataset_text, char_truncated = _render_dataset_for_prompt(
        payload,
        max_chars=max_chars,
    )
    if char_truncated:
        payload = DatasetPayload(
            source=payload.source,
            format=payload.format,
            data=payload.data,
            record_count=payload.record_count,
            truncated=True,
        )
    return payload, dataset_text


def normalize_tau_bench_task(row: dict[str, Any], *, selection_index: int) -> dict[str, Any]:
    """Normalize tau-bench task rows into prompt-friendly structured samples."""

    description = _decode_jsonish(row.get("description"))
    user_scenario = _decode_jsonish(row.get("user_scenario"))
    initial_state = _decode_jsonish(row.get("initial_state"))
    evaluation_criteria = _decode_jsonish(row.get("evaluation_criteria"))

    actions = []
    communicate_info = []
    nl_assertions = []
    if isinstance(evaluation_criteria, dict):
        actions = evaluation_criteria.get("actions") or []
        communicate_info = evaluation_criteria.get("communicate_info") or []
        nl_assertions = evaluation_criteria.get("nl_assertions") or []

    return {
        "selection_index": selection_index,
        "task_id": str(row.get("id", f"row_{selection_index}")),
        "domain": row.get("domain"),
        "description": description,
        "user_scenario": user_scenario,
        "initial_state": initial_state,
        "ground_truth_actions": actions,
        "ground_truth_communicate_info": communicate_info,
        "ground_truth_nl_assertions": nl_assertions,
        "validated": bool(row.get("validated", False)),
    }


def load_llm_config(
    config_path: Path,
    *,
    model_override: str | None = None,
    base_url_override: str | None = None,
    require_api_key: bool,
) -> LLMConfig:
    """Load cloud LLM configuration from JSON plus optional safe overrides."""

    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    elif require_api_key:
        raise FileNotFoundError(
            f"LLM config not found: {config_path}. Copy configs/openai.example.json "
            "to configs/openai.json and fill in your API key."
        )

    api_key = data.get("api_key") or os.getenv("OPENAI_API_KEY")
    api_key_env = data.get("api_key_env")
    if api_key_env and not data.get("api_key"):
        api_key = os.getenv(str(api_key_env))

    config = LLMConfig(
        provider=str(data.get("provider") or "openai").lower(),
        model=model_override or data.get("model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
        api_key=api_key,
        base_url=base_url_override or data.get("base_url") or os.getenv("OPENAI_BASE_URL"),
    )

    if require_api_key and (not config.api_key or config.api_key in PLACEHOLDER_API_KEYS):
        raise ValueError(
            f"Missing API key in {config_path}. Set `api_key` or `api_key_env` in that file."
        )
    return config


def build_user_prompt(
    payload: DatasetPayload,
    dataset_text: str,
    *,
    workflow_count: int | None,
    extra_instruction: str | None,
) -> str:
    """Build the user message sent to the workflow generator model."""

    workflow_count_text = (
        "Infer an appropriate number of workflows from the dataset."
        if workflow_count is None
        else f"Generate exactly {workflow_count} workflow(s)."
    )
    record_count = "unknown" if payload.record_count is None else str(payload.record_count)
    truncation_note = (
        "The dataset content below was truncated; generate workflows from the visible sample and state assumptions."
        if payload.truncated
        else "The dataset content below is complete for this request."
    )
    extra = extra_instruction.strip() if extra_instruction else "No extra user constraints."

    return f"""Dataset source: {payload.source}
Dataset format: {payload.format}
Visible record count: {record_count}
Dataset completeness: {truncation_note}

Task:
- Convert the dataset into ODABench-style executable workflow JSON.
- {workflow_count_text}
- Preserve task intent from the dataset.
- Create a DAG using typed edges. Avoid serial edges when tasks can safely run in parallel.
- Include initial backend and batch-size hints, but keep them as hints for later optimization.
- Prefer app/tool nodes for external actions and llm nodes for reasoning or text generation.
- Extra constraints: {extra}

Dataset content:
{dataset_text}
"""


def generate_workflow_from_payload(
    payload: DatasetPayload,
    dataset_text: str,
    *,
    output_path: Path,
    llm_config: LLMConfig,
    workflow_count: int | None,
    extra_instruction: str | None,
    generation_metadata: dict[str, str],
    dry_run_prompt: bool,
    skip_semantic_validation: bool,
    insert_checker_nodes: bool,
) -> dict:
    """Call OpenAI on a prepared dataset payload and write workflow JSON."""

    user_prompt = build_user_prompt(
        payload,
        dataset_text,
        workflow_count=workflow_count,
        extra_instruction=extra_instruction,
    )

    if dry_run_prompt:
        preview = {
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
            "response_schema": WORKFLOW_SCHEMA,
            "generation_metadata": generation_metadata,
        }
        _ensure_parent_dir(output_path)
        output_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        return preview

    workflow = call_openai_for_workflow(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        provider=llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
    )
    workflow = hydrate_tool_parameters_from_samples(workflow, payload.data)
    if insert_checker_nodes:
        workflow = insert_llm_checker_nodes(workflow)
    workflow["generation_metadata"] = generation_metadata

    if not skip_semantic_validation:
        validate_workflow_semantics(workflow)

    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return workflow


def generate_workflow(
    dataset_path: Path,
    *,
    output_path: Path,
    llm_config: LLMConfig,
    workflow_count: int | None,
    max_records: int | None,
    max_chars: int,
    extra_instruction: str | None,
    generation_metadata: dict[str, str],
    dry_run_prompt: bool,
    skip_semantic_validation: bool,
    insert_checker_nodes: bool,
) -> dict:
    """Load a local dataset, call OpenAI, validate the result, and write workflow JSON."""

    payload, dataset_text = load_local_dataset(
        dataset_path,
        max_records=max_records,
        max_chars=max_chars,
    )
    return generate_workflow_from_payload(
        payload,
        dataset_text,
        output_path=output_path,
        llm_config=llm_config,
        workflow_count=workflow_count,
        extra_instruction=extra_instruction,
        generation_metadata=generation_metadata,
        dry_run_prompt=dry_run_prompt,
        skip_semantic_validation=skip_semantic_validation,
        insert_checker_nodes=insert_checker_nodes,
    )


def call_openai_for_workflow(
    *,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> dict:
    """Call an OpenAI or OpenAI-compatible API with structured JSON output."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai Python package is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    if provider != "openai":
        return call_chat_completions_for_workflow(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        text=response_text_format(),
    )

    output_text = getattr(response, "output_text", None) or _collect_response_text(response)
    if not output_text:
        raise RuntimeError("OpenAI response did not contain output_text.")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model returned invalid JSON: {exc}") from exc


def call_chat_completions_for_workflow(
    *,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> dict:
    """Call an OpenAI-compatible Chat Completions endpoint."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": WORKFLOW_SCHEMA_NAME,
                "strict": True,
                "schema": WORKFLOW_SCHEMA,
            },
        },
    )
    output_text = response.choices[0].message.content
    if not output_text:
        raise RuntimeError("Chat completion response did not contain message content.")
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model returned invalid JSON: {exc}") from exc


def validate_workflow_semantics(workflow: dict) -> None:
    """Run lightweight semantic checks that are useful before interpretation."""

    if workflow.get("schema_version") != "wfgen.v1":
        raise ValueError("schema_version must be wfgen.v1")

    workflows = workflow.get("workflows")
    if not isinstance(workflows, list) or not workflows:
        raise ValueError("workflows must be a non-empty list")

    for wf in workflows:
        workflow_id = wf.get("workflow_id", "<missing>")
        nodes = wf.get("nodes", [])
        edges = wf.get("edges", [])
        node_ids = [node.get("node_id") for node in nodes]
        if any(not node_id for node_id in node_ids):
            raise ValueError(f"{workflow_id}: every node must have node_id")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError(f"{workflow_id}: duplicate node_id detected")

        node_set = set(node_ids)
        for edge in edges:
            src = edge.get("from_node")
            dst = edge.get("to_node")
            if src not in node_set:
                raise ValueError(f"{workflow_id}: edge source {src!r} is not a node")
            if dst not in node_set:
                raise ValueError(f"{workflow_id}: edge target {dst!r} is not a node")
            if src == dst:
                raise ValueError(f"{workflow_id}: self-edge on {src!r}")

        if _has_cycle(node_set, edges):
            raise ValueError(f"{workflow_id}: workflow graph must be acyclic")


def _has_cycle(node_set: set[str], edges: list[dict]) -> bool:
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_set}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_set}
    for edge in edges:
        src = edge["from_node"]
        dst = edge["to_node"]
        outgoing[src].append(dst)
        indegree[dst] += 1

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for dst in outgoing[node_id]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return visited != len(node_set)


def _render_dataset_for_prompt(payload: DatasetPayload, *, max_chars: int) -> tuple[str, bool]:
    rendered = json.dumps(payload.data, indent=2, ensure_ascii=True)
    if max_chars and len(rendered) > max_chars:
        rendered = rendered[:max_chars]
        rendered += "\n... TRUNCATED ..."
        return rendered, True
    return rendered, False


def _decode_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _ensure_parent_dir(path: Path) -> None:
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)


def _truncate_records(data: Any, max_records: int | None) -> tuple[Any, bool]:
    if max_records is None:
        return data, False
    if isinstance(data, list) and len(data) > max_records:
        return data[:max_records], True
    if isinstance(data, dict):
        truncated = False
        new_data = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > max_records:
                new_data[key] = value[:max_records]
                truncated = True
            else:
                new_data[key] = value
        return new_data, truncated
    return data, False


def _count_records(data: Any) -> int | None:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("tasks", "data", "examples", "records", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
    return None


def _collect_response_text(response: Any) -> str | None:
    """Fallback extraction for SDK versions without response.output_text."""

    output = getattr(response, "output", None)
    if not output:
        return None
    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for content_item in content:
            text = getattr(content_item, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks) if chunks else None


def build_generation_metadata(
    *,
    llm_config: LLMConfig,
    input_source: str,
    sample_selection: str,
    selected_samples_output: Path | None,
) -> dict[str, str]:
    """Build non-secret metadata recorded in generated workflow outputs."""

    return {
        "provider": llm_config.provider,
        "model": llm_config.model,
        "base_url": llm_config.base_url or "",
        "api_kind": "responses" if llm_config.provider == "openai" else "chat_completions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_source": input_source,
        "sample_selection": sample_selection,
        "selected_samples_output": str(selected_samples_output) if selected_samples_output else "",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ODABench-style workflow JSON from tau-bench or a local dataset.",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Local dataset path when --dataset-source=local. Supports .json, .jsonl, .csv, and text.",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["tau-bench", "local"],
        default="tau-bench",
        help="Dataset source. Defaults to downloading fuvty/tau-bench-synthetic.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Cloud LLM config JSON. Defaults to configs/openai.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("workflow.generated.json"),
        help="Where to write the generated workflow JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override. Otherwise read from config, OPENAI_MODEL, or default.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--tau-repo",
        default=TAU_BENCH_REPO,
        help=f"Hugging Face dataset repo. Defaults to {TAU_BENCH_REPO}.",
    )
    parser.add_argument(
        "--tau-config",
        default=TAU_BENCH_CONFIG,
        help=f"Hugging Face dataset config. Defaults to {TAU_BENCH_CONFIG}.",
    )
    parser.add_argument(
        "--tau-split",
        default=TAU_BENCH_SPLIT,
        help=f"Hugging Face dataset split. Defaults to {TAU_BENCH_SPLIT}.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of fixed tau-bench samples to select. Defaults to {DEFAULT_SAMPLE_COUNT}.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Starting row for the fixed tau-bench sample window. Defaults to 0.",
    )
    parser.add_argument(
        "--selected-samples-output",
        type=Path,
        default=DEFAULT_SELECTED_SAMPLES_PATH,
        help="Where to save the normalized selected tau-bench samples.",
    )
    parser.add_argument(
        "--no-save-selected-samples",
        action="store_true",
        help="Do not save the selected tau-bench sample JSON.",
    )
    parser.add_argument(
        "--workflow-count",
        type=int,
        default=None,
        help="Exact number of workflows to generate. By default the model infers this.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit dataset rows/items sent to the model.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_DATASET_CHARS,
        help="Maximum rendered dataset characters sent to the model.",
    )
    parser.add_argument(
        "--extra-instruction",
        default=None,
        help="Additional generation constraint, for example target device or available tools.",
    )
    parser.add_argument(
        "--dry-run-prompt",
        action="store_true",
        help="Write the prompt and schema instead of calling the API.",
    )
    parser.add_argument(
        "--skip-semantic-validation",
        action="store_true",
        help="Skip local DAG checks after receiving the model output.",
    )
    parser.add_argument(
        "--no-insert-llm-analysis-nodes",
        "--no-insert-llm-checker-nodes",
        dest="no_insert_llm_analysis_nodes",
        action="store_true",
        help="Do not insert LLM checker nodes after tool outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        llm_config = load_llm_config(
            args.config,
            model_override=args.model,
            base_url_override=args.base_url,
            require_api_key=not args.dry_run_prompt,
        )

        if args.dataset_source == "tau-bench":
            selected_samples_output = None if args.no_save_selected_samples else args.selected_samples_output
            payload, dataset_text = load_tau_bench_samples(
                repo_id=args.tau_repo,
                config_name=args.tau_config,
                split=args.tau_split,
                sample_count=args.sample_count,
                sample_offset=args.sample_offset,
                max_chars=args.max_chars,
                selected_samples_output=selected_samples_output,
            )
            workflow_count = args.workflow_count or args.sample_count
            extra_instruction = _join_instructions(
                TAU_BENCH_EXTRA_INSTRUCTION,
                args.extra_instruction,
            )
            generation_metadata = build_generation_metadata(
                llm_config=llm_config,
                input_source=payload.source,
                sample_selection=(
                    f"fixed_contiguous_window offset={args.sample_offset} "
                    f"count={args.sample_count}"
                ),
                selected_samples_output=selected_samples_output,
            )
            generate_workflow_from_payload(
                payload,
                dataset_text,
                output_path=args.output,
                llm_config=llm_config,
                workflow_count=workflow_count,
                extra_instruction=extra_instruction,
                generation_metadata=generation_metadata,
                dry_run_prompt=args.dry_run_prompt,
                skip_semantic_validation=args.skip_semantic_validation,
                insert_checker_nodes=not args.no_insert_llm_analysis_nodes,
            )
        else:
            if args.dataset is None:
                raise ValueError("A local dataset path is required when --dataset-source=local")
            generation_metadata = build_generation_metadata(
                llm_config=llm_config,
                input_source=str(args.dataset),
                sample_selection=f"local max_records={args.max_records}",
                selected_samples_output=None,
            )
            generate_workflow(
                args.dataset,
                output_path=args.output,
                llm_config=llm_config,
                workflow_count=args.workflow_count,
                max_records=args.max_records,
                max_chars=args.max_chars,
                extra_instruction=args.extra_instruction,
                generation_metadata=generation_metadata,
                dry_run_prompt=args.dry_run_prompt,
                skip_semantic_validation=args.skip_semantic_validation,
                insert_checker_nodes=not args.no_insert_llm_analysis_nodes,
            )
    except Exception as exc:
        print(f"wfgen: error: {exc}", file=sys.stderr)
        return 1

    mode = "prompt preview" if args.dry_run_prompt else "workflow JSON"
    print(f"wfgen: wrote {mode} to {args.output}")
    return 0


def _join_instructions(*instructions: str | None) -> str | None:
    clean = [instruction.strip() for instruction in instructions if instruction and instruction.strip()]
    return "\n\n".join(clean) if clean else None


if __name__ == "__main__":
    raise SystemExit(main())
