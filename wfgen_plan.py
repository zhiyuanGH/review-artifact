"""MOBench stage 1: workflow planning.

This script is the user-facing entrypoint for selecting benchmark prompts,
generating logical workflow JSON, and writing planned workflow visualizations
under <run-dir>/planning.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from MOBench.workflow_planning.generate import (  # noqa: E402
    TAU_BENCH_CONFIG,
    TAU_BENCH_REPO,
    TAU_BENCH_SPLIT,
    normalize_tau_bench_task,
)
from MOBench.deployment_planning.interpreter import sanitize_identifier  # noqa: E402
from MOBench.workflow_planning.tau_bench_builder import build_workflow_document  # noqa: E402
from MOBench.visualizations.visualize_workflow import render_html  # noqa: E402
from MOBench.workflow_generation import (  # noqa: E402
    DEFAULT_QWEN_API_KEY_ENV,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
)
from MOBench.workflow_generation.artifacts import write_json  # noqa: E402
from MOBench.workflow_generation.cloud_llm import CloudLLMConfig, smoke_test  # noqa: E402
from MOBench.workflow_generation.compact_prompt_builder import (  # noqa: E402
    build_compact_prompt_bundle,
    build_compact_sample_prompt_bundle,
    write_compact_prompt_artifacts,
)
from MOBench.workflow_generation.json_generator import (  # noqa: E402
    generate_cloud_workflow_json,
    generate_compact_chain_workflow_json,
    merge_workflow_documents,
)
from MOBench.workflow_generation.prompt_builder import build_prompt_bundle, build_sample_prompt_bundle, write_prompt_artifacts  # noqa: E402
from MOBench.workflow_generation.sample_selector import select_tau_bench_samples  # noqa: E402
from MOBench.workflow_planning import PLANNING_STAGE_DIR  # noqa: E402


ENTRYPOINT_NAME = "workflow-planning"
ENTRYPOINT_STAGES = ["planning", "generation", "visualization"]
DEFAULT_CLOUD_CONFIG_PATH = Path("configs/qwen.local.json")

CATEGORY_PRIORITY = [
    "retail_cancel_order",
    "retail_return_items",
    "retail_exchange_multi_item",
    "retail_modify_multi_item",
    "retail_payment_update",
    "retail_user_address_update",
    "retail_compound_address_item_update",
    "airline_cancel_reservation",
    "airline_update_flights",
    "airline_update_baggages",
    "retail_order_address_update",
    "retail_exchange_single_item",
    "retail_modify_single_item",
    "retail_compound_cancel_return",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir, args.sample_count)
    output_dir = run_dir / PLANNING_STAGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plan] run_dir={run_dir}", flush=True)
    print(f"[plan] output_dir={output_dir}", flush=True)
    print(
        f"[plan] generation_mode={args.generation_mode} prompt_builder={args.prompt_builder} "
        f"sample_count={args.sample_count}",
        flush=True,
    )

    selected_payload, selection = select_tau_bench_samples(
        sample_count=args.sample_count,
        selection_strategy=args.selection_strategy,
        sample_offset=args.sample_offset,
        seed=args.seed,
        include_empty_actions=args.include_empty_actions,
        tau_repo=args.tau_repo,
        tau_config=args.tau_config,
        tau_split=args.tau_split,
        selected_samples=args.selected_samples,
    )
    print(
        f"[plan] selected {len(selected_payload.get('samples') or [])} sample(s) "
        f"method={selection.get('method')} indices={selection.get('indices')}",
        flush=True,
    )
    generation_artifacts: dict[str, Any] = {}

    selected_path = output_dir / "selected_samples.json"
    write_json(selected_path, selected_payload)
    print(f"[plan] wrote selected samples: {selected_path}", flush=True)
    generation_artifacts["selected_samples"] = str(selected_path)
    prompt_bundle = build_combined_prompt_bundle(selected_payload, args)
    prompt_bundle = write_planning_prompt_artifacts(prompt_bundle, output_dir, args)
    generation_artifacts["prompts"] = prompt_bundle.artifacts
    print(f"[plan] wrote combined prompt artifacts: {output_dir / 'prompts'}", flush=True)

    if args.generation_mode == "cloud":
        workflow_doc, cloud_artifacts = generate_cloud_workflow(output_dir, selected_payload, args)
        generation_artifacts.update(cloud_artifacts)
    else:
        workflow_doc = build_workflow_document(selected_payload)
        workflow_doc["generation_metadata"] = {
            "generation_mode": "deterministic",
            "prompt_builder": args.prompt_builder,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

    workflow_path = output_dir / "workflow.json"
    write_json(workflow_path, workflow_doc, ensure_ascii=True)
    generation_artifacts["workflow_json"] = str(workflow_path)
    print(f"[plan] wrote workflow JSON: {workflow_path}", flush=True)

    visualization_paths = []
    if not args.no_visualize:
        visualization_paths = render_visualizations(workflow_doc, output_dir / "visualizations")
        print(f"[plan] rendered {len(visualization_paths)} planned visualization(s)", flush=True)

    summary_path = write_summary(output_dir, selected_payload, workflow_doc, visualization_paths, args)
    manifest_path = write_manifest(
        output_dir,
        run_dir,
        args,
        selection,
        selected_path,
        workflow_path,
        visualization_paths,
        summary_path,
        generation_artifacts,
    )

    print(f"wfgen-plan: wrote selected samples to {selected_path}")
    print(f"wfgen-plan: wrote workflow JSON to {workflow_path}")
    if visualization_paths:
        print(f"wfgen-plan: wrote {len(visualization_paths)} visualizations to {output_dir / 'visualizations'}")
    print(f"wfgen-plan: wrote manifest to {manifest_path}")
    return 0


def generate_cloud_workflow(
    output_dir: Path,
    selected_payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cloud_config_path = resolve_cloud_config_path(args.cloud_config)
    local_cloud_config = read_local_cloud_config(cloud_config_path)
    base_url = args.base_url
    model = args.model
    if local_cloud_config and args.base_url == DEFAULT_QWEN_BASE_URL:
        base_url = str(
            local_cloud_config.get("base_url")
            or local_cloud_config.get("openAiCompatible")
            or local_cloud_config.get("openai_compatible")
            or base_url
        )
    if local_cloud_config and args.model == DEFAULT_QWEN_MODEL and local_cloud_config.get("model"):
        model = str(local_cloud_config["model"])
    cloud_config = CloudLLMConfig(
        provider=args.cloud_provider,
        model=model,
        base_url=base_url,
        api_key_env=args.api_key_env,
        api_key_file=str(cloud_config_path) if cloud_config_path else None,
        timeout_seconds=args.cloud_timeout,
        temperature=args.cloud_temperature,
        max_tokens=args.cloud_max_tokens,
    )
    artifacts: dict[str, Any] = {
        "cloud_config": cloud_config.metadata(),
        "samples": [],
    }

    smoke_path = output_dir / "llm" / "qwen_smoke_test.json"
    if args.skip_api_smoke_test:
        print("[cloud-llm] skipping Qwen smoke test", flush=True)
        smoke_payload = {
            "status": "skipped",
            "reason": "--skip-api-smoke-test",
            "provider": cloud_config.provider,
            "model": cloud_config.model,
            "base_url": cloud_config.base_url,
            "api_key_env": cloud_config.api_key_env,
        }
    else:
        print(
            f"[cloud-llm] smoke test start provider={cloud_config.provider} "
            f"model={cloud_config.model} base_url={cloud_config.base_url}",
            flush=True,
        )
        smoke_payload = smoke_test(cloud_config)
    write_json(smoke_path, smoke_payload)
    print(
        f"[cloud-llm] smoke test status={smoke_payload.get('status')} "
        f"wall_ms={float(smoke_payload.get('wall_ms') or 0.0):.2f} artifact={smoke_path}",
        flush=True,
    )
    artifacts["qwen_smoke_test"] = str(smoke_path)
    if smoke_payload.get("status") != "ok" and not args.skip_api_smoke_test:
        raise RuntimeError(f"Qwen API smoke test failed; see {smoke_path}")

    per_sample_docs = []
    samples = selected_payload.get("samples") or []
    for sample_idx, sample in enumerate(samples):
        sample_label = sample_artifact_label(sample_idx, sample)
        sample_dir = output_dir / "samples" / sample_label
        sample_payload = {
            "source_dataset": selected_payload.get("source_dataset"),
            "config": selected_payload.get("config"),
            "split": selected_payload.get("split"),
            "selection": {
                "method": "single_sample_cloud_generation",
                "count": 1,
                "indices": [sample.get("selection_index")],
            },
            "samples": [sample],
        }
        prompt = build_sample_prompt(selected_payload, sample, args)
        prompt = write_planning_prompt_artifacts(prompt, sample_dir, args)
        print(
            f"[plan] sample {sample_idx + 1}/{len(samples)} wrote prompt artifacts: {sample_dir / 'prompts'} "
            f"user_prompt_chars={len(prompt.user_prompt_text)} catalog_entries={len(prompt.tool_list)}",
            flush=True,
        )
        if args.prompt_builder == "compact-chain":
            sample_doc, sample_llm_artifacts = generate_compact_chain_workflow_json(
                config=cloud_config,
                prompt=prompt,
                selected_payload=sample_payload,
                output_dir=sample_dir,
                insert_checker_nodes=not args.no_insert_llm_checker_nodes,
                call_label=f"sample {sample_idx + 1}/{len(samples)} selection_index={sample.get('selection_index')}",
            )
        else:
            sample_doc, sample_llm_artifacts = generate_cloud_workflow_json(
                config=cloud_config,
                prompt=prompt,
                selected_payload=sample_payload,
                output_dir=sample_dir,
                insert_checker_nodes=not args.no_insert_llm_checker_nodes,
                call_label=f"sample {sample_idx + 1}/{len(samples)} selection_index={sample.get('selection_index')}",
            )
        per_sample_docs.append(sample_doc)
        artifacts["samples"].append(
            {
                "sample_index": sample_idx,
                "selection_index": sample.get("selection_index"),
                "task_id": sample.get("task_id"),
                "artifact_dir": str(sample_dir),
                "prompts": prompt.artifacts,
                **sample_llm_artifacts,
            }
        )

    workflow_doc = merge_workflow_documents(
        per_sample_docs,
        selected_payload,
        provider=cloud_config.provider,
        model=cloud_config.model,
        base_url=cloud_config.base_url,
    )
    workflow_doc.setdefault("generation_metadata", {})["prompt_builder"] = args.prompt_builder
    return workflow_doc, artifacts


def build_combined_prompt_bundle(selected_payload: dict[str, Any], args: argparse.Namespace):
    if args.prompt_builder == "compact-chain":
        return build_compact_prompt_bundle(
            selected_payload,
            workflow_count=len(selected_payload.get("samples") or []),
            extra_instruction=args.extra_instruction,
        )
    return build_prompt_bundle(
        selected_payload,
        workflow_count=len(selected_payload.get("samples") or []),
        extra_instruction=args.extra_instruction,
    )


def build_sample_prompt(selected_payload: dict[str, Any], sample: dict[str, Any], args: argparse.Namespace):
    if args.prompt_builder == "compact-chain":
        return build_compact_sample_prompt_bundle(
            selected_payload,
            sample,
            extra_instruction=args.extra_instruction,
        )
    return build_sample_prompt_bundle(
        selected_payload,
        sample,
        extra_instruction=args.extra_instruction,
    )


def write_planning_prompt_artifacts(prompt, output_dir: Path, args: argparse.Namespace):
    if args.prompt_builder == "compact-chain":
        return write_compact_prompt_artifacts(prompt, output_dir)
    return write_prompt_artifacts(prompt, output_dir)


def read_local_cloud_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sample_artifact_label(sample_idx: int, sample: dict[str, Any]) -> str:
    selection_index = sample.get("selection_index", sample_idx)
    domain = sample.get("domain") or "sample"
    return f"{sample_idx:02d}_{domain}_{selection_index}"


def resolve_cloud_config_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PACKAGE_ROOT / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MOBench stage 1: workflow planning. Select prompts, generate logical "
            "workflow JSON, and render planned HTML under <run-dir>/planning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run root. Planning artifacts are written under <run-dir>/planning.")
    parser.add_argument("--generation-mode", choices=["cloud", "deterministic"], default="cloud", help="Workflow generation backend.")
    parser.add_argument("--prompt-builder", choices=["full", "compact-chain"], default="full", help="Prompt builder used for cloud planning.")
    parser.add_argument("--cloud-provider", default="qwen", help="Cloud LLM provider label recorded in metadata.")
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="Cloud LLM model used for workflow JSON generation.")
    parser.add_argument("--base-url", default=DEFAULT_QWEN_BASE_URL, help="OpenAI-compatible cloud LLM base URL.")
    parser.add_argument("--api-key-env", default=DEFAULT_QWEN_API_KEY_ENV, help="Environment variable containing the cloud LLM API key.")
    parser.add_argument("--cloud-config", type=Path, default=DEFAULT_CLOUD_CONFIG_PATH, help="Optional local JSON config with api_key. This file should be git-ignored.")
    parser.add_argument("--cloud-timeout", type=int, default=120, help="Cloud LLM request timeout in seconds.")
    parser.add_argument("--cloud-temperature", type=float, default=0.0, help="Cloud LLM generation temperature.")
    parser.add_argument("--cloud-max-tokens", type=int, default=8192, help="Cloud LLM max tokens for workflow JSON generation.")
    parser.add_argument("--extra-instruction", default=None, help="Additional instruction appended to the cloud generation prompt.")
    parser.add_argument("--skip-api-smoke-test", action="store_true", help="Skip the Qwen smoke test before workflow generation.")
    parser.add_argument("--no-insert-llm-checker-nodes", action="store_true", help="Do not add LLM checker nodes after generated tool nodes.")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of prompts/workflows to select.")
    parser.add_argument(
        "--selection-strategy",
        choices=["diverse", "fixed"],
        default="diverse",
        help="Use automatic diverse coverage or a fixed contiguous window.",
    )
    parser.add_argument("--sample-offset", type=int, default=0, help="Starting row for fixed selection.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic tie-breaking seed.")
    parser.add_argument("--include-empty-actions", action="store_true", help="Allow samples with no tool actions.")
    parser.add_argument("--tau-repo", default=TAU_BENCH_REPO)
    parser.add_argument("--tau-config", default=TAU_BENCH_CONFIG)
    parser.add_argument("--tau-split", default=TAU_BENCH_SPLIT)
    parser.add_argument("--selected-samples", type=Path, default=None, help="Use an existing selected_samples.json instead of loading tau-bench.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip planned workflow HTML visualizations.")
    return parser.parse_args(argv)


def select_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.selected_samples:
        payload = json.loads(args.selected_samples.read_text(encoding="utf-8"))
        samples = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError(f"--selected-samples does not contain a samples list: {args.selected_samples}")
        samples = [sample for sample in samples if isinstance(sample, dict)]
        if not samples:
            raise ValueError(f"--selected-samples contains no samples: {args.selected_samples}")
        if args.sample_count and len(samples) > args.sample_count:
            samples = samples[: args.sample_count]
        selection = dict(payload.get("selection") or {}) if isinstance(payload, dict) else {}
        selection.update(
            {
                "method": selection.get("method", "existing_selected_samples"),
                "count": len(samples),
                "indices": [sample.get("selection_index") for sample in samples],
                "source_file": str(args.selected_samples),
            }
        )
        return samples, selection
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Missing dependency: datasets. Install it in the active Python environment.") from exc

    dataset = load_dataset(args.tau_repo, name=args.tau_config, split=args.tau_split)
    normalized = [
        normalize_tau_bench_task(row, selection_index=idx)
        for idx, row in enumerate(dataset)
    ]

    if args.selection_strategy == "fixed":
        end = args.sample_offset + args.sample_count
        if args.sample_offset < 0 or end > len(normalized):
            raise ValueError(f"Fixed window {args.sample_offset}:{end} is outside dataset size {len(normalized)}")
        samples = [annotate_sample(sample, "fixed_window") for sample in normalized[args.sample_offset:end]]
        return samples, {
            "method": "fixed_contiguous_window",
            "offset": args.sample_offset,
            "count": len(samples),
            "indices": [sample["selection_index"] for sample in samples],
        }

    samples = select_diverse(normalized, args.sample_count, seed=args.seed, include_empty=args.include_empty_actions)
    return samples, {
        "method": "auto_diverse",
        "count": len(samples),
        "indices": [sample["selection_index"] for sample in samples],
        "seed": args.seed,
        "criteria": [
            "maximize domain and action-sequence coverage",
            "prefer executable samples with ground-truth tool actions",
            "choose deterministic row positions for reproducibility",
        ],
    }


def select_diverse(
    samples: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    include_empty: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates = []
    for sample in samples:
        actions = sample.get("ground_truth_actions") or []
        if not actions and not include_empty:
            continue
        category = category_for_sample(sample)
        annotated = annotate_sample(sample, category)
        grouped[category].append(annotated)
        candidates.append(annotated)

    for group_samples in grouped.values():
        group_samples.sort(key=lambda item: item["selection_index"])
        if seed:
            rng.shuffle(group_samples)

    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for category in CATEGORY_PRIORITY:
        if len(selected) >= count:
            break
        if not grouped.get(category):
            continue
        add_if_new(selected, seen_indices, grouped[category][0])

    remaining_categories = sorted(
        (category for category in grouped if category not in CATEGORY_PRIORITY),
        key=lambda category: (grouped[category][0].get("domain", ""), category),
    )
    for category in remaining_categories:
        if len(selected) >= count:
            break
        add_if_new(selected, seen_indices, grouped[category][0])

    candidates.sort(key=lambda item: item["selection_index"])
    if seed:
        rng.shuffle(candidates)
    for sample in candidates:
        if len(selected) >= count:
            break
        add_if_new(selected, seen_indices, sample)

    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} candidate samples available for requested count {count}")

    return selected


def add_if_new(selected: list[dict[str, Any]], seen_indices: set[int], sample: dict[str, Any]) -> None:
    idx = int(sample["selection_index"])
    if idx in seen_indices:
        return
    seen_indices.add(idx)
    selected.append(sample)


def category_for_sample(sample: dict[str, Any]) -> str:
    domain = sample.get("domain") or "unknown"
    actions = [action.get("name", "") for action in sample.get("ground_truth_actions") or []]
    action_set = set(actions)
    product_reads = actions.count("get_product_details")

    if domain == "airline":
        if "update_reservation_flights" in action_set:
            return "airline_update_flights"
        if "update_reservation_baggages" in action_set:
            return "airline_update_baggages"
        if "cancel_order" in action_set:
            return "airline_cancel_reservation"
        return "airline_other"

    if "cancel_order" in action_set and "return_delivered_order_items" in action_set:
        return "retail_compound_cancel_return"
    if "modify_pending_order_address" in action_set and "modify_pending_order_items" in action_set:
        return "retail_compound_address_item_update"
    if "exchange_delivered_order_items" in action_set:
        return "retail_exchange_multi_item" if product_reads >= 2 else "retail_exchange_single_item"
    if "modify_pending_order_items" in action_set:
        return "retail_modify_multi_item" if product_reads >= 2 else "retail_modify_single_item"
    if "return_delivered_order_items" in action_set:
        return "retail_return_items"
    if "cancel_order" in action_set:
        return "retail_cancel_order"
    if "modify_pending_order_payment" in action_set:
        return "retail_payment_update"
    if "modify_pending_order_address" in action_set:
        return "retail_order_address_update"
    if "modify_user_address" in action_set:
        return "retail_user_address_update"
    if not actions:
        return f"{domain}_no_actions"
    return f"{domain}_other"


def annotate_sample(sample: dict[str, Any], category: str) -> dict[str, Any]:
    result = json.loads(json.dumps(sample, ensure_ascii=False))
    actions = [action.get("name", "") for action in result.get("ground_truth_actions") or []]
    result["selection_reason"] = category.replace("_", " ")
    result["diversity_tags"] = [
        str(result.get("domain") or "unknown"),
        category,
        "actions:" + "->".join(actions),
    ]
    return result


def render_visualizations(workflow_doc: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, workflow in enumerate(workflow_doc.get("workflows", [])):
        name = sanitize_identifier(workflow.get("name") or workflow.get("workflow_id") or f"workflow_{idx}")
        output_path = output_dir / f"{idx:02d}_{name}.html"
        output_path.write_text(render_html(workflow, workflow_doc, None), encoding="utf-8")
        paths.append(output_path)
    return paths


def write_summary(
    output_dir: Path,
    selected_payload: dict[str, Any],
    workflow_doc: dict[str, Any],
    visualization_paths: list[Path],
    args: argparse.Namespace,
) -> Path:
    rows = []
    for sample, workflow in zip(selected_payload["samples"], workflow_doc.get("workflows", []), strict=False):
        actions = [action.get("name", "") for action in sample.get("ground_truth_actions") or []]
        prompt = (((sample.get("user_scenario") or {}).get("instructions") or {}).get("reason_for_call") or "").replace("\n", " ")
        rows.append(
            "| "
            + " | ".join(
                [
                    str(sample.get("selection_index")),
                    str(sample.get("domain")),
                    str(sample.get("selection_reason")),
                    str(workflow.get("workflow_id")),
                    " -> ".join(actions),
                    prompt[:180],
                ]
            )
            + " |"
        )
    summary = [
        "# MOBench Plan Results",
        "",
        f"Entrypoint: `{ENTRYPOINT_NAME}`",
        f"Stages: `{' -> '.join(ENTRYPOINT_STAGES)}`",
        f"Generation mode: `{args.generation_mode}`",
        f"Prompt builder: `{args.prompt_builder}`",
        f"Generated at UTC: `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`",
        f"Selection method: `{selected_payload['selection']['method']}`",
        f"Sample count: `{len(selected_payload['samples'])}`",
        f"Workflow count: `{len(workflow_doc.get('workflows', []))}`",
        f"Visualizations: `{len(visualization_paths)}`",
        "",
        "Next stage: run `python wfgen_deploy.py --run-dir <run-dir>` to assign deployment backends and compile runners.",
        "",
        "| Index | Domain | Selection reason | Workflow | Actions | Prompt preview |",
        "| ---: | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    path = output_dir / "PLAN_SUMMARY.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    return path


def write_manifest(
    output_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    selection: dict[str, Any],
    selected_path: Path,
    workflow_path: Path,
    visualization_paths: list[Path],
    summary_path: Path,
    generation_artifacts: dict[str, Any],
) -> Path:
    cloud_config_path = resolve_cloud_config_path(args.cloud_config)
    manifest = {
        "entrypoint": ENTRYPOINT_NAME,
        "pipeline_stages": ENTRYPOINT_STAGES,
        "run_dir": str(run_dir),
        "stage_dir": str(output_dir),
        "generation_mode": args.generation_mode,
        "prompt_builder": args.prompt_builder,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sample_count": args.sample_count,
        "selection_strategy": args.selection_strategy,
        "selection": selection,
        "cloud": {
            "provider": args.cloud_provider,
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "api_key_file": str(cloud_config_path) if cloud_config_path else "",
            "timeout_seconds": args.cloud_timeout,
            "temperature": args.cloud_temperature,
            "max_tokens": args.cloud_max_tokens,
        } if args.generation_mode == "cloud" else None,
        "generation_artifacts": generation_artifacts,
        "artifacts": {
            "selected_samples": str(selected_path),
            "workflow_json": str(workflow_path),
            "visualizations": [str(path) for path in visualization_paths],
            "summary": str(summary_path),
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def resolve_run_dir(explicit: Path | None, sample_count: int) -> Path:
    if explicit:
        return explicit.resolve()
    return default_run_dir(sample_count)


def default_run_dir(sample_count: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PACKAGE_ROOT / "results" / f"wfgen_{stamp}_n{sample_count}").resolve()


if __name__ == "__main__":
    raise SystemExit(main())
